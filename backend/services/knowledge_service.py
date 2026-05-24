"""知识库上传、解析、分片和向量索引服务。"""

import shutil
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.tables import KnowledgeChunk
from database.tables import KnowledgeFile
from memory.store import vector_memory
from tools.file_tool import FileTool
from utils.common import generate_uuid
from utils.common import safe_file_name
from utils.common import split_text
from utils.document_classifier import classify_document
from utils.exception import ParameterException


class KnowledgeService:
    """企业知识库入库服务。"""

    allowed_suffixes = {".pdf", ".txt", ".docx"}

    def __init__(self, db_session: Session) -> None:
        self._db_session = db_session
        self._file_tool = FileTool()

    async def upload_file(
        self,
        upload_file: UploadFile,
        upload_user_id: int = 1,
    ) -> dict[str, object]:
        """上传并索引一个知识库文件。

        参数:
            upload_file: FastAPI 上传文件。
            upload_user_id: 用户标识。

        返回:
            文件与分片元数据。

        异常:
            ParameterException: 文件类型或大小不合法时抛出。
        """
        file_suffix = Path(upload_file.filename or "").suffix.lower()
        self._validate_upload(file_suffix, upload_file.size or 0)
        save_path = await self._save_upload_file(upload_file)
        tool_result = self._file_tool.run({"file_path": str(save_path)})
        chunks = split_text(tool_result.content)
        file_record = self._create_file_record(
            upload_file=upload_file,
            save_path=save_path,
            upload_user_id=upload_user_id,
        )
        document_category = classify_document(
            file_record.file_name,
            "\n".join(chunks[:3]),
        )
        self._create_chunks(
            file_record=file_record,
            chunks=chunks,
            document_category=document_category,
        )
        self._db_session.commit()
        return {
            "file_id": file_record.id,
            "file_name": file_record.file_name,
            "document_category": document_category,
            "chunk_count": len(chunks),
        }

    def ingest_local_file(
        self,
        source_path: Path,
        upload_user_id: int = 1,
    ) -> dict[str, object]:
        """索引 Gradio 界面选择的本地文件。

        参数:
            source_path: 本地源文件路径。
            upload_user_id: 用户标识。

        返回:
            文件与分片元数据。

        异常:
            ParameterException: 源文件不合法时抛出。
        """
        if not source_path.exists() or not source_path.is_file():
            raise ParameterException("上传文件不存在")
        self._validate_upload(source_path.suffix.lower(), source_path.stat().st_size)
        save_path = self._copy_local_file(source_path)
        tool_result = self._file_tool.run({"file_path": str(save_path)})
        chunks = split_text(tool_result.content)
        file_record = self._create_local_file_record(
            source_path=source_path,
            save_path=save_path,
            upload_user_id=upload_user_id,
        )
        document_category = classify_document(
            file_record.file_name,
            "\n".join(chunks[:3]),
        )
        self._create_chunks(
            file_record=file_record,
            chunks=chunks,
            document_category=document_category,
        )
        self._db_session.commit()
        return {
            "file_id": file_record.id,
            "file_name": file_record.file_name,
            "document_category": document_category,
            "chunk_count": len(chunks),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """检索已索引的知识分片。

        参数:
            query: 查询文本。
            top_k: 最大返回结果数量。

        返回:
            本地向量记忆中的匹配分片。

        异常:
            无。
        """
        return vector_memory.search(query=query, top_k=top_k)

    def _validate_upload(self, file_suffix: str, file_size: int) -> None:
        """校验上传文件元数据。"""
        settings = get_settings()
        if file_suffix not in self.allowed_suffixes:
            raise ParameterException("仅允许上传PDF、TXT、DOCX文件")
        max_size = settings.max_upload_mb * 1024 * 1024
        if file_size > max_size:
            raise ParameterException("上传文件超过大小限制")

    async def _save_upload_file(self, upload_file: UploadFile) -> Path:
        """将上传文件保存到本地存储。"""
        settings = get_settings()
        original_name = upload_file.filename or "upload"
        file_name = f"{generate_uuid()}_{safe_file_name(original_name)}"
        save_path = settings.upload_dir / file_name
        file_bytes = await upload_file.read()
        save_path.write_bytes(file_bytes)
        return save_path

    def _copy_local_file(self, source_path: Path) -> Path:
        """将本地文件复制到上传目录。"""
        settings = get_settings()
        file_name = f"{generate_uuid()}_{safe_file_name(source_path.name)}"
        save_path = settings.upload_dir / file_name
        shutil.copy2(source_path, save_path)
        return save_path

    def _create_file_record(
        self,
        upload_file: UploadFile,
        save_path: Path,
        upload_user_id: int,
    ) -> KnowledgeFile:
        """创建知识库文件数据库记录。"""
        file_size = int(save_path.stat().st_size / 1024)
        file_record = KnowledgeFile(
            file_name=upload_file.filename or save_path.name,
            file_suffix=save_path.suffix.replace(".", ""),
            file_size=file_size,
            save_path=str(save_path),
            upload_user_id=upload_user_id,
        )
        self._db_session.add(file_record)
        self._db_session.flush()
        return file_record

    def _create_local_file_record(
        self,
        source_path: Path,
        save_path: Path,
        upload_user_id: int,
    ) -> KnowledgeFile:
        """为本地界面上传创建知识库文件记录。"""
        file_size = int(save_path.stat().st_size / 1024)
        file_record = KnowledgeFile(
            file_name=source_path.name,
            file_suffix=source_path.suffix.replace(".", ""),
            file_size=file_size,
            save_path=str(save_path),
            upload_user_id=upload_user_id,
        )
        self._db_session.add(file_record)
        self._db_session.flush()
        return file_record

    def _create_chunks(
        self,
        file_record: KnowledgeFile,
        chunks: list[str],
        document_category: str,
    ) -> None:
        """创建分片记录和本地向量。"""
        for index, chunk in enumerate(chunks):
            vector_id = generate_uuid()
            chunk_record = KnowledgeChunk(
                file_id=file_record.id,
                chunk_text=chunk,
                vector_id=vector_id,
                chunk_index=index,
            )
            self._db_session.add(chunk_record)
            vector_memory.add_text(
                vector_id=vector_id,
                text=chunk,
                metadata={
                    "file_id": file_record.id,
                    "file_name": file_record.file_name,
                    "chunk_index": index,
                    "document_category": document_category,
                },
            )
