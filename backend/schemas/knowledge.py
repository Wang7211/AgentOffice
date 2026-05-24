"""知识库数据模型。"""

from pydantic import BaseModel


class KnowledgeUploadResponse(BaseModel):
    """知识库上传响应模型。"""

    file_id: int
    file_name: str
    chunk_count: int


class KnowledgeSearchRequest(BaseModel):
    """知识库检索请求模型。"""

    query: str
    top_k: int = 5
