"""通用辅助函数。"""

import hashlib
import re
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SENSITIVE_KEYS = ("password", "token", "secret", "key", "mobile", "phone")
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_datetime() -> datetime:
    """返回带时区信息的当前时间。

    返回:
        当前 UTC 时间。

    异常:
        无。
    """
    return datetime.now(timezone.utc)


def local_isoformat(value: datetime) -> str:
    """Serialize UTC DB datetimes as local dashboard/client time."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TIMEZONE).replace(tzinfo=None).isoformat()


def generate_uuid() -> str:
    """生成紧凑 UUID 字符串。

    返回:
        不包含分隔符的 UUID 字符串。

    异常:
        无。
    """
    return uuid.uuid4().hex


def normalize_text(content: str) -> str:
    """清洗文档抽取文本。

    参数:
        content: 原始文本内容。

    返回:
        已压缩重复空白的规范化文本。

    异常:
        无。
    """
    cleaned_text = content.replace("\u200b", "")
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def split_text(content: str, chunk_size: int = 800) -> list[str]:
    """将长文本拆分成稳定分片。

    参数:
        content: 清洗后的文本内容。
        chunk_size: 单个分片最大长度。

    返回:
        非空文本分片列表。

    异常:
        无。
    """
    normalized_text = normalize_text(content)
    chunks: list[str] = []
    current_text = ""
    for paragraph in normalized_text.splitlines():
        if len(current_text) + len(paragraph) + 1 > chunk_size:
            if current_text:
                chunks.append(current_text.strip())
            current_text = paragraph
        else:
            current_text = f"{current_text}\n{paragraph}".strip()
    if current_text:
        chunks.append(current_text.strip())
    return chunks or [normalized_text]


def safe_file_name(file_name: str) -> str:
    """返回文件系统安全的文件名。

    参数:
        file_name: 原始文件名。

    返回:
        保留扩展名的安全文件名。

    异常:
        无。
    """
    path = Path(file_name)
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", path.stem).strip("_")
    suffix = re.sub(r"[^0-9A-Za-z.]+", "", path.suffix.lower())
    safe_stem = stem or "upload"
    return f"{safe_stem}{suffix}"


def text_hash(content: str) -> str:
    """为文本内容生成哈希值。

    参数:
        content: 待哈希文本。

    返回:
        十六进制摘要字符串。

    异常:
        无。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def mask_sensitive(data: dict[str, object]) -> dict[str, object]:
    """在写日志前脱敏敏感字段。

    参数:
        data: 原始字典。

    返回:
        已脱敏敏感值的字典。

    异常:
        无。
    """
    masked_data: dict[str, object] = {}
    for key, value in data.items():
        lower_key = key.lower()
        if any(sensitive_key in lower_key for sensitive_key in SENSITIVE_KEYS):
            masked_data[key] = "***"
        else:
            masked_data[key] = value
    return masked_data
