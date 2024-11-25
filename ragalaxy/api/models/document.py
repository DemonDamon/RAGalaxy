from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Document(BaseModel):
    """文档模型"""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    error_message: Optional[str] = None 