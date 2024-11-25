from fastapi import APIRouter, UploadFile, File
from services.document_service import DocumentService
from tasks.indexing import create_index_task

router = APIRouter()

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # 保存文档
    doc_service = DocumentService()
    doc_id = await doc_service.save_document(file)
    
    # 创建异步索引任务
    create_index_task.delay(doc_id)
    
    return {"message": "文档已接收，正在处理", "doc_id": doc_id} 