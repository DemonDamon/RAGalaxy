from fastapi import UploadFile
import aiofiles
import uuid

class DocumentService:
    async def save_document(self, file: UploadFile) -> str:
        doc_id = str(uuid.uuid4())
        
        # 保存文件
        async with aiofiles.open(f"storage/documents/{doc_id}.pdf", "wb") as f:
            content = await file.read()
            await f.write(content)
            
        return doc_id 