from celery import Task
from services.index_service import IndexService

class CreateIndexTask(Task):
    async def run(self, doc_id: str):
        index_service = IndexService()
        await index_service.create_index(doc_id) 