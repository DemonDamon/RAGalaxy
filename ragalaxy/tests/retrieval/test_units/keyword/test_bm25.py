import pytest
from ragalaxy.retrieval.keyword.bm25 import BM25Retriever

class TestBM25Retriever:
    async def test_retrieve_single_query(self, sample_documents):
        retriever = BM25Retriever(documents=sample_documents)
        results = await retriever.retrieve("人工智能", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r["score"], float) for r in results)
        assert all("content" in r for r in results)
    
    async def test_batch_retrieve(self, sample_documents, sample_queries):
        retriever = BM25Retriever(documents=sample_documents)
        results = await retriever.batch_retrieve(sample_queries, top_k=2)
        
        assert len(results) == len(sample_queries)
        assert all(len(r) == 2 for r in results)