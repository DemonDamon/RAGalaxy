import pytest
import time
from ragalaxy.retrieval.semantic.dense import DenseRetriever

def test_dense_retrieval_performance(sample_documents, sample_queries):
    retriever = DenseRetriever(model_name="BAAI/bge-small-zh")
    
    # 测试检索延迟
    start_time = time.time()
    results = await retriever.retrieve(sample_queries[0], top_k=10)
    latency = time.time() - start_time
    
    assert latency < 0.5  # 延迟应小于500ms
    
    # 测试批量检索吞吐量
    start_time = time.time()
    batch_results = await retriever.batch_retrieve(sample_queries, top_k=10)
    throughput = len(sample_queries) / (time.time() - start_time)
    
    assert throughput > 10  # 每秒处理超过10个查询