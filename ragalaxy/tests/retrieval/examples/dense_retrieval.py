from ragalaxy.retrieval.semantic.dense import DenseRetriever

async def main():
    # 初始化检索器
    retriever = DenseRetriever(
        model_name="BAAI/bge-small-zh",
        vector_store="faiss"
    )
    
    # 添加文档
    documents = [
        "人工智能正在改变我们的生活",
        "机器学习技术日新月异",
        "深度学习在图像识别领域取得重大突破"
    ]
    
    # 检索相关文档
    results = await retriever.retrieve(
        query="人工智能的发展",
        top_k=2
    )
    
    # 打印结果
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())