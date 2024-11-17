from typing import List, Dict, Any, Callable
from rank_bm25 import BM25Okapi
from .base import BaseKeywordRetriever

class BM25Retriever(BaseKeywordRetriever):
    """基于BM25算法的关键词检索实现"""
    
    def __init__(self, documents: List[str], tokenizer: str = "space", **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = self._select_tokenizer(tokenizer)
        # 对文档进行分词
        self.tokenized_docs = self.tokenizer(documents)
        # 初始化BM25模型
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.documents = documents

    def _select_tokenizer(self, tokenizer_name: str) -> Callable[[List[str]], List[List[str]]]:
        """选择分词器"""
        if tokenizer_name == "space":
            return lambda texts: [text.split() for text in texts]
        elif tokenizer_name == "porter":
            from nltk.stem import PorterStemmer
            from nltk.tokenize import word_tokenize
            stemmer = PorterStemmer()
            return lambda texts: [[stemmer.stem(token) for token in word_tokenize(text)] for text in texts]
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return lambda texts: [tokenizer.tokenize(text) for text in texts]

    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 对查询进行分词
        tokenized_query = query.split()
        
        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k结果
        top_k_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'content': self.documents[idx],
                'score': float(scores[idx])
            })
            
        return results

    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        return [await self.retrieve(query, top_k) for query in queries]