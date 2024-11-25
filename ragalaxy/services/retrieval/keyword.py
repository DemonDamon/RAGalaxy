from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from .base import BaseRetriever, SearchResult
import numpy as np

class BM25Retriever(BaseRetriever):
    """BM25检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tokenizer = self._init_tokenizer()
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []
        
    def _init_tokenizer(self):
        """初始化分词器"""
        from transformers import AutoTokenizer
        tokenizer_name = self.config.get("tokenizer", "bert-base-chinese")
        return AutoTokenizer.from_pretrained(tokenizer_name)
        
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        return self.tokenizer.tokenize(text)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档"""
        try:
            # 保存文档
            self.documents.extend(documents)
            
            # 构建BM25索引
            tokenized_docs = [
                self._tokenize(doc["content"]) 
                for doc in documents
            ]
            self.bm25 = BM25Okapi(tokenized_docs)
            return True
        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return False
            
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """BM25检索"""
        if not self.bm25:
            return []
            
        # 1. 对查询分词
        tokenized_query = self._tokenize(query)
        
        # 2. BM25检索
        scores = self.bm25.get_scores(tokenized_query)
        
        # 3. 获取top-k结果
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 4. 格式化结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    content=self.documents[idx]["content"],
                    score=float(scores[idx]),
                    metadata=self.documents[idx].get("metadata", {}),
                    source="bm25"
                ))
                
        return results 