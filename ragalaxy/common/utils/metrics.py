from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

class MetricsUtils:
    """评估指标工具类"""
    
    @staticmethod
    def calculate_retrieval_metrics(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> Dict[str, float]:
        """计算检索评估指标"""
        # 计算精确率、召回率和F1
        precision = len(set(relevant_docs) & set(retrieved_docs)) / len(retrieved_docs) if retrieved_docs else 0
        recall = len(set(relevant_docs) & set(retrieved_docs)) / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    @staticmethod
    def calculate_generation_metrics(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算生成评估指标"""
        from rouge import Rouge
        rouge = Rouge()
        
        # 计算ROUGE分数
        scores = rouge.get_scores(predictions, references, avg=True)
        
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    
    @staticmethod
    def calculate_recall(
        relevant: List[str],
        retrieved: List[str]
    ) -> float:
        """计算召回率"""
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        return len(relevant_set.intersection(retrieved_set)) / len(relevant_set)
    
    @staticmethod
    def calculate_precision(
        relevant: List[str],
        retrieved: List[str]
    ) -> float:
        """计算精确率"""
        if not retrieved:
            return 0.0
        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        return len(relevant_set.intersection(retrieved_set)) / len(retrieved_set)
    
    @staticmethod
    def calculate_f1(
        precision: float,
        recall: float
    ) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_semantic_similarity(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> float:
        """计算语义相似度"""
        return float(cosine_similarity(embeddings1.reshape(1, -1), 
                                    embeddings2.reshape(1, -1))[0, 0])