from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class DatasetSchema:
    """数据集模式"""
    question: str = "question"
    context: str = "context"
    answer: str = "answer"
    id: str = "id"

class PreprocessUtils:
    """预处理工具类"""
    
    @staticmethod
    def validate_qa_dataset(
        df: pd.DataFrame,
        schema: DatasetSchema = DatasetSchema()
    ) -> bool:
        """验证问答数据集"""
        required_cols = [schema.question, schema.answer]
        return all(col in df.columns for col in required_cols)
    
    @staticmethod
    def validate_corpus_dataset(
        df: pd.DataFrame,
        schema: DatasetSchema = DatasetSchema()
    ) -> bool:
        """验证语料数据集"""
        required_cols = [schema.context, schema.id]
        return all(col in df.columns for col in required_cols)
    
    @staticmethod
    def cast_dataset(
        df: pd.DataFrame,
        schema: DatasetSchema = DatasetSchema()
    ) -> pd.DataFrame:
        """转换数据集格式"""
        # 确保所有文本列为字符串类型
        text_cols = [schema.question, schema.answer, schema.context]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df