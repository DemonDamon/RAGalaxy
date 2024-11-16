from typing import Dict, List, Union, Any
import pandas as pd
from pathlib import Path
from .decorators import data_node
from .metrics import ExactMatch, F1Score, PrecisionScore, RecallScore
import json

class QAData:
    """QA数据处理类"""
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data if data is not None else pd.DataFrame()
        self._metrics = {
            "exact_match": ExactMatch(),
            "f1": F1Score(),
            "precision": PrecisionScore(),
            "recall": RecallScore()
        }
        
    @data_node("make_qa")
    def make_qa(self, texts: List[Dict[str, Any]], **kwargs) -> pd.DataFrame:
        """从文本创建QA数据"""
        records = []
        for text in texts:
            chunks = text["chunks"]
            for i, chunk in enumerate(chunks):
                record = {
                    "id": f"{text.get('id', '')}_chunk_{i}",
                    "context": chunk,
                    "question": self._generate_question(chunk),
                    "entities": text.get("entities", {}),
                    "metadata": text.get("metadata", {})
                }
                records.append(record)
        
        self.data = pd.DataFrame.from_records(records)
        return self.data
    
    def evaluate(self, predictions: List[Dict[str, str]], metrics: List[str] = None) -> Dict[str, float]:
        """评估QA质量"""
        if not metrics:
            metrics = ["exact_match", "f1"]
            
        results = {}
        for metric_name in metrics:
            if metric_name not in self._metrics:
                continue
                
            metric = self._metrics[metric_name]
            metric.reset()
            
            for pred in predictions:
                if pred["id"] not in self.data.index:
                    continue
                    
                pred_ans = pred["answer"]
                gold_ans = self.data.loc[pred["id"], "answer"]
                score = metric(pred_ans, gold_ans)
                metric.update(score)
                
            results[metric_name] = metric.compute()
                
        return results
    
    def _generate_question(self, chunk: str, llm=None) -> str:
        """生成问题
        Args:
            chunk: 文本块
            llm: 可选的语言模型
        Returns:
            生成的问题
        """
        # 如果有LLM，优先使用LLM生成
        if llm:
            prompt = f"""
            基于以下文本生成一个清晰的问题:
            {chunk}
            问题需要:
            1. 以问号结尾
            2. 能够考察对文本的理解
            3. 有明确的答案
            """
            return llm(prompt)
        
        # 否则使用规则生成
        sentences = chunk.split('.')
        key_sentence = sentences[0].strip()
        
        topic_indicators = {
            "定义": ["是", "指", "定义为", "被称为"],
            "过程": ["首先", "然后", "接着", "最后"],
            "比较": ["相比", "与", "不同", "区别"],
            "因果": ["因为", "所以", "导致", "原因"]
        }
        
        question_templates = {
            "定义": "什么是{}?",
            "过程": "描述{}的过程是什么?",
            "比较": "{}与其他相关概念有什么区别?",
            "因果": "是什么导致了{}?",
            "default": "关于{}的主要内容是什么?"
        }
        
        topic_type = "default"
        for type_, indicators in topic_indicators.items():
            if any(ind in key_sentence for ind in indicators):
                topic_type = type_
                break
                
        import re
        topic = re.sub(r'[，。？！、]', '', key_sentence)
        if len(topic) > 20:
            topic = topic[:20] + "..."
            
        return question_templates[topic_type].format(topic)
    
    def validate(self) -> bool:
        """验证数据集格式和质量"""
        if self.data.empty:
            return False
            
        # 1. 检查必需列
        required_columns = {"id", "context", "question", "answer"}
        if not all(col in self.data.columns for col in required_columns):
            return False
            
        # 2. 检查数据完整性
        if self.data[list(required_columns)].isnull().any().any():
            return False
            
        # 3. 检查ID唯一性
        if self.data["id"].duplicated().any():
            return False
            
        return True
        
    def clean(self) -> "QAData":
        """清理数据集"""
        if not self.validate():
            # 1. 删除空值行
            self.data = self.data.dropna(subset=["context", "question", "answer"])
            
            # 2. 确保ID唯一
            self.data = self.data.drop_duplicates(subset=["id"], keep="first")
            
            # 3. 重置索引
            self.data = self.data.reset_index(drop=True)
            
        return self
    
    def load_dataset(self, dataset_name: str, dataset_path: str) -> "QAData":
        """加载标准问答数据集
        Args:
            dataset_name: 数据集名称 (musique/2wikimultihop/hotpotqa)
            dataset_path: 数据集路径
        Returns:
            self
        """
        # 参考 HippoRAG 的数据集处理
        dataset_loaders = {
            "musique": self._load_musique,
            "2wikimultihop": self._load_2wiki,
            "hotpotqa": self._load_hotpot
        }
        
        if dataset_name not in dataset_loaders:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        loader = dataset_loaders[dataset_name]
        self.data = loader(dataset_path)
        return self
    
    def save_dataset(self, output_path: str, format: str = "parquet") -> None:
        """保存数据集
        Args:
            output_path: 输出路径
            format: 输出格式 (parquet/csv/json)
        """
        if not self.validate():
            raise ValueError("数据集格式无效")
            
        save_funcs = {
            "parquet": lambda df, path: df.to_parquet(path),
            "csv": lambda df, path: df.to_csv(path, index=False),
            "json": lambda df, path: df.to_json(path, orient="records")
        }
        
        if format not in save_funcs:
            raise ValueError(f"不支持的格式: {format}")
            
        save_funcs[format](self.data, output_path)
    
    def _load_musique(self, path: str) -> pd.DataFrame:
        """加载 MuSiQue 数据集"""
        with open(path) as f:
            data = json.load(f)
        records = []
        for item in data:
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "context": item.get("context", ""),
                "metadata": {"dataset": "musique"}
            }
            records.append(record)
        return pd.DataFrame.from_records(records)
    
    def _load_2wiki(self, path: str) -> pd.DataFrame:
        """加载 2WikiMultiHopQA 数据集"""
        with open(path) as f:
            data = json.load(f)
        records = []
        for item in data:
            record = {
                "id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "context": " ".join(item.get("context", [])),
                "metadata": {"dataset": "2wikimultihop"}
            }
            records.append(record)
        return pd.DataFrame.from_records(records)
    
    def _load_hotpot(self, path: str) -> pd.DataFrame:
        """加载 HotpotQA 数据集"""
        with open(path) as f:
            data = json.load(f)
        records = []
        for item in data:
            record = {
                "id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "context": " ".join(item.get("context", [])),
                "metadata": {"dataset": "hotpotqa"}
            }
            records.append(record)
        return pd.DataFrame.from_records(records)
    
    def process_batch(self, batch_size: int = 32) -> None:
        """批量处理数据
        Args:
            batch_size: 批次大小
        """
        if self.data.empty:
            return
        
        total = len(self.data)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = self.data.iloc[start_idx:end_idx]
            
            # 1. 批量生成问题
            if "question" not in batch.columns:
                questions = []
                for chunk in batch["context"]:
                    question = self._generate_question(chunk)
                    questions.append(question)
                self.data.loc[batch.index, "question"] = questions
            
            # 2. 批量评估
            if "answer" in batch.columns:
                metrics = ["exact_match", "f1"]
                batch_results = {}
                for metric_name in metrics:
                    if metric_name not in self._metrics:
                        continue
                    
                    metric = self._metrics[metric_name]
                    metric.reset()
                    
                    for _, row in batch.iterrows():
                        score = metric(row["prediction"], row["answer"])
                        metric.update(score)
                    
                    batch_results[metric_name] = metric.compute()
                
                # 更新评估结果
                for metric_name, score in batch_results.items():
                    self.data.loc[batch.index, f"score_{metric_name}"] = score
    
    def get_batch_metrics(self) -> Dict[str, float]:
        """获取批量评估指标"""
        if self.data.empty:
            return {}
        
        metrics = {}
        score_columns = [col for col in self.data.columns if col.startswith("score_")]
        
        for col in score_columns:
            metric_name = col.replace("score_", "")
            metrics[metric_name] = self.data[col].mean()
        
        return metrics