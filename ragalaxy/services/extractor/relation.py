from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import BaseExtractor, Entity, Relation

class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.id2label = self._load_labels()
        
    def _init_tokenizer(self):
        """初始化分词器"""
        model_name = self.config.get(
            "model",
            "microsoft/BioGPT-Large-PubMedQA"  # 示例模型
        )
        return AutoTokenizer.from_pretrained(model_name)
        
    def _init_model(self):
        """初始化模型"""
        model_name = self.config.get(
            "model",
            "microsoft/BioGPT-Large-PubMedQA"
        )
        device = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        
    def _load_labels(self) -> Dict[int, str]:
        """加载关系标签"""
        return self.model.config.id2label
        
    async def extract(
        self,
        text: str,
        entity_pairs: List[tuple[Entity, Entity]]
    ) -> List[Relation]:
        """抽取实体间的关系"""
        relations = []
        
        for e1, e2 in entity_pairs:
            # 1. 构建输入
            input_text = self._build_input(text, e1, e2)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 2. 预测关系
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(-1)
                
            # 3. 获取关系类型
            relation_type = self.id2label[predictions.item()]
            
            # 4. 创建关系
            if relation_type != "no_relation":
                relations.append(Relation(
                    source_id=e1.id,
                    target_id=e2.id,
                    type=relation_type,
                    properties={
                        "confidence": float(
                            torch.softmax(outputs.logits, dim=-1).max()
                        )
                    }
                ))
                
        return relations
        
    def _build_input(self, text: str, e1: Entity, e2: Entity) -> str:
        """构建模型输入
        添加特殊标记以标识实体位置
        """
        # 在实体前后添加标记
        text_list = list(text)
        text_list.insert(e2.end, " [E2] ")
        text_list.insert(e2.start, " [/E2] ")
        text_list.insert(e1.end, " [E1] ")
        text_list.insert(e1.start, " [/E1] ")
        
        return "".join(text_list) 