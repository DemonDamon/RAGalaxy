from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import Dict, Any, List
from ..base import BaseExtractor

class BertExtractor(BaseExtractor):
    """基于BERT的实体抽取器"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
    def extract(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[t.item()] for t in predictions[0]]
        
        entities = self._merge_entities(tokens, labels)
        return {"entities": entities}
        
    def _merge_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, str]]:
        entities = []
        current_entity = []
        current_label = None
        
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_entity),
                        "type": current_label
                    })
                current_entity = [token]
                current_label = label[2:]
            elif label.startswith("I-") and current_entity:
                current_entity.append(token)
            elif label == "O":
                if current_entity:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_entity),
                        "type": current_label
                    })
                current_entity = []
                current_label = None
                
        return entities