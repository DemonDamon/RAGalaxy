from typing import Dict, Any
import spacy
from ..base import BaseExtractor

class SpacyExtractor(BaseExtractor):
    """基于Spacy的实体关系抽取器"""
    
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def extract(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        
        # 抽取实体
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        # 抽取关系(简单的依存关系)
        relations = []
        for token in doc:
            if token.dep_ not in ("punct", "det"):
                relations.append({
                    "source": token.head.text,
                    "relation": token.dep_,
                    "target": token.text
                })
                
        return {
            "entities": entities,
            "relations": relations
        }