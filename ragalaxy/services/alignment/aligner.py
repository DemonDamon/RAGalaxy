from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import torch

class EntityAligner:
    """实体对齐服务
    核心功能: 跨知识库的实体对齐
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.model = SentenceTransformer(
            config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.threshold = config.get("threshold", 0.8)
        
    async def align(
        self,
        source_entities: List[Dict],
        target_entities: List[Dict]
    ) -> List[Tuple[str, str, float]]:
        """对齐实体"""
        # 1. 生成实体表示
        source_texts = [
            f"{e['name']} {e.get('description', '')}" 
            for e in source_entities
        ]
        target_texts = [
            f"{e['name']} {e.get('description', '')}" 
            for e in target_entities
        ]
        
        source_embeds = self.model.encode(source_texts)
        target_embeds = self.model.encode(target_texts)
        
        # 2. 计算相似度并匹配
        alignments = []
        for i, s_embed in enumerate(source_embeds):
            scores = torch.cosine_similarity(
                torch.tensor(s_embed).unsqueeze(0),
                torch.tensor(target_embeds),
                dim=1
            )
            
            max_score, max_idx = torch.max(scores, dim=0)
            if max_score > self.threshold:
                alignments.append((
                    source_entities[i]["id"],
                    target_entities[max_idx]["id"],
                    float(max_score)
                ))
                
        return alignments 