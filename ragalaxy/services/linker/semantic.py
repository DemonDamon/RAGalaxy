from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from .base import BaseLinker, EntityMention, EntityCandidate

class SemanticLinker(BaseLinker):
    """基于语义的实体链接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._init_model()
        self.kb_embeddings = {}  # 知识库实体的预计算嵌入
        self.kb_entities = {}    # 知识库实体信息
        
    def _init_model(self) -> SentenceTransformer:
        """初始化编码模型"""
        model_name = self.config.get(
            "model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        device = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return SentenceTransformer(model_name).to(device)
        
    async def load_kb(self, kb_path: str):
        """加载知识库"""
        import json
        from pathlib import Path
        
        # 1. 加载知识库实体
        kb_file = Path(kb_path)
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
            
        # 2. 计算实体表示
        texts = []
        for entity in kb_data:
            self.kb_entities[entity["id"]] = entity
            text = f"{entity['name']} {entity.get('description', '')}"
            texts.append(text)
            
        # 3. 批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )
        
        # 4. 保存嵌入
        for entity_id, embedding in zip(self.kb_entities.keys(), embeddings):
            self.kb_embeddings[entity_id] = embedding
            
    async def get_candidates(
        self,
        mention: EntityMention,
        top_k: int = 5
    ) -> List[EntityCandidate]:
        """获取候选实体"""
        # 1. 编码mention
        mention_text = mention.text
        mention_embedding = self.model.encode(mention_text)
        
        # 2. 计算相似度并排序
        scores = {}
        for entity_id, embedding in self.kb_embeddings.items():
            if self.kb_entities[entity_id]["type"] == mention.type:
                score = torch.cosine_similarity(
                    torch.tensor(mention_embedding),
                    torch.tensor(embedding),
                    dim=0
                )
                scores[entity_id] = score.item()
                
        # 3. 获取top-k候选
        candidates = []
        for entity_id, score in sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            entity = self.kb_entities[entity_id]
            candidates.append(EntityCandidate(
                id=entity_id,
                name=entity["name"],
                type=entity["type"],
                score=score,
                description=entity.get("description", ""),
                attributes=entity.get("attributes", {})
            ))
            
        return candidates
        
    async def link(
        self,
        mention: EntityMention,
        candidates: List[EntityCandidate]
    ) -> EntityCandidate:
        """实体消歧"""
        # 简单实现:返回得分最高的候选
        return max(candidates, key=lambda x: x.score) 