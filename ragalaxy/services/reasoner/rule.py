from typing import List, Dict, Any
import networkx as nx
from .base import BaseReasoner, Rule, Inference

class RuleReasoner(BaseReasoner):
    """基于规则的推理器"""
    
    async def load_rules(self, rules_path: str):
        """加载规则
        规则格式:
        {
            "id": "rule1",
            "name": "传递规则",
            "pattern": {
                "relations": ["父亲", "父亲"],
                "inference": "祖父"
            },
            "confidence": 0.9
        }
        """
        import json
        from pathlib import Path
        
        rules_file = Path(rules_path)
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            
        self.rules = [Rule(**rule) for rule in rules_data]
        
    async def infer(
        self,
        graph: nx.MultiDiGraph,
        entity_id: str,
        max_depth: int = 2
    ) -> List[Inference]:
        """执行规则推理"""
        inferences = []
        
        # 1. 获取实体的邻居路径
        paths = self._find_paths(graph, entity_id, max_depth)
        
        # 2. 应用规则
        for rule in self.rules:
            pattern = rule.pattern
            required_relations = pattern["relations"]
            
            # 检查每条路径
            for path in paths:
                if len(path) - 1 == len(required_relations):
                    # 获取路径上的关系
                    relations = []
                    for i in range(len(path) - 1):
                        edge_data = graph.get_edge_data(
                            path[i], 
                            path[i + 1]
                        )
                        if edge_data:
                            relations.append(
                                edge_data[0]["type"]  # 获取第一个关系类型
                            )
                            
                    # 检查关系序列是否匹配规则
                    if relations == required_relations:
                        inferences.append(Inference(
                            source_id=path[0],
                            target_id=path[-1],
                            relation_type=pattern["inference"],
                            rule_id=rule.id,
                            confidence=rule.confidence,
                            evidence=[{
                                "path": path,
                                "relations": relations
                            }]
                        ))
                        
        return inferences
        
    def _find_paths(
        self,
        graph: nx.MultiDiGraph,
        source: str,
        max_depth: int
    ) -> List[List[str]]:
        """查找所有可能的路径"""
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
                
            paths.append(path[:])
            
            for neighbor in graph.neighbors(current):
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
                    
        dfs(source, [source], 0)
        return paths
        
    async def explain(
        self,
        inference: Inference
    ) -> Dict[str, Any]:
        """解释推理过程"""
        # 查找使用的规则
        rule = next(r for r in self.rules if r.id == inference.rule_id)
        
        return {
            "rule": {
                "name": rule.name,
                "pattern": rule.pattern
            },
            "path": inference.evidence[0]["path"],
            "relations": inference.evidence[0]["relations"],
            "confidence": inference.confidence
        } 