from typing import List, Dict, Any
import networkx as nx
from collections import defaultdict

class RuleLearner:
    """规则学习器
    核心功能: 从图中学习高置信度的规则模式
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.min_support = config.get("min_support", 5)
        self.min_confidence = config.get("min_confidence", 0.6)
        
    def learn_rules(self, graph: nx.MultiDiGraph) -> List[Dict]:
        """学习规则"""
        # 1. 收集路径模式
        patterns = defaultdict(int)
        relations = defaultdict(int)
        
        # 统计两跳路径
        for node in graph.nodes():
            paths = self._get_two_hop_paths(graph, node)
            for path in paths:
                # 路径模式: (关系1,关系2) -> 直接关系
                pattern = (path["path"][0], path["path"][1])
                direct = path["direct"]
                
                if direct:  # 有直接关系
                    patterns[pattern] += 1
                relations[pattern] += 1
                    
        # 2. 生成规则
        rules = []
        for pattern, support in patterns.items():
            confidence = support / relations[pattern]
            if support >= self.min_support and confidence >= self.min_confidence:
                rules.append({
                    "pattern": list(pattern),
                    "support": support,
                    "confidence": confidence
                })
                
        return sorted(rules, key=lambda x: x["confidence"], reverse=True)
        
    def _get_two_hop_paths(
        self, 
        graph: nx.MultiDiGraph, 
        source: str
    ) -> List[Dict]:
        """获取两跳路径"""
        paths = []
        
        # 获取一跳邻居
        for _, n1, r1 in graph.edges(source, data=True):
            # 获取二跳邻居
            for _, n2, r2 in graph.edges(n1, data=True):
                if n2 != source:
                    # 检查是否存在直接关系
                    direct = graph.has_edge(source, n2)
                    paths.append({
                        "path": [r1["type"], r2["type"]],
                        "nodes": [source, n1, n2],
                        "direct": direct
                    })
                    
        return paths 