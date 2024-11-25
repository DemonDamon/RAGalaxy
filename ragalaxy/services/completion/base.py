from typing import List, Dict, Any
import networkx as nx

class GraphCompletion:
    """图谱补全服务
    核心功能:
    1. 缺失关系预测
    2. 实体属性补全
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.6)
        
    async def complete_relations(
        self, 
        graph: nx.MultiDiGraph,
        entity_id: str
    ) -> List[Dict]:
        """预测实体的潜在关系"""
        # 1. 基于路径模式预测
        patterns = self._extract_patterns(graph)
        
        # 2. 应用模式预测新关系
        predictions = []
        for pattern in patterns:
            if pattern["confidence"] > self.min_confidence:
                new_relations = self._apply_pattern(
                    graph, 
                    entity_id, 
                    pattern
                )
                predictions.extend(new_relations)
                
        return predictions
        
    def _extract_patterns(self, graph: nx.MultiDiGraph) -> List[Dict]:
        """提取关系模式"""
        # 简单实现:统计两跳路径的关系组合
        patterns = {}
        
        for node in graph.nodes():
            # 获取两跳路径
            for path in nx.single_source_shortest_path_length(
                graph, node, cutoff=2
            ).items():
                if path[1] == 2:  # 只考虑两跳路径
                    pattern = self._get_path_pattern(graph, node, path[0])
                    if pattern:
                        key = f"{pattern[0]}-{pattern[1]}"
                        patterns[key] = patterns.get(key, 0) + 1
                        
        # 计算置信度
        total = sum(patterns.values())
        return [
            {
                "relations": key.split("-"),
                "confidence": count / total
            }
            for key, count in patterns.items()
        ]
        
    def _get_path_pattern(
        self, 
        graph: nx.MultiDiGraph,
        source: str,
        target: str
    ) -> List[str]:
        """获取路径的关系模式"""
        try:
            path = nx.shortest_path(graph, source, target)
            if len(path) == 3:  # 两跳路径
                r1 = graph.get_edge_data(path[0], path[1])[0]["type"]
                r2 = graph.get_edge_data(path[1], path[2])[0]["type"]
                return [r1, r2]
        except:
            pass
        return None
        
    def _apply_pattern(
        self,
        graph: nx.MultiDiGraph,
        entity_id: str,
        pattern: Dict
    ) -> List[Dict]:
        """应用模式预测新关系"""
        predictions = []
        relations = pattern["relations"]
        
        # 查找满足第一个关系的邻居
        for _, neighbor, data in graph.edges(entity_id, data=True):
            if data["type"] == relations[0]:
                # 查找满足第二个关系的邻居
                for _, target, target_data in graph.edges(neighbor, data=True):
                    if target_data["type"] == relations[1]:
                        predictions.append({
                            "source": entity_id,
                            "target": target,
                            "relation": f"{relations[0]}_{relations[1]}",
                            "confidence": pattern["confidence"]
                        })
                        
        return predictions 