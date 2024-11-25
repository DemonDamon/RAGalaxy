from typing import List, Dict, Any
import networkx as nx
import json
from pathlib import Path
from .base import BaseGraphStore

class NetworkXStore(BaseGraphStore):
    """NetworkX图存储"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph = nx.MultiDiGraph()
        self.save_path = Path(config.get("save_path", "data/graph.json"))
        
    async def connect(self) -> bool:
        """加载图数据"""
        try:
            if self.save_path.exists():
                # 从JSON文件加载图
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
            return True
        except Exception as e:
            print(f"加载图数据失败: {str(e)}")
            return False
            
    async def disconnect(self) -> bool:
        """保存图数据"""
        try:
            # 保存为JSON格式
            data = nx.node_link_data(self.graph)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存图数据失败: {str(e)}")
            return False
            
    async def add_entities(self, entities: List[Dict]) -> bool:
        """添加实体节点"""
        try:
            for entity in entities:
                self.graph.add_node(
                    entity["id"],
                    **entity.get("properties", {})
                )
            return True
        except Exception as e:
            print(f"添加实体失败: {str(e)}")
            return False
            
    async def add_relations(self, relations: List[Dict]) -> bool:
        """添加关系边"""
        try:
            for relation in relations:
                self.graph.add_edge(
                    relation["source_id"],
                    relation["target_id"],
                    type=relation["type"],
                    **relation.get("properties", {})
                )
            return True
        except Exception as e:
            print(f"添加关系失败: {str(e)}")
            return False
            
    async def query(self, query: str) -> List[Dict]:
        """执行图查询
        注意: 这里使用简单的DSL来查询,而不是Cypher
        查询格式: {
            "type": "neighbors"|"path"|"subgraph",
            "params": {...}
        }
        """
        try:
            query_dict = json.loads(query)
            query_type = query_dict["type"]
            params = query_dict["params"]
            
            if query_type == "neighbors":
                return self._query_neighbors(**params)
            elif query_type == "path":
                return self._query_path(**params)
            elif query_type == "subgraph":
                return self._query_subgraph(**params)
            else:
                raise ValueError(f"不支持的查询类型: {query_type}")
                
        except Exception as e:
            print(f"查询执行失败: {str(e)}")
            return []
            
    def _query_neighbors(
        self, 
        node_id: str, 
        depth: int = 1,
        relation_types: List[str] = None
    ) -> List[Dict]:
        """查询邻居节点"""
        results = []
        
        # 获取指定深度的邻居
        neighbors = nx.single_source_shortest_path_length(
            self.graph, node_id, cutoff=depth
        )
        
        # 过滤关系类型
        for neighbor_id, distance in neighbors.items():
            edges = self.graph.get_edge_data(node_id, neighbor_id)
            if edges and (not relation_types or 
                         any(e["type"] in relation_types for e in edges.values())):
                results.append({
                    "node": dict(self.graph.nodes[neighbor_id]),
                    "distance": distance
                })
                
        return results
        
    def _query_path(
        self,
        source_id: str,
        target_id: str,
        cutoff: int = None
    ) -> List[Dict]:
        """查询最短路径"""
        try:
            paths = list(nx.all_shortest_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=cutoff
            ))
            
            return [{
                "path": path,
                "length": len(path) - 1
            } for path in paths]
        except nx.NetworkXNoPath:
            return []
            
    def _query_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True
    ) -> Dict:
        """查询子图"""
        # 创建子图
        subgraph = self.graph.subgraph(node_ids)
        
        return {
            "nodes": [
                {
                    "id": node,
                    "properties": dict(data)
                }
                for node, data in subgraph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": data["type"],
                    "properties": {
                        k: v for k, v in data.items() 
                        if k != "type"
                    }
                }
                for u, v, data in subgraph.edges(data=True)
            ] if include_edges else []
        } 