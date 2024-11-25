from typing import Dict, Any
import json

class QueryOptimizer:
    """查询优化器
    核心功能:
    1. 查询重写
    2. 路径优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def optimize(self, query: str) -> str:
        """优化查询"""
        # 1. 解析查询
        query_dict = json.loads(query)
        
        # 2. 应用优化规则
        if query_dict["type"] == "path":
            return self._optimize_path_query(query_dict)
        elif query_dict["type"] == "neighbors":
            return self._optimize_neighbor_query(query_dict)
            
        return query
        
    def _optimize_path_query(self, query: Dict) -> str:
        """优化路径查询
        1. 添加路径长度限制
        2. 过滤无关的关系类型
        """
        params = query["params"]
        
        # 添加最大长度限制
        if "cutoff" not in params:
            params["cutoff"] = self.config.get("max_path_length", 3)
            
        # 如果指定了关系类型,添加过滤
        if "relation_types" in params:
            params["filter_edges"] = {
                "type": params.pop("relation_types")
            }
            
        return json.dumps(query)
        
    def _optimize_neighbor_query(self, query: Dict) -> str:
        """优化邻居查询
        1. 限制深度
        2. 添加关系过滤
        """
        params = query["params"]
        
        # 限制最大深度
        max_depth = self.config.get("max_neighbor_depth", 2)
        if "depth" not in params or params["depth"] > max_depth:
            params["depth"] = max_depth
            
        return json.dumps(query) 