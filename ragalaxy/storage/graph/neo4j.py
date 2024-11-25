from typing import List, Dict, Any
import neo4j
from .base import BaseGraphStore

class Neo4jStore(BaseGraphStore):
    """Neo4j图数据库存储"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.driver = None
        self.database = config.get("database", "neo4j")
        
    async def connect(self) -> bool:
        """连接Neo4j"""
        try:
            uri = self.config.get("uri", "bolt://localhost:7687")
            user = self.config.get("user", "neo4j")
            password = self.config.get("password", "ragalaxy")
            
            self.driver = neo4j.AsyncGraphDatabase.driver(
                uri, auth=(user, password)
            )
            return True
        except Exception as e:
            print(f"Neo4j连接失败: {str(e)}")
            return False
            
    async def disconnect(self) -> bool:
        """断开连接"""
        if self.driver:
            await self.driver.close()
            return True
        return False
        
    async def add_entities(self, entities: List[Dict]) -> bool:
        """添加实体"""
        async with self.driver.session(database=self.database) as session:
            try:
                # 批量创建实体
                cypher = """
                UNWIND $entities as entity
                MERGE (n:Entity {id: entity.id})
                SET n += entity.properties
                """
                await session.run(cypher, {"entities": entities})
                return True
            except Exception as e:
                print(f"添加实体失败: {str(e)}")
                return False
                
    async def add_relations(self, relations: List[Dict]) -> bool:
        """添加关系"""
        async with self.driver.session(database=self.database) as session:
            try:
                # 批量创建关系
                cypher = """
                UNWIND $relations as rel
                MATCH (s:Entity {id: rel.source_id})
                MATCH (t:Entity {id: rel.target_id})
                MERGE (s)-[r:RELATES {type: rel.type}]->(t)
                SET r += rel.properties
                """
                await session.run(cypher, {"relations": relations})
                return True
            except Exception as e:
                print(f"添加关系失败: {str(e)}")
                return False
                
    async def query(self, cypher: str) -> List[Dict]:
        """执行查询"""
        async with self.driver.session(database=self.database) as session:
            try:
                result = await session.run(cypher)
                return [record.data() for record in await result.fetch()]
            except Exception as e:
                print(f"查询执行失败: {str(e)}")
                return [] 