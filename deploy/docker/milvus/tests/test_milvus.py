from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


# 连接到 Milvus
def connect_to_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )


# 创建集合
def create_collection():
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    
    # 定义 schema
    schema = CollectionSchema(fields=fields, description="test collection")
    
    # 创建集合
    collection = Collection(name="test_collection", schema=schema)
    return collection


def main():
    try:
        # 连接 Milvus
        connect_to_milvus()
        print("Successfully connected to Milvus")
        
        # 如果集合已存在，先删除
        if utility.has_collection("test_collection"):
            utility.drop_collection("test_collection")
        
        # 创建集合
        collection = create_collection()
        print("Successfully created collection")
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Successfully created index")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 断开连接
        connections.disconnect("default")


if __name__ == "__main__":
    main()