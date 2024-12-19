import redis
import json


def test_redis_connection():
    """测试Redis连接"""
    try:
        # 连接Redis
        r = redis.Redis(
            host="localhost",  # 或者使用 "redis" 如果在docker网络中
            port=6379,
            password="ragalaxy",  # 与docker-compose中设置的密码一致
            decode_responses=True
        )
        
        # 测试连接
        r.ping()
        
        # 测试基本操作
        r.set("test_key", "test_value")
        value = r.get("test_key")
        
        print("Redis连接成功!")
        print(f"测试读写: {value}")
        return True
        
    except redis.ConnectionError as e:
        print(f"Redis连接失败: {str(e)}")
        return False
    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        return False


if __name__ == "__main__":
    test_redis_connection()