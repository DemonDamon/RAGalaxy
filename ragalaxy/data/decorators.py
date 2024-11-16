from functools import wraps
from typing import Callable
import logging

def data_node(name: str = None):
    """数据处理节点装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            node_name = name or func.__name__
            logging.info(f"Processing node: {node_name}")
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logging.error(f"Error in node {node_name}: {str(e)}")
                raise
        return wrapper
    return decorator