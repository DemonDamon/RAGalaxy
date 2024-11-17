import pytest
from typing import List, Dict
import numpy as np

@pytest.fixture
def sample_documents() -> List[str]:
    return [
        "人工智能是计算机科学的一个重要分支",
        "机器学习是人工智能的核心技术之一",
        "深度学习是机器学习的一个重要方向",
        "自然语言处理是人工智能的重要应用领域"
    ]

@pytest.fixture
def sample_queries() -> List[str]:
    return [
        "什么是人工智能",
        "机器学习的应用",
        "深度学习技术"
    ]

@pytest.fixture
def dense_vectors() -> np.ndarray:
    return np.random.rand(4, 768)  # 4 documents, 768 dimensions