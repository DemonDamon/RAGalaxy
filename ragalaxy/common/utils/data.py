from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def to_dataframe(
        data: Union[List[Dict], Dict[str, List]],
        orient: str = "records"
    ) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame.from_records(data) if orient == "records" else pd.DataFrame(data)
    
    @staticmethod
    def select_top_k(
        data: pd.DataFrame,
        score_col: str,
        k: int,
        ascending: bool = False
    ) -> pd.DataFrame:
        """选择Top-K"""
        return data.nlargest(k, score_col) if not ascending else data.nsmallest(k, score_col)
    
    @staticmethod
    def flatten_dict(
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DataUtils.flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)