from typing import Optional, Any, Iterator
from tqdm import tqdm

class ProgressUtils:
    """进度工具类"""
    
    @staticmethod
    def track_progress(
        iterable: Iterator,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "it"
    ) -> Iterator:
        """跟踪进度"""
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            ncols=80
        )
    
    @staticmethod
    def create_progress_bar(
        total: int,
        desc: Optional[str] = None,
        unit: str = "it"
    ) -> tqdm:
        """创建进度条"""
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            ncols=80
        )