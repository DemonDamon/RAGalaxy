from typing import Dict, Type, Union, List, Any
from pathlib import Path
from .base import BaseParser, BaseChunker, BaseExtractor
from .parser.pdf_parser import PDFParser
from .parser.docx_parser import DocxParser
from .parser.markdown_parser import MarkdownParser
from .chunker.semantic_chunker import SemanticChunker
from .chunker.sentence_chunker import SentenceChunker
from .extractor.spacy_extractor import SpacyExtractor
from .extractor.bert_extractor import BertExtractor
from .config import DataConfig
from .utils import DataUtils
from .decorators import data_node
from .dataset import Dataset
from .qa import QAData
import pandas as pd

class DataFactory:
    """数据处理组件工厂类"""
    
    _parsers: Dict[str, Type[BaseParser]] = {
        "pdf": PDFParser,
        "docx": DocxParser,
        "markdown": MarkdownParser
    }
    
    _chunkers: Dict[str, Type[BaseChunker]] = {
        "semantic": SemanticChunker,
        "sentence": SentenceChunker
    }
    
    _extractors: Dict[str, Type[BaseExtractor]] = {
        "spacy": SpacyExtractor,
        "bert": BertExtractor
    }
    
    def __init__(self, config: Union[str, DataConfig, Dict] = None):
        """初始化工厂
        Args:
            config: 配置文件路径、配置对象或配置字典
        """
        if isinstance(config, str):
            self.config = DataConfig(config)
        elif isinstance(config, DataConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = DataConfig(**config)
        else:
            self.config = DataConfig()
            
    @data_node("batch_process")
    def process_files(self, file_pattern: str) -> List[Dict[str, Any]]:
        """批量处理文件
        Args:
            file_pattern: 文件匹配模式
        Returns:
            处理结果列表
        """
        dataset = Dataset(self.config.get("data_dir", "."))
        files = dataset.load_files(file_pattern)
        return dataset.process_batch(files, self.process_file)
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """处理单个文件的完整流程"""
        file_type = DataUtils.get_file_type(file_path)
        
        # 1. 解析文档
        parser = self.create_parser(file_type)
        text = parser.parse(file_path)
        
        # 2. 分块
        chunker_type = self.config.get("chunker_type", "sentence")
        chunker = self.create_chunker(chunker_type)
        chunks = chunker.split(text)
        
        # 3. 实体抽取
        extractor_type = self.config.get("extractor_type", "spacy")
        extractor = self.create_extractor(extractor_type)
        entities = extractor.extract(text)
        
        return {
            "text": text,
            "chunks": chunks,
            "entities": entities
        }
    
    @classmethod
    def create_parser(cls, parser_type: str, **kwargs) -> BaseParser:
        """创建解析器实例"""
        if parser_type not in cls._parsers:
            raise ValueError(f"Unknown parser type: {parser_type}")
        return cls._parsers[parser_type](**kwargs)
    
    @classmethod
    def create_chunker(cls, chunker_type: str, **kwargs) -> BaseChunker:
        """创建分块器实例"""
        if chunker_type not in cls._chunkers:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
        return cls._chunkers[chunker_type](**kwargs)
    
    @classmethod
    def create_extractor(cls, extractor_type: str, **kwargs) -> BaseExtractor:
        """创建抽取器实例"""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        return cls._extractors[extractor_type](**kwargs)
    
    @data_node("create_qa")
    def create_qa_dataset(self, file_pattern: str) -> QAData:
        """从文件创建QA数据集"""
        processed_files = self.process_files(file_pattern)
        qa_data = QAData()
        qa_data.make_qa(processed_files)
        
        if not qa_data.validate():
            raise ValueError("Failed to create valid QA dataset")
        
        return qa_data
    
    @data_node("process_to_qa")
    def process_to_qa(self, file_pattern: str) -> pd.DataFrame:
        """处理文件并创建QA数据
        Args:
            file_pattern: 文件匹配模式
        Returns:
            QA数据DataFrame
        """
        texts = self.process_files(file_pattern)
        qa = QAData()
        return qa.make_qa(texts)
    
    @data_node("evaluate_qa")
    def evaluate_qa(self, qa_data: QAData) -> Dict[str, float]:
        """评估QA数据质量
        Args:
            qa_data: QA数据对象
        Returns:
            评估结果
        """
        metrics = self.config.get("qa_metrics", ["coverage", "diversity"])
        return qa_data.evaluate(metrics)