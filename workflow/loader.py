# coding: utf-8
from typing import Any
from langchain.document_loaders import DirectoryLoader, TextLoader

class LawLoader(DirectoryLoader):
    """加载法律书籍，同时仅支持一种文件类型。"""
    
    def __init__(self, path: str, file_type: str = "md", **kwargs: Any) -> None:
        """
        初始化 LawLoader。

        参数:
            path (str): 要加载文件的目录路径。
            file_type (str): 要搜索的文件扩展名（例如 'md'、'txt'、'csv'）。
                            默认为 'md'。
            **kwargs (Any): 传递给 DirectoryLoader 的其他参数。
        """
        if not isinstance(file_type, str) or file_type not in ('md', 'txt', 'csv'):
            raise ValueError("file_type 必须是一个表示单一文件扩展名的字符串（例如 'md'、'txt'、'csv'）。")

        loader_cls = TextLoader
        glob_pattern = f"**/*.{file_type}"  # 确保仅搜索一种文件类型
        
        super().__init__(path, loader_cls=loader_cls, glob=glob_pattern, **kwargs)


