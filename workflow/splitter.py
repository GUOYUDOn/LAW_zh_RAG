# coding: utf-8
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List

class LawSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, doc_type: str = "md", **kwargs: Any) -> None:
        """
        初始化一个 LawSplitter，并传入额外参数。
        """
        self.doc_type = doc_type

        separators = [r"第\S*条 "]
        is_separator_regex = True

        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        super().__init__(separators=separators, is_separator_regex=is_separator_regex, **kwargs)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        拆分文档并添加额外的元数据。
        """
        texts, metadatas = [], []

        for docs in documents:
            sub_docs = self.md_splitter.split_text(docs.page_content)
            
            for sub_doc in sub_docs:
                texts.append(sub_doc.page_content)

                # 将 header1 ~ header4 合并为 title
                headers = []
                for h in ["header1", "header2", "header3", "header4"]:
                    if h in sub_doc.metadata and sub_doc.metadata[h]:
                        headers.append(sub_doc.metadata[h])
                combined_title = " ".join(headers) if headers else ""

                # 合并原有的元数据，并添加额外的字段
                new_metadata = sub_doc.metadata| docs.metadata 


                new_metadata["book"] = sub_doc.metadata.get("header1")
                new_metadata["title"] = combined_title
                new_metadata["doc_type"] = self.doc_type

                metadatas.append(new_metadata)

        return self.create_documents(texts, metadatas=metadatas)
