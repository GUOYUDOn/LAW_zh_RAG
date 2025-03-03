from elasticsearch import Elasticsearch
from langchain.schema import BaseRetriever, Document
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Dict
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder
from collections import defaultdict

from start_es import es
# from models import get_rerank

def es_search(es, query_vector, index_name="law_data", top_k=20):
    """
    在 Elasticsearch 中使用向量搜索法律文档。
    """
    query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},  
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'para_vector')",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es.search(index=index_name, body=query)
    
    results = [
        {
            "_id": hit["_id"],
            "para": hit["_source"]["para"],
            "title": hit["_source"]["title"],
            "book": hit["_source"]["book"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results


class WebRetriever(BaseRetriever):
    """实现网络检索并分块"""

    search: DuckDuckGoSearchAPIWrapper = Field(..., description="DuckDuckGo Search API Wrapper")
    num_search_results: int = Field(3, description="Number of search results to retrieve.")
    
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=lambda: RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for web documents."
    )

    def __init__(self, search: DuckDuckGoSearchAPIWrapper, num_search_results: int = 2):
        super().__init__(search=search, num_search_results=num_search_results)  
        self.search = search
        self.num_search_results = num_search_results
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

    def _get_web_documents(self, query_text: str) -> List[Document]:
        try:
            results = self.search.results(query_text, self.num_search_results)
        except Exception as e:
            print(f"Search failed: {e}")
            results = []

        docs = [
            Document(
                page_content=res["snippet"],
                metadata={"link": res["link"], "title": res["title"]}
            )
            for res in results
        ]

        return self.text_splitter.split_documents(docs)

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        """实现 BaseRetriever 的抽象方法，使其可以被实例化"""
        return self._get_web_documents(query)
    

def convert_es_to_documents(es_results: List[Dict]):
    '''
    将 Elasticsearch 搜索结果转换为 Document 对象列表。
    '''
    documents = []
    for doc in es_results:
        metadata = {k: v for k, v in doc.items() if k != "para"}  # 除文本信息之外全部存入元数据
        documents.append(Document(page_content=doc.get("para", ""), metadata=metadata))
    return documents


def rerank_documents(query_text : str, docs: List[Document], rerank_model : CrossEncoder):
    '''
    将es检索和网页检索的结果进行重排序
    '''
    if not docs:
        return []
    
    input_pairs = [(query_text, doc.page_content) for doc in docs]
    scores = rerank_model.predict(input_pairs)
    
    for i, doc in enumerate(docs):
        doc.metadata["rerank_score"] = float(scores[i])
    
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
    return sorted_docs


def merge_documents(docs: List[Document], top_k: int = 10):
    '''
    合并重排序后的前k个文档
    '''
    selected_docs = docs[:top_k]
    books, webs = defaultdict(list), defaultdict(list)
    others = []
    
    for doc in selected_docs:
        content = doc.page_content
        metadata = doc.metadata
        
        if "book" in metadata:
            title = metadata.get("title", "").strip()
            full_text = f"{title}:{content}" if title else content
            books[metadata["book"]].append(full_text)
            
        elif "link" in metadata:
            title = metadata.get("title", "").strip()
            full_text = f"{title}:{content}" if title else content
            webs[metadata["link"]].append(full_text)
            
        else:
            title = metadata.get("title", "").strip()
            full_text = f"{title}:{content}" if title else content
            others.append(full_text)
            
    final_text = []
    for book, text in books.items():
        final_text.append("\n".join(text) + f"\n参考来源：{book}\n")
    for link, text in webs.items():
        final_text.append("\n".join(text) + f"\n参考来源：{link}\n")
    final_text.extend(others)
        
    final_output = "\n\n".join(final_text)
    return final_output   
    
# if __name__ == "__main__":
#     if es.ping():
#         print("Connected to Elasticsearch!")
#     else:
#         print("Connection failed.")

# if __name__ == "__main__":
#     query_text = "劳动合同终止后公司是否需支付经济补偿金？劳动合同,终止,经济补偿金"
#     from models import get_embedding
#     query_vector = get_embedding(query_text)
#     results = es_search(es, query_vector)
    # results = sorted(results, key=lambda x: x["score"], reverse=True)
    # scores = [result["score"] for result in results]
    # paras = [result["para"][:30] for result in results]
    # print(scores)
    # print(paras)
    # documents = convert_es_to_documents(results)
    # print(len(documents))
    # print(documents[0].page_content)
    # print(documents[0].metadata)


# if __name__ == "__main__":
#     search_api = DuckDuckGoSearchAPIWrapper()
#     retriever = WebRetriever(search_api, num_search_results=3)

#     query = "劳动合同终止后公司是否需支付经济补偿金？劳动合同,终止,经济补偿金"
#     documents = retriever._get_web_documents(query)

#     for i, doc in enumerate(documents):
#         print(f"Result {i+1}:")
#         print(f"Title: {doc.metadata['title']}")
#         print(f"Snippet: {doc.page_content}")
#         print(f"Link: {doc.metadata['link']}")
#         print("-" * 50)

# if __name__ == "__main__":
#     query_text = "劳动合同终止后公司是否需支付经济补偿金？劳动合同,终止,经济补偿金"
#     from models import get_embedding, get_rerank
#     query_vector = get_embedding(query_text)
#     results = es_search(es, query_vector)
#     documents = convert_es_to_documents(results)
#     ddd = rerank_documents(query_text=query_text, docs=documents, rerank_model=get_rerank())
#     ttt = merge_documents(ddd, top_k=3)
#     print(ttt)