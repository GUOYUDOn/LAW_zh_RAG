from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from workflow.splitter import LawSplitter
from workflow.loader import LawLoader
from workflow.start_es import es
from workflow.save import save_as_json
from workflow.utils import bulk_import
from workflow.models import get_embedding

# 加载所有md数据
path_data = "data_book"
loader = LawLoader(path_data, file_type="md")
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 文档分割
splitter = LawSplitter(chunk_size=200, chunk_overlap=30)
split_documents_mk = splitter.split_documents(documents)
print(f"Split into {len(split_documents_mk)} smaller documents")

for doc in tqdm(split_documents_mk, desc="计算向量中", unit="doc"):    
    doc.metadata["para_vector"] = get_embedding(doc.page_content) # 计算段落内容向量
    doc.metadata["title_vector"] = get_embedding(doc.metadata.get("title", ""))  # 计算标题向量，可以避免title为空情况
    
split_documents_mk[10].metadata  # 查看文档的metadata

# 将向量化后的文档以及元数据写入JSON文件
save_as_json(split_documents_mk, index_name="law_data", output_path="data_book/lawbook.json")

# 将本地json文件写入es
bulk_file_path = "data_book/lawbook.json"
index_name = "law_data"

success, errors = bulk_import(es, bulk_file_path, index_name)

# query = {
#     "query": {
#         "match": {
#             "title": "宪法"
#         }
#     },
#     "size" : 2
# }

# response = es.search(index="law_data", body=query)

# print(f"匹配到 {response['hits']['total']['value']} 篇文档：")
# for hit in response["hits"]["hits"]:
#     print(f"\n文档标题: {hit['_source']['title']}")
#     print(f"相关度分数: {hit['_score']}")
#     print(f"文档内容: {hit['_source']['para'][:100]}...")  # 仅显示前 100 个字符
#     print(f"来源: {hit['_source']['source']}")
    
# import numpy as np
# from numpy.linalg import norm
# from elasticsearch import Elasticsearch

# # 计算余弦相似度，避免除零错误
# def cosine_similarity(vec1, vec2):
#     norm1, norm2 = norm(vec1), norm(vec2)
#     if norm1 == 0 or norm2 == 0:
#         return 0  # 避免除以零
#     return np.dot(vec1, vec2) / (norm1 * norm2)

# query = {"query": {"match_all": {}}, "size": 2000}  
# response = es.search(index="law_data", body=query)  

# query_vector = np.array([-0.0005] * 1024)  # 需要查询的向量

# # 计算相似度
# results = []
# for hit in response["hits"]["hits"]:
#     doc_vector = np.array(hit["_source"]["title_vector"])
#     similarity = cosine_similarity(query_vector, doc_vector)
#     results.append((hit["_source"], similarity))

# # 按相似度排序，取前 2 条
# results.sort(key=lambda x: x[1], reverse=True)
# top_results = results[:2]
# print(f"从 {len(response['hits']['hits'])} 篇文档中找到最相似的 2 篇：\n")

# for idx, (doc, sim) in enumerate(top_results, start=1):
#     print(f"第 {idx} 篇匹配文档")
#     print(f"相似度得分: {sim:.4f}")
#     print(f"文档标题: {doc['title']}")
#     print(f"内容预览: {doc['para'][:50]}...")  # 只显示前 50 个字符
#     print(f"来源: {doc['source']}\n")
#     print("-" * 50)

# # 查看当前的索引状态
# indices = es.cat.indices(v=True)
# print("所有索引:")
# print(indices)

