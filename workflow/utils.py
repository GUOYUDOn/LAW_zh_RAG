import json
from elasticsearch import Elasticsearch, helpers

def create_index(es: Elasticsearch, index_name: str, shards: int = 3, replicas: int = 1, dims: int = 1024):
    """
    创建 Elasticsearch 索引。

    参数:
    - es: Elasticsearch 客户端实例
    - index_name: 要创建的索引名称
    - shards: 主分片数，默认为 3
    - replicas: 副本数，默认为 1
    - dims: 向量维度，默认为 1024

    返回:
    - 成功创建索引时返回 "索引 {index_name} 已创建"
    - 若索引已存在，则返回 "索引 {index_name} 已存在"
    """
    if es.indices.exists(index=index_name):
        return f"索引 {index_name} 已存在"

    mapping_body = {
        "settings": {
            "index": {
                "number_of_shards": shards,  
                "number_of_replicas": replicas  
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "para": {"type": "text"},  
                "book": {"type": "text"},
                "doc_type": {"type": "keyword"},
                "doc_id": {"type": "keyword"},  
                "source": {"type": "text"},
                "title_vector": {"type": "dense_vector", "dims": dims},  
                "para_vector": {"type": "dense_vector", "dims": dims}  
            }
        }
    }

    es.indices.create(index=index_name, body=mapping_body)
    return f"索引 {index_name} 已创建"



def generate_actions(file_path, index_name):
    """
    读取文件 file_path，将其转换为 Elasticsearch 批量插入的格式。
    
    参数:
    - file_path: 存储数据的 JSON 文件路径
    - index_name: 目标索引名称

    生成:
    - 每次返回一个字典，格式适配 helpers.bulk(...)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            action_line = f.readline().strip()  # 读取第一行（索引元数据）
            if not action_line: 
                break  # 读取到文件结尾
            if not action_line:
                continue  # 跳过空行
            
            try:
                action_data = json.loads(action_line)  # 解析第一行 JSON
                index_meta = action_data.get("index")
                if not index_meta:              
                    continue  # 如果第一行数据格式不符，则跳过
                
                doc_line = f.readline().strip()  # 读取第二行（文档内容）
                if not doc_line:
                    break  # 如果没有文档内容，说明文件格式不完整
                doc_data = json.loads(doc_line)

                # 生成符合 helpers.bulk 格式的数据
                yield {
                    "_op_type": "index",
                    "_index": index_meta["_index"], 
                    "_source": doc_data
                }
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                continue
        
            
def bulk_import(es: Elasticsearch, file_path: str, index_name: str):
    """
    使用 Elasticsearch 的 helpers.bulk 进行批量导入。

    参数:
    - es: Elasticsearch 客户端实例
    - file_path: 存储 JSON 数据的文件路径
    - index_name: 目标索引名称

    返回:
    - 导入成功的文档数量
    - 如果有错误，返回错误信息
    """
    try:
        success, errors = helpers.bulk(es, generate_actions(file_path, index_name))
        print(f"成功写入 {success} 条记录")
        if errors:
            print("存在错误：", errors)
        return success, errors
    except Exception as e:
        print(f"批量导入失败: {e}")
        return 0, str(e)