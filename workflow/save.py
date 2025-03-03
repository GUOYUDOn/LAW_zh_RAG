from pathlib import Path
import json

def save_as_json(documents, index_name="law_data", output_path="data_book/lawbook.json"):
    """
    将文档列表转换为 Elasticsearch 格式的 NDJSON 文件，并保存到本地。

    参数:
    - documents: 需要保存的文档列表（通常是 split_documents_mk）。
    - index_name: 索引名称，默认为 "law_data"。
    - output_path: 输出文件路径，默认为 "data_book/test2.json"。

    返回:
    - 无（但会在本地生成 NDJSON 格式的 JSON 文件）。
    """
    bulk_data = []

    for doc in documents:
        index_line = {
            "index": {
                "_index": index_name,
            }
        }

        # 处理向量数据
        title_vector = doc.metadata.get("title_vector", [])
        if hasattr(title_vector, "tolist"):
            title_vector = title_vector.tolist()

        para_vector = doc.metadata.get("para_vector", [])
        if hasattr(para_vector, "tolist"):
            para_vector = para_vector.tolist()

        data_line = {
            "title": doc.metadata.get("title", ""),
            "para": doc.page_content,
            "book": doc.metadata.get("book", ""),
            "doc_type": doc.metadata.get("doc_type", ""),
            "source": doc.metadata.get("source", ""),
            "title_vector": title_vector,
            "para_vector": para_vector
        }

        bulk_data.append(index_line)
        bulk_data.append(data_line)

    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 将数据写入文件
    with output_file.open("w", encoding="utf-8") as f:
        for line in bulk_data:
            f.write(json.dumps(line, ensure_ascii=False))
            f.write("\n")

    print(f"Bulk data has been written to {output_file.absolute()}")
