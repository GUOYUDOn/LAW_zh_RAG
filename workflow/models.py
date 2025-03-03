from dotenv import load_dotenv
import os
import torch
from datetime import datetime
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.messages import HumanMessage

# print("Starting to load the local model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model_path = 'to//your//path'
BGE_embedding_model = SentenceTransformer(embedding_model_path, device=device)
reranker_model_path = 'to//your//path'
BGE_reranker_model = CrossEncoder(reranker_model_path, device=device)

# print("Local model loaded successfully.")

def get_model(
        size: str = "medium",  # 可选: "large", "medium", "small"
        provider: str = "qwen",  # 可选: "qwen", "openai", "deepseek"
        streaming: bool = True,
        callbacks=None) -> ChatOpenAI:
    """
    获取模型实例，根据 provider 和 size 获取对应的模型名称和 API Key。
    """
    model_mapping = {
        "openai": {
            "large": "gpt-4-turbo",
            "medium": "gpt-3.5-turbo",
            "small": "gpt-3.5-turbo-16k"
        },
        "deepseek": {
            "large": "deepseek-chat",
            "medium": "deepseek-chat",
            "small": "deepseek-chat"
        },
        "qwen": {
            "large": "qwen2.5-72b-instruct",
            "medium": "qwen2.5-32b-instruct",
            "small": "qwen2.5-7b-instruct"
        }
    }

    api_bases = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1"  
    }

    load_dotenv()
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "qwen": os.getenv("QWEN_API_KEY")
    }

    if provider not in model_mapping:
        raise ValueError(f"Unsupported provider '{provider}'. Choose from 'openai', 'deepseek', 'qwen'.")

    if size not in model_mapping[provider]:
        raise ValueError(f"Unsupported size '{size}'. Choose from 'large', 'medium', 'small'.")

    model_name = model_mapping[provider][size]
    
    api_key = api_keys[provider]
    if api_key is None:
        raise ValueError(f"API Key for provider '{provider}' is not set. Please set it in environment variables.")

    api_base = api_bases[provider]

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        streaming=streaming,
        callbacks=callbacks
    )
    
def get_embedding(query_text : str):
    return BGE_embedding_model.encode(query_text, normalize_embeddings=True)

def get_rerank():
    return BGE_reranker_model

def test_api(provider, size):   
    try:
        start_time = datetime.now()  
        model = get_model(size=size, provider=provider)
        response = model.invoke([HumanMessage(content="Please reply with exactly 'Yes' and nothing else.")])
        duration = (datetime.now() - start_time).total_seconds()  
        print(f"✅ Success: {provider} - {size} -> Response: {response.content} (Duration: {duration:.2f}s)")
        return True
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()  
        print(f"❌ Failed: {provider} - {size} -> {str(e)} (Duration: {duration:.2f}s)")
        return False
    
    
if __name__ == "__main__":
    # 测试API是否可用
    providers = ["openai", "deepseek", "qwen"]
    sizes = ["large", "medium", "small"]

    for provider in providers:
        for size in sizes:
            test_api(provider, size)

