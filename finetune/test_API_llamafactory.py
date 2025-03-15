import requests

def get_available_models(server_url: str = "http://70.69.205.56:55436"):
    """
    获取远程服务器上的可用模型列表。
    """
    url = f"{server_url}/v1/models"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json()
        return models.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return []



def chat_with_model(user_input: str,
                    server_url: str = "http://70.69.205.56:55436", 
                    model_name: str = "gpt-3.5-turbo"):
    """
    发送问题到远程服务器上的模型，并返回答案。
    """
    url = f"{server_url}/v1/chat/completions"

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_input}],
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "max_tokens": 2048,
        "stop": None,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"
    
    
if __name__ == "__main__":
    # models = get_available_models()
    # print("可用模型:", models)
    
    while True:
        user_input = input("\n请输入你的问题 (输入 'exit' 退出): ")
        if user_input.lower() == "exit":
            print("对话已结束。")
            break
        response = chat_with_model(user_input)
        print("\nAssistant:", response)