import json
import pandas as pd
from typing import List

def Json2Alpaca(input_file_path: str , output_file_path: str):
    """
    将单轮对话的JSON文件转成Alpaca格式。
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    
    print(f'Starting conversion, target file: {input_file_path}')
    converted_datas = []
    for data in datas:
        instruction = data.get('input', '')
        output = data.get('output', '')
        converted_data = {
            "instruction": instruction,
            "input": "",
            "output": output
        }
        converted_datas.append(converted_data)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_datas, f, ensure_ascii=False, indent=2)    
    print(f'Conversion completed, the output file has been saved to {output_file_path}.')
    

def Csv2Alpaca(input_file_path: str , output_file_path: str):
    """
    将单轮对话的CSV文件转成Alphaca格式。
    """
    df = pd.read_csv(input_file_path)
    print(f'Starting conversion, target file: {input_file_path}')
    
    df_filtered = df[df['is_best'] == 1]
    converted_datas = []
    for _, row in df_filtered.iterrows():
        if pd.notna(row['question']) and row['question'].strip():
            instruction = row['question']
        else:
            instruction = row['title']
        output = row['reply']
        converted_data = {
            "instruction": instruction,
            "input": "",
            "output": output
        }
        converted_datas.append(converted_data)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_datas, f, ensure_ascii=False, indent=2)
    print(f'Conversion completed, the output file has been saved to {output_file_path}.')
        
        
def Json2ShareGPT(input_file_path: str , output_file_path: str):
    """
    将多轮对话的JSON文件转成ShareGPT格式。
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    print(f'Starting conversion, target file: {input_file_path}')
    converted_datas = []
    for data in datas:
        conversations = []    
        for line in data.get('history', []):
            if line.startswith('[客户]'):
                conversations.append({
                    'from': 'user',
                    'value': line.replace("[客户] ", "")
                })
            elif line.startswith('[律师]'):
                conversations.append({
                    'from': 'assistant',
                    'value': line.replace("[律师] ", "")
                })
        converted_data = {
            'conversations': conversations,
            'system': '法律咨询助手'
        }
        converted_datas.append(converted_data)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_datas, f, ensure_ascii=False, indent=2)
    print(f'Conversion completed, the output file has been saved to {output_file_path}.')
    
    
def merge_json(input_file_paths: List[str], output_file_path: str):
    """
    将多个JSON文件合并成一个JSON文件。
    """
    merged_data = []
    print(f'Starting merge, target files : {len(input_file_paths)}')
    
    for input_file_path in input_file_paths:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f'Warning: {input_file_path} is not a JSON file.')
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f'Merge completed, the output file has been saved to {output_file_path}.')
    
            

if __name__ == '__main__':
    
    input_file_path1 = 'data_book//finetune_data//CrimeKgAssitant_lawQA.json'   # 单轮
    input_file_path2 = 'data_book//finetune_data//FAQ.csv'   # 单轮
    input_file_path3 = 'data_book//finetune_data//alpaca_gpt4_data_zh.json'    # 单轮，alpaca格式
    input_file_path4 = 'data_book//finetune_data//CivilCode_selfinstruct4o.json'   # 单轮，alpaca格式
    input_file_path5 = 'data_book//finetune_data//legal_counsel_multi_turn_with_article_v2.json'   # 多轮
    
    
    output_file_path1 = 'data_book//finetune_data//CrimeKgAssitant_alpaca.json'
    output_file_path2 = 'data_book//finetune_data//FAQ_alpaca.json'
    output_file_path3 = 'data_book//finetune_data//MultiTurn_ShareGPT.json'
    
    Json2Alpaca(input_file_path1, output_file_path1)
    print('-------------------------------------------------------------------------------------')
    Csv2Alpaca(input_file_path2, output_file_path2)
    print('-------------------------------------------------------------------------------------')
    Json2ShareGPT(input_file_path5, output_file_path3)