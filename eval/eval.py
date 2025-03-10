from workflow.retriever import es_search, convert_es_to_documents, rerank_documents, WebRetriever, merge_documents
from workflow.generation import formalize_question, eval_single_question, identify_intent, generate_response_without_stream
from workflow.models import get_embedding, get_rerank
from workflow.start_es import es
from workflow.chain import REFUSED_RESPONSE, retrieve_and_merge

import json
import random
from typing import List
from tqdm import tqdm
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def load_random_questions(filepath="data_book//3-8.json", nums = 100, seed=42):
    '''
    随机选取问题（默认100个），用于测试
    '''
    random.seed(seed)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    random_selected_questions = random.sample(questions, min(nums, len(questions)))
    
    return random_selected_questions


def load_random_questions_with_answers(filepath="data_book//3-8.json", nums = 100, seed=42):
    '''
    随机选取问题与答案（默认100个），用于测试
    '''
    random.seed(seed)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qa_pairs = [{"question": item["question"], "answer": item.get("answer", "")} for item in data]
    random_selected_qa_pairs = random.sample(qa_pairs, min(nums, len(qa_pairs)))
    
    return random_selected_qa_pairs 


def eval_chain(questions: List):
    '''
    单轮对话效果增强，仅保留“提问增强部分”。
    '''
    enhenced_questions = []
    for question in tqdm(questions):
        curr_enhenced_question = formalize_question(question)
        enhenced_questions.append(curr_enhenced_question)
    
    return enhenced_questions


def get_retriever_results(enhenced_questions: List, top_k_es: int = 20, top_k_web: int = 2, top_k_rerank: int = 4):
    '''
    批量获取检索结果（无网络检索）
    '''
    refer_results = []
    # search_api = DuckDuckGoSearchAPIWrapper()
    # web_retriever = WebRetriever(search_api, num_search_results=top_k_web)
    
    for question in tqdm(enhenced_questions):
        question_vectors = get_embedding(question)
        es_results = es_search(es, question_vectors, index_name="law_data", top_k=top_k_es)
        es_docs = convert_es_to_documents(es_results)
        # web_docs = web_retriever.invoke(question)
        
        # docs_all = es_docs + web_docs
        docs_all = es_docs
        sorted_docs = rerank_documents(question, docs_all, rerank_model=get_rerank())
        curr_refer = sorted_docs[:top_k_rerank]
        
        refer_content = []   # 该问题的检索结果（重排序后）：List[document] -> List[str]
        for doc in curr_refer:
            content = doc.page_content
            metadata = doc.metadata
            
            title = metadata.get("title", "").strip()
            full_text = f"{title}:{content}" if title else content
            
            refer_content.append(full_text)
        refer_results.append(refer_content)
    
    return refer_results


def get_single_retriever_results(enhenced_question: str, top_k_es: int = 20, top_k_web: int = 2, top_k_rerank: int = 4):
    '''
    获取检索结果（输入为单个字符串，无网络检索）
    '''
    question_vectors = get_embedding(enhenced_question)
    
    es_results = es_search(es, question_vectors, index_name="law_data", top_k=top_k_es)
    es_docs = convert_es_to_documents(es_results)
    
    sorted_docs = rerank_documents(enhenced_question, es_docs, rerank_model=get_rerank())
    refer_content = merge_documents(sorted_docs, top_k=top_k_rerank)
    
    return refer_content
        
        
def get_eval(questions: List, enhanced_questions: List, refer_results: List, top_k_rerank: int = 5):
    '''
    进行评估，计算每个问题的得分、命中以及精确率，并返回全局命中率、全局精确率以及每个问题的评估数据。
    参数：
        questions: 原始问题列表
        enhanced_questions: 增强后的问题列表
        refer_results: 每个增强后问题对应的检索结果列表
    返回：
        global_hit: 全局命中率
        global_precision: 全局精确率
        eval_data: 包含每个问题评估详情的列表，每项为字典，包括原始问题、增强后的问题、检索结果文本、得分、hit、precision
    '''
    eval_data = []
    final_hit = []
    final_precision = []
    for orig, enh, refer_result in tqdm(zip(questions, enhanced_questions, refer_results), total=len(questions)):
        refer_txt = "\n".join([f"第{i+1}条检索结果：{text}" for i, text in enumerate(refer_result)])
        score = eval_single_question(orig, refer_txt)
        score = max(0, min(score, top_k_rerank))
        
        hit = 1 if score >= 1 else 0
        precision = score / len(refer_result) if len(refer_result) > 0 else 0
        
        final_hit.append(hit)
        final_precision.append(precision)
        
        eval_data.append({
            "original_question": orig,
            "enhanced_question": enh,
            "retrieval_results": refer_txt,
            "score": score,
            "hit": hit,
            "precision": precision
        })

    global_hit = sum(final_hit) / len(final_hit)
    global_precision = sum(final_precision) / len(final_precision)
    
    return global_hit, global_precision, eval_data


def single_turn_generation(question: str, top_k_es: int = 20, top_k_web: int = 2, top_k_rerank: int = 5):
    '''简易单轮对话工作流，非流式输出结果'''
    if not identify_intent(question):
        return REFUSED_RESPONSE
    else:
        enhenced_question = formalize_question(question)
        refer_text = get_single_retriever_results(enhenced_question, top_k_es, top_k_web, top_k_rerank)  # 无网络检索
        # refer_text = retrieve_and_merge(enhenced_question, top_k_es, top_k_web, top_k_rerank)
        reponse = generate_response_without_stream(enhenced_question, refer_text)
    return reponse





if __name__ == "__main__":
    print(single_turn_generation("张三在自家后院挖地窖，强迫李四每天直播半小时吃螺蛳粉，这个行为构成非法拘禁罪吗？"))