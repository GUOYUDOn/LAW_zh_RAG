import logging
from typing import Optional, Generator
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


from conversation import ConversationManager
from generation import identify_intent, judge_relevance, enhance_question, formalize_question, generate_response
from retriever import es_search, convert_es_to_documents, rerank_documents, merge_documents, WebRetriever
from start_es import es
from models import get_model, get_embedding, get_rerank

REFUSED_RESPONSE = "对不起，我是一位法律顾问，仅能回答与法律相关的问题。请提供一个与法律相关的提问，我会尽力帮助您！"
    

def run_workflow(user_input: str,
                 user_id: str = "user", 
                 conversation_manager: Optional[ConversationManager] = None,  
                 conversation_id: Optional[str] = None,
                 top_k_es: int = 20,
                 top_k_web: int = 2,
                 top_k_rerank: int = 4) -> Generator[str, None, None]:
    """完整的工作流实现"""
    
    if conversation_manager is None: 
        conversation_manager = ConversationManager()
    
    if not conversation_id:
        conversation_id = conversation_manager.start_new_conversation(user_id)

    logging.basicConfig(
    handlers=[logging.FileHandler("run.log", encoding="utf-8")],  
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
    history = conversation_manager.get_history(user_id, conversation_id)
    logging.info(f"User({user_id}) Conversation({conversation_id}) History: {history}")

        
    if len(history) == 0:  # 该会话的首次提问
        is_law_related = identify_intent(user_input)
        
        if not is_law_related:
            conversation_manager.add_turn(user_id, conversation_id, user_input, enhanced_user_msg="", assistant_msg=REFUSED_RESPONSE, ref_texts=[])
            yield REFUSED_RESPONSE
            return
        
        enhanced_query = formalize_question(user_input)
    
    else:   # 处理非首次提问
        last_enhanced_query = history[-1].get("enhanced_user", "") 
        is_prev_related = judge_relevance(user_input, last_enhanced_query)
        
        if is_prev_related:
            enhanced_query = enhance_question(user_input, last_enhanced_query)
            enhanced_query = formalize_question(enhanced_query)
        
        else:
            is_law_related = identify_intent(user_input)
            if not is_law_related:
                conversation_manager.add_turn(user_id, conversation_id, user_input, enhanced_user_msg="", assistant_msg=REFUSED_RESPONSE, ref_texts=[])
                yield REFUSED_RESPONSE
                return
            enhanced_query = formalize_question(user_input)
        
    refer_text = retrieve_and_merge(enhanced_query, top_k_es=top_k_es, top_k_web=top_k_web, top_k_rerank=top_k_rerank)
    
    response = generate_response(user_input, refer_text)
    
    full_ans_list = []
    for token in response:
        full_ans_list.append(token)
        yield token
    full_ans = "".join(full_ans_list)
    
    conversation_manager.add_turn(user_id, conversation_id, user_input, enhanced_user_msg=enhanced_query, assistant_msg=full_ans, ref_texts=[refer_text])
    
    
    
def retrieve_and_merge(query_text: str, top_k_es: int = 20, top_k_web: int = 2, top_k_rerank: int = 4) -> str:
    """ES检索 + 网络检索 + 交叉编码器重排 -> 合并文本"""
    
    query_vector = get_embedding(query_text)
    es_results = es_search(es, query_vector, index_name="law_data", top_k=top_k_es)
    es_docs = convert_es_to_documents(es_results)

    search_api = DuckDuckGoSearchAPIWrapper()
    web_retriever = WebRetriever(search_api, num_search_results=top_k_web)
    web_docs = web_retriever.invoke(query_text)  

    docs_all = es_docs + web_docs
    sorted_docs = rerank_documents(query_text, docs_all, rerank_model=get_rerank())
    merged_text = merge_documents(sorted_docs, top_k=top_k_rerank)
    
    return merged_text
        
        
if __name__ == "__main__":
    for token in run_workflow("张三在自家后院挖地窖，强迫李四每天直播半小时吃螺蛳粉，这个行为构成非法拘禁罪吗？"):
        print(token, end="", flush=True) 