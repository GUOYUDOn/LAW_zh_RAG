import re
from langchain_core.messages import HumanMessage
from langchain.output_parsers import BooleanOutputParser
from langchain.callbacks.base import BaseCallbackHandler

from models import get_model, get_eval_model
from prompt import CHECK_LAW_PROMPT, CHECK_RELEVANCE_PROMPT, ENHANCE_QUESTION_PROMPT, FORMALIZE_QUESTION_PROMPT, GENERATION_PROMPT
from prompt import BASELINE_GENERATION_PROMPT, EVAL_SINGLE_PROMPT, EVAL_END2END_PROMPT, EVAL_ACCURACY_END2END_PROMPT

def identify_intent(query_text : str):
    '''
    意图识别
    '''
    model_identify = get_model(size="medium")
    prompt = CHECK_LAW_PROMPT.format(question=query_text)
    response = model_identify.invoke([HumanMessage(content=prompt)]).content
    parser = BooleanOutputParser()
    return parser.parse(response)

def judge_relevance(query_text : str, prev_query_text : str):
    '''
    判断当前问题是否与前一个问题相关
    '''
    model_judge = get_model(size="large")
    prompt = CHECK_RELEVANCE_PROMPT.format(question=query_text, prev_question=prev_query_text)
    response = model_judge.invoke([HumanMessage(content=prompt)]).content
    parser = BooleanOutputParser()
    return parser.parse(response)

def enhance_question(query_text : str, prev_query_text : str):
    '''
    对当前问题进行指代补全与增强
    '''
    model_enhance = get_model(size="medium")
    prompt = ENHANCE_QUESTION_PROMPT.format(query_text=query_text, prev_query_text=prev_query_text)
    response = model_enhance.invoke([HumanMessage(content=prompt)]).content
    return response

def formalize_question(query_text : str):
    '''
    将用户输入的问题转换为正式书面表达，并提取关键词
    '''
    model_formalize = get_model(size="medium")
    prompt = FORMALIZE_QUESTION_PROMPT.format(query_text=query_text)
    response = model_formalize.invoke([HumanMessage(content=prompt)]).content
    return response


class StreamCallbackHandler(BaseCallbackHandler):
    """
    自定义的流式输出回调处理器，用于将模型的输出逐步保存，然后被上层调用时 yield 出去。
    """
    def __init__(self):
        super().__init__()
        self.buffer = ""
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当模型生成一个新的 token 时会调用此方法。"""
        self.buffer += token

    def on_llm_end(self, response, **kwargs) -> None:
        """当本次生成结束时会调用此方法。"""
        self.done = True

    def fetch_and_clear(self) -> str:
        """获取当前 buffer 中的所有 token 并清空 buffer。"""
        content = self.buffer
        self.buffer = ""
        return content
    
    
def generate_response(query_text: str, refer_text: str):
    """
    流式生成最终回答
    """
    prompt = GENERATION_PROMPT.format(query_text=query_text, refer_text=refer_text)
    stream_handler = StreamCallbackHandler()
    model_generate = get_model(size="large", provider="qwen", streaming=True, callbacks=[stream_handler])

    for chunk in model_generate.stream([HumanMessage(content=prompt)]):
        yield chunk.content  


def generate_response_without_stream(query_text: str, refer_text: str):
    """
    非流式输出生成最终回答
    """
    prompt = GENERATION_PROMPT.format(query_text=query_text, refer_text=refer_text)
    model_generate = get_model(size="large", provider="qwen", streaming=False)
    response = model_generate.invoke([HumanMessage(content=prompt)]).content
    return response
      
        
def eval_single_question(query_text: str, refer_text: str):
    """
    评估单轮对话检索结果
    """
    model_eval = get_eval_model(streaming=False)
    prompt = EVAL_SINGLE_PROMPT.format(query_text=query_text, refer_text=refer_text)
    response = model_eval.invoke([HumanMessage(content=prompt)]).content
    match = re.search(r'\b([0-5])\b', response)  # 匹配独立的 0-5 数字
    return int(match.group(1)) if match else 0  


def generate_baseline_response(query_text: str):
    """
    基线模型生成回答
    """
    model_baseline = get_model(size="large")
    prompt = BASELINE_GENERATION_PROMPT.format(query_text=query_text)
    response = model_baseline.invoke([HumanMessage(content=prompt)]).content
    return response


def eval_end2end(query_text: str, response1: str, response2: str):
    """
    端到端评估基线模型与RAG模型结果
    """
    model_eval = get_eval_model(streaming=False)
    prompt = EVAL_END2END_PROMPT.format(query_text=query_text, response1=response1, response2=response2)
    response = model_eval.invoke([HumanMessage(content=prompt)]).content
    return response


def eval_end2end_accuracy(query_text: str, answer_text: str, response: str):
    """
    端到端RAG模型准确率评估
    """
    model_eval = get_eval_model(streaming=False)
    prompt = EVAL_ACCURACY_END2END_PROMPT.format(query_text=query_text, answer_text=answer_text, response=response)
    response = model_eval.invoke([HumanMessage(content=prompt)]).content
    parser = BooleanOutputParser()
    return parser.parse(response)

# if __name__ == "__main__":
#     print(identify_intent("今天天气怎么样"))
#     print(identify_intent("劳动合同到期后，公司是否需要赔偿？"))
#     print(judge_relevance("劳动合同到期后，公司是否需要赔偿？", "今天天气怎么样"))
#     print(judge_relevance("如果不赔偿我该怎么办", "劳动合同到期后，公司是否需要赔偿？"))
#     print(enhance_question("如果不赔偿我该怎么办", "劳动合同到期后，公司是否需要赔偿？"))
#     print(formalize_question("劳动合同到期后，公司是否需要赔偿呀？"))
#     print(formalize_question("劳动合同到期后公司不进行赔偿a，我该怎么办"))
#     print(generate_baseline_response("张三在自家后院挖地窖，强迫李四每天直播半小时吃螺蛳粉，这个行为构成非法拘禁罪吗？"))