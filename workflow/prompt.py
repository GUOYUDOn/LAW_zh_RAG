from langchain.prompts import PromptTemplate 


# 检测用户提出的问题是否与法律问题有关
check_law_prompt_template = """
你是一个专业律师，请判断下面问题是否和法律相关，相关请回答yes，不相关请回答no，不允许其它回答，不允许在答案中添加编造成分。
问题: {question}
"""
CHECK_LAW_PROMPT = PromptTemplate(template=check_law_prompt_template, input_variables=["question"])


# 检测用户当前提出的问题是否与前一个问题有关
check_relevance_prompt_template = """
前一个问题是：" {prev_question} "  
当前问题是：" {question} "  
请你判断当前问题是否与前一个问题相关，并考虑以下情况：  
1. 当前问题是否是对前一个问题的补充、追问、澄清或延续？  
2. 当前问题是否包含指代词（如“这个行为”“这种情况”等），并且这些指代词所指代的内容可以从前一个问题中推断出来？  
3. 当前问题是否涉及与前一个问题相同的法律概念、事实背景或核心论点，即使措辞有所不同？  
如果满足上述任意一项，请回答yes。  
如果当前问题的主题、讨论方向完全不同，或者无法从前一个问题推断出其相关性，请回答no。注意只能回答yes或no，不允许其它回答。不允许编造信息，不允许添加额外解释或推测。
"""  

CHECK_RELEVANCE_PROMPT = PromptTemplate(template=check_relevance_prompt_template, input_variables=["question", "prev_question"])


# 对当前用户的问题进行指代补全与增强
enhance_question_prompt_template = """
你是一位专业的法律智能助手，任务是优化用户输入的问题，使其更加清晰、完整，并结合上一轮问题补全可能缺失的上下文。
先前的问题："{prev_query_text}"
当前的问题："{query_text}"
请基于上述两个问题：
1. 合并两个问题，避免重复表达。
2. 去重，删除冗余内容。
3. 补全语义，使问题更加完整，确保 AI 更容易理解。
你的输出：仅输出优化后的问题，不要包含解释或其他内容。
"""
ENHANCE_QUESTION_PROMPT = PromptTemplate(template=enhance_question_prompt_template, input_variables=["query_text", "prev_query_text"])


# 将提问变为正式化表达并提取关键词
formalize_question_prompt_template = """
你是一位专业的法律智能助手，任务是优化用户输入的问题，使其更加正式、精准，并提取核心关键词，以便后续检索。
用户原始问题：" {query_text} "
你的任务：
1. 转换为正式书面表达：去除口语化词汇（如 “啊”、“呢”、“吧”），调整语法，使问题更加专业、正式。
2. 提取关键词：找出该问题最核心的法律概念，确保可用于数据库检索。
3. 优化检索问题：基于正式化的问题，将关键词附加在末尾，形成适合检索的 Query。
请直接输出优化后的问题 + 问号 + 关键词，不要换行，关键词之间用逗号连接，不要添加额外解释或说明。
"""
FORMALIZE_QUESTION_PROMPT = PromptTemplate(template=formalize_question_prompt_template, input_variables=["query_text"])


#生成最终问题
generation_prompt_template = """
你是一位经验丰富的专业律师。请基于以下参考信息 {refer_text}，并结合你的专业知识，严谨、清晰地回答用户的问题：{query_text}。
要求：
1.法律依据：明确引用相关法律条款或案例，并提供解释。  
2.逻辑分析：结合实际情况进行专业的法律分析，避免含糊不清的回答。  
3.结论与建议：提供清晰、可执行的法律建议，并考虑不同情况的应对方案。  
请确保你的回答严谨、合规，并符合专业律师的表达方式。
"""
GENERATION_PROMPT = PromptTemplate(template=generation_prompt_template, input_variables=["query_text", "refer_text"])