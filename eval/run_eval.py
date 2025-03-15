import argparse
import csv
import json
from tqdm import tqdm

from eval import load_random_questions, load_random_questions_with_answers, eval_chain, get_retriever_results, get_eval, single_turn_generation, single_turn_generation_finetune
from generation import eval_end2end_accuracy, generate_baseline_response, generate_baseline_response_without_RAG

def eval1(question_file_path: str = "data_book//3-8.json",
          question_nums: int = 100,
          top_k_es: int = 20, 
          top_k_web: int = 1, 
          top_k_rerank: int = 4,
          is_store: bool = True,
          output_file_path: str = "data_book//single_turn_eval_results.csv"):
    '''
    进行最终单轮对话检索结果的测试
    '''
    # 注意：修改top_k_rerank参数的时候要修改PROMPT中的数字！
    
    questions = load_random_questions(question_file_path, question_nums, seed = 42)
    print(f"Loaded {len(questions)} questions.")
    
    print("Starting question enhancement...")
    enhanced_questions = eval_chain(questions)
    print("Question enhancement completed.")

    print("Starting retrieval results generation...")
    retrieval_results = get_retriever_results(enhanced_questions, top_k_es, top_k_web, top_k_rerank)
    print("Retrieval results generated.")
    
    print("Starting evaluation...")
    global_hit, global_precision, eval_data = get_eval(questions, enhanced_questions, retrieval_results, top_k_rerank)
    print("Global Hit Score:", global_hit)
    print("Global Precision Score:", global_precision)
    
    if is_store:
        print("Storing results to CSV...")
        csv_filename = output_file_path
        
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["original_question", "enhanced_question", "retrieval_results", "hit", "precision"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in eval_data:
                writer.writerow({
                    "original_question": item["original_question"],
                    "enhanced_question": item["enhanced_question"],
                    "retrieval_results": item["retrieval_results"],
                    "hit": item["hit"],
                    "precision": item["precision"]
                })
        print("Results saved to", csv_filename)
    else:
        print("Skipping CSV storage as per parameter.")

    return global_hit, global_precision



def eval2(question_file_path: str = "data_book//3-8.json",
          question_nums: int = 100,
          top_k_es: int = 20, 
          top_k_web: int = 2, 
          top_k_rerank: int = 5,
          is_store: bool = True,
          output_file_path: str = "data_book//eval_end2end.json"):
    '''
    测评端到端模型的准确率，并将结果存入JSON文件中。
    '''
    pairs_qa = load_random_questions_with_answers(question_file_path, question_nums, seed = 42)
    print(f"Loaded {len(pairs_qa)} question-answer pairs.")

    print("Starting evaluation...")
    eval_result = []
    for pair_qa in tqdm(pairs_qa):
        question = pair_qa["question"]
        answer = pair_qa["answer"]
        response = single_turn_generation(question, top_k_es, top_k_web, top_k_rerank)
        score = eval_end2end_accuracy(question, answer, response)
        eval_result.append({"question": question, 
                            "answer": answer, 
                            "response": response, 
                            "score": score})
    
    
    global_accuracy = sum([item["score"] for item in eval_result]) / len(eval_result)
    print("(End to End Mode) Global Accuracy Score:", global_accuracy)
    
    if is_store:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to {output_file_path}")

    return eval_result, global_accuracy    


def eval3(question_file_path: str = "data_book//3-8.json",
          question_nums: int = 100,
          is_store: bool = True,
          output_file_path: str = "data_book//eval_end2end_baseline.json"):
    '''
    对照组，测评基座模型的正确率。
    '''
    pairs_qa = load_random_questions_with_answers(question_file_path, question_nums, seed = 42)
    print(f"Loaded {len(pairs_qa)} question-answer pairs.")

    print("Starting evaluation...")
    eval_result = []
    for pair_qa in tqdm(pairs_qa):
        question = pair_qa["question"]
        answer = pair_qa["answer"]
        response = generate_baseline_response(question)
        score = eval_end2end_accuracy(question, answer, response)
        eval_result.append({"question": question, 
                            "answer": answer, 
                            "response": response, 
                            "score": score})
    
    
    global_accuracy = sum([item["score"] for item in eval_result]) / len(eval_result)
    print("(End to End Mode) Global Accuracy Score:", global_accuracy)
    
    if is_store:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to {output_file_path}")

    return eval_result, global_accuracy 



def eval4(question_file_path: str = "data_book//3-8.json",
          question_nums: int = 100,
          is_store: bool = True,
          output_file_path: str = "data_book//eval_end2end_finetune.json"):
    '''
    测评微调后模型的准确率，不使用RAG。
    '''
    pairs_qa = load_random_questions_with_answers(question_file_path, question_nums, seed = 56)
    print(f"Loaded {len(pairs_qa)} question-answer pairs.")

    print("Starting evaluation...")
    eval_result = []
    for pair_qa in tqdm(pairs_qa):
        question = pair_qa["question"]
        answer = pair_qa["answer"]
        # response = generate_baseline_response_without_RAG(question)
        response = single_turn_generation_finetune(question)
        score = eval_end2end_accuracy(question, answer, response)
        eval_result.append({"question": question, 
                            "answer": answer, 
                            "response": response, 
                            "score": score})
    
    
    global_accuracy = sum([item["score"] for item in eval_result]) / len(eval_result)
    print("(End to End Mode) Global Accuracy Score:", global_accuracy)
    
    if is_store:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to {output_file_path}")

    return eval_result, global_accuracy  


# if __name__ == "__main__":
    # eval1(question_nums = 300, top_k_rerank = 4)   # 单轮对话检索能力测评
    
# if __name__ == "__main__":
#     eval2(question_nums = 100, top_k_rerank = 5)

# if __name__ == "__main__":
#     eval3(question_nums = 100)

# if __name__ == "__main__":
#     eval4(question_nums = 100, is_store = True, output_file_path = "data_book//eval_end2end-0.5b.json")