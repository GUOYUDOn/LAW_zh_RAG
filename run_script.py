import argparse

from workflow.chain import run_workflow
from workflow.conversation import ConversationManager

current_conversation_id = None

def chat_loop(user_id: str):
    """
    交互式聊天循环，保持 conversation_id 不变
    """
    
    global current_conversation_id
    conversation_manager = ConversationManager()
    current_conversation_id = conversation_manager.start_new_conversation(user_id)
    
    print(f"对话已启动，用户 ID: {user_id}，对话 ID: {current_conversation_id}")
    print("输入 'exit' 退出对话")

    while True:
        try:
            user_input = input("\nuser: ")
            if user_input.lower() == "exit":
                print("对话结束，再见！")
                break

            response_generator = run_workflow(
                user_input=user_input,
                user_id=user_id,
                conversation_manager=conversation_manager,
                conversation_id=current_conversation_id
            )

            print("\nlaw_AI:", end=" ", flush=True)
            for token in response_generator:
                print(token, end="", flush=True)
            print() 

        except KeyboardInterrupt:
            print("\n用户中断对话。")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律咨询AI系统")
    parser.add_argument("--user_id", type=str, default="user", help="用户ID（可选）")
    args = parser.parse_args()

    chat_loop(args.user_id)
