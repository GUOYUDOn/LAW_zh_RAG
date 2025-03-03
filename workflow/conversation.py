import uuid
import time
from collections import deque
from typing import List, Dict, Optional

class Conversation:
    def __init__(self, max_rounds: int = 5):
        self.id: str = str(uuid.uuid4())  # 生成唯一会话 ID
        self.history: deque = deque(maxlen=max_rounds)  # 存储对话历史
        self.last_activity_time: float = time.time()  # 记录最后活跃时间

    def add_turn(self, user_msg: str, enhanced_user_msg : str, assistant_msg: str, ref_texts: Optional[List[str]] = None):
        self.history.append({
            "user": user_msg,
            "enhanced_user": enhanced_user_msg,
            "assistant": assistant_msg,
            "ref_texts": ref_texts if ref_texts else []
        })
        self.last_activity_time = time.time()

    def get_history(self) -> List[Dict]:
        return list(self.history)

    def clear(self):
        self.history.clear()


class ConversationManager:
    def __init__(self, max_rounds: int = 5, timeout_minutes: int = 30):
        self._user_conversations: Dict[str, Dict[str, Conversation]] = {}  # 用户 -> 会话列表
        self.max_rounds = max_rounds
        self.timeout_minutes = timeout_minutes

    def start_new_conversation(self, user_id: str) -> str:
        """
        为指定用户创建新会话，并返回会话 ID。
        """
        if user_id not in self._user_conversations:
            self._user_conversations[user_id] = {}

        conv = Conversation(max_rounds=self.max_rounds)
        self._user_conversations[user_id][conv.id] = conv
        return conv.id

    def add_turn(self, user_id: str, conversation_id : str, user_msg : str, enhanced_user_msg : str, assistant_msg : str, ref_texts : Optional[List[str]] = None):
        """
        在指定用户的指定会话中添加一轮对话。
        """
        if user_id not in self._user_conversations or conversation_id not in self._user_conversations[user_id]:
            raise ValueError(f"Session {conversation_id} does not exist or user {user_id} has no access to this session")
        
        self._user_conversations[user_id][conversation_id].add_turn(user_msg, enhanced_user_msg, assistant_msg, ref_texts)

    def get_history(self, user_id: str, conversation_id: str) -> List[Dict]:
        """
        获取指定用户的指定会话历史记录。
        """
        if user_id not in self._user_conversations or conversation_id not in self._user_conversations[user_id]:
            return []
        return self._user_conversations[user_id][conversation_id].get_history()

    def clear_conversation(self, user_id: str, conversation_id: str):
        """
        清空某个用户的某个会话的所有记录。
        """
        if user_id in self._user_conversations and conversation_id in self._user_conversations[user_id]:
            self._user_conversations[user_id][conversation_id].clear()

    def clear_user_conversations(self, user_id: str):
        """
        清除某个用户的所有会话。
        """
        if user_id in self._user_conversations:
            self._user_conversations[user_id].clear()

    def cleanup_expired_conversations(self):
        """
        清理所有 30 分钟内无活动的会话。
        """
        current_time = time.time()
        to_remove = []

        for user_id, convs in self._user_conversations.items():
            for conv_id, conv in list(convs.items()):
                if (current_time - conv.last_activity_time) >= self.timeout_minutes * 60:
                    to_remove.append((user_id, conv_id))

        for user_id, conv_id in to_remove:
            self._user_conversations[user_id].pop(conv_id, None)
            if not self._user_conversations[user_id]:  # 如果用户的会话都删除了，清空整个用户的 key
                self._user_conversations.pop(user_id, None)

    def clear_all_conversations(self):
        """
        清空所有用户的所有会话。
        """
        self._user_conversations.clear()

    def get_all_user_conversations(self, user_id: str) -> List[str]:
        """
        获取指定用户的所有会话 ID。
        """
        return list(self._user_conversations.get(user_id, {}).keys())


if __name__ == "__main__":
    manager = ConversationManager(max_rounds=5, timeout_minutes=30)

    # 1. 创建用户 A 的新会话
    user_id_a = "user_123"
    cid_a1 = manager.start_new_conversation(user_id_a)
    print(f"用户 {user_id_a} 新会话 ID: {cid_a1}")

    # 2. 创建用户 B 的新会话
    user_id_b = "user_456"
    cid_b1 = manager.start_new_conversation(user_id_b)
    print(f"用户 {user_id_b} 新会话 ID: {cid_b1}")

    # 3. 添加对话
    manager.add_turn(user_id_a, cid_a1, user_msg="你好", enhanced_msg="你好！",assistant_msg="你好！", ref_texts="doc1")
    manager.add_turn(user_id_a, cid_a1, user_msg="天气怎么样",enhanced_msg="天气如何?", assistant_msg="今天天气不错", ref_texts="doc2")

    manager.add_turn(user_id_b, cid_b1, user_msg="讲个笑话", enhanced_msg="讲述一个笑话", assistant_msg="为什么鸡过马路？因为对面更好玩！", ref_texts=[])

    # 4. 获取用户 A 的对话历史
    history_a1 = manager.get_history(user_id_a, cid_a1)
    print(f"用户 {user_id_a} 的会话 {cid_a1} 历史记录: {history_a1}")

    # 5. 获取用户 B 的对话历史
    history_b1 = manager.get_history(user_id_b, cid_b1)
    print(f"用户 {user_id_b} 的会话 {cid_b1} 历史记录: {history_b1}")

    # 6. 清除用户 A 的会话
    manager.clear_conversation(user_id_a, cid_a1)
    print(manager.get_history(user_id_a, cid_a1))

    # 7. 一键清空所有会话
    # manager.clear_all_conversations()
