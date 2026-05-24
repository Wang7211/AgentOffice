"""共享的单例记忆实例。"""

from memory.chat_memory import ChatMemory
from memory.vector_memory import LocalVectorMemory


chat_memory = ChatMemory()
vector_memory = LocalVectorMemory()
agent_memory = LocalVectorMemory("agent_memory_index.json")
