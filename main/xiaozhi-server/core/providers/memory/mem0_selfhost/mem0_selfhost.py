# -*- coding: utf-8 -*-
"""
Mem0 自托管版本 - 本地部署的长期记忆系统
使用 Qdrant 本地文件存储 + DashScope Embedding + DashScope LLM
无需额外服务，直接集成到 xiaozhi-server 进程中
"""
import traceback
from ..base import MemoryProviderBase, logger

TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory=None):
        super().__init__(config)
        self.use_memory = False

        dashscope_api_key = config.get("dashscope_api_key", "")
        if not dashscope_api_key:
            logger.bind(tag=TAG).error("mem0_selfhost: dashscope_api_key 未配置")
            return

        qdrant_path = config.get("qdrant_path", "./data/qdrant_memory")
        collection_name = config.get("collection_name", "xiaozhi_memory")
        llm_model = config.get("llm_model", "qwen-plus")
        embedding_model = config.get("embedding_model", "text-embedding-v3")
        embedding_dims = config.get("embedding_dims", 1024)

        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": qdrant_path,
                    "collection_name": collection_name,
                    "embedding_model_dims": embedding_dims,
                    "on_disk": True,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": dashscope_api_key,
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model,
                    "api_key": dashscope_api_key,
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "embedding_dims": embedding_dims,
                }
            },
            "version": "v1.1",
        }

        try:
            from mem0 import Memory
            self.memory_client = Memory.from_config(mem0_config)
            self.use_memory = True
            logger.bind(tag=TAG).info(
                f"mem0_selfhost 初始化成功，Qdrant路径: {qdrant_path}, "
                f"LLM: {llm_model}, Embedder: {embedding_model}"
            )
        except Exception as e:
            logger.bind(tag=TAG).error(f"mem0_selfhost 初始化失败: {str(e)}")
            logger.bind(tag=TAG).error(traceback.format_exc())

    async def save_memory(self, msgs, session_id=None):
        if not self.use_memory:
            return None
        if len(msgs) < 2:
            return None
        try:
            messages = [
                {"role": message.role, "content": message.content}
                for message in msgs
                if message.role != "system"
            ]
            result = self.memory_client.add(messages, user_id=self.role_id)
            logger.bind(tag=TAG).debug(f"Save memory result: {result}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存记忆失败: {str(e)}")
            return None

    async def query_memory(self, query: str) -> str:
        if not self.use_memory:
            return ""
        if not query or not query.strip():
            return ""
        try:
            if not getattr(self, "role_id", None):
                return ""

            results = self.memory_client.search(query, user_id=self.role_id)
            if not results or "results" not in results:
                return ""

            memories = []
            for entry in results["results"]:
                timestamp = entry.get("updated_at", "")
                memory = entry.get("memory", "")
                if memory:
                    if timestamp:
                        try:
                            dt = timestamp.split(".")[0]
                            formatted_time = dt.replace("T", " ")
                        except Exception:
                            formatted_time = timestamp
                        memories.append((timestamp, f"[{formatted_time}] {memory}"))
                    else:
                        memories.append(("", memory))

            memories.sort(key=lambda x: x[0], reverse=True)
            memories_str = "\n".join(f"- {m[1]}" for m in memories)
            logger.bind(tag=TAG).debug(f"Query results: {memories_str}")
            return memories_str
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询记忆失败: {str(e)}")
            return ""
