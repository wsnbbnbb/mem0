
import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import sys
# ensure parent directory is on path so `import AgentMem` finds the package
sys.path.append("/root/nfs/hmj/proj/mem0/evaluation")
from dotenv import load_dotenv
from tqdm import tqdm
from jinja2 import Template
from AgentMem import Memory
from AgentMem.configs.llms.base import BaseLlmConfig
from AgentMem.llms.ollama import OllamaLLM
from AgentMem.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH

load_dotenv()

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            # "model": "qwen3:8b",
            "model": "gpt-oss:20b",
            "temperature": 0.1,
            "max_tokens": 4096,
        }
    },
        "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
    #  "graph_store": {
    #         "provider": "neo4j",# or neo4j-community
    #         "config": {
    #             "url": "bolt://localhost:7687",
    #             "username": "neo4j", # or neo4j
    #             "password": "Neo4j2025",
    #             "database": "neo4j",
    #         }
    #     }
}
config_chat = {
    "llm": {
        "provider": "ollama",
        "config": {
            # "model": "qwen3:8b",
            "model": "gpt-oss:20b",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    }
}
# config = {
#         "llm": {
#         "provider": "vllm",
#         "config": {
#             "model": "Qwen/Qwen2.5-7B-Instruct",
#             "vllm_base_url": "http://localhost:8000/v1",
#             "api_key": "vllm-api-key",
#             "temperature": 0,
#             "max_tokens": 2000,
#         },
#     },
    # "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
        # "graph_store": {
        #     "provider": "neo4j",# or neo4j-community
        #     "config": {
        #         "url": "bolt://localhost:7687",
        #         "username": "neo4j", # or neo4j
        #         "password": "Neo4j2025",
        #         "database": "neo4j",
        #     }
        # }
    # }
# config_chat = {
#     "llm": {
#         "provider": "vllm",
#         "config": {
#             "model": "Qwen/Qwen2.5-7B-Instruct",
#             "vllm_base_url": "http://localhost:8000/v1",
#              "api_key": "vllm-api-key",
#             "temperature": 0.1,
#             "max_tokens": 4000,
#         }
#     }
# }
            
# 自定义指令
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

class MemoryManager:
    def __init__(self, data_path=None, output_path="results.json", top_k=10, filter_memories=False, is_graph=False, batch_size=2):
        self.mem0_client = Memory.from_config(config)
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        # config_chat = BaseLlmConfig(model="qwen3:8b", temperature=0.7, max_tokens=4096, top_p=1.0)
        
        # llm = OllamaLLM(config_chat)
        # self.chat_llm = llm
        self.chat_llm = self.mem0_client.llm
        if data_path:
            self.load_data()

        self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH if self.is_graph else ANSWER_PROMPT

    # ========== 加载数据 ==========
    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    # ========== 添加记忆 ==========
    def add_memory(self, user_id, message, metadata, retries=3, prompt=None):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    user_id=user_id,messages=message, metadata=metadata
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp}, prompt=custom_instructions)

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # 删除旧记忆
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # 开两个线程并行存储
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=1):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]
            for future in futures:
                future.result()

    # ========== 搜索记忆 ==========
    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k,
                        filter_memories=self.filter_memories, enable_graph=True, output_format="v1.1"
                    )
                else:
                    memories = self.mem0_client.search(
                        user_id=user_id, query = query,limit=self.top_k
                    )
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)
        end_time = time.time()
        if not self.is_graph:
            memories = memories["results"]
            semantic_memories = [
                {"memory": m["memory"], "timestamp": m["metadata"]["timestamp"], "score": round(m["score"], 2)}
                for m in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {"memory": m["memory"], "timestamp": m["metadata"]["timestamp"], "score": round(m["score"], 2)}
                for m in memories["results"]
            ]
            graph_memories = [
                {"source": r["source"], "relationship": r["relationship"], "target": r["target"]}
                for r in memories["relations"]
            ]
        return semantic_memories, graph_memories,end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories,speaker_1_memory_time  = self.search_memory(speaker_1_user_id, question)
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time= self.search_memory(speaker_2_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )
        with open("/root/nfs/hmj/proj/mem0/evaluation/src/memMini/1.txt","w") as f:
            f.write(answer_prompt)  

        t1 = time.time()
        response =self.chat_llm.generate_response(messages=[{"role": "system", "content": answer_prompt}])
        t2 = time.time()
        response_time = t2 - t1
        # response = self.mem0_client.generate_response(messages=[{"role": "system", "content": answer_prompt}])
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        
        return {
        "response": response,
        "speaker_1_memories": speaker_1_memories,
        "speaker_2_memories": speaker_2_memories,
        "speaker_1_memory_time": speaker_1_memory_time,
        "speaker_2_memory_time": speaker_2_memory_time,
        "speaker_1_graph_memories": speaker_1_graph_memories,
        "speaker_2_graph_memories": speaker_2_graph_memories,
        "response_time": response_time,
        "question": question,
        "answer": answer,
        "category": category
    }


    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(qa, total=len(qa), desc=f"Questions for conversation {idx}", leave=False):
                result = self.answer_question(speaker_a_user_id, speaker_b_user_id, question_item["question"], question_item["answer"], question_item.get("category", -1))
                self.results[idx].append(result)

                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    manager = MemoryManager(
        data_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1.json",
        output_path="/root/nfs/hmj/proj/mem0/evaluation/src/memMini/results/exp_results.json",
        top_k=30,
        filter_memories=False,
        is_graph=False
    )
    # 如果output文件不存在，mkdir创建它
    os.makedirs(os.path.dirname(manager.output_path), exist_ok=True)
    manager.process_all_conversations()  # 添加记忆
    manager.process_data_file("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1.json")  # 搜索问答
