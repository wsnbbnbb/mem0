import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("/root/nfs/hmj/proj/mem0/evaluation")
from dotenv import load_dotenv
from tqdm import tqdm
from jinja2 import Template
from AgentMem.memory.AMemo import Memory

load_dotenv()

# AMemo 配置
config = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
    "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "amemo_eval", "host": "localhost", "port": 6333},
    },
}

# 聊天 LLM 配置（用于生成答案）
config_chat = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    }
}

# 自定义指令（可选，AMemo 不使用）
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

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """



class MemoryManager:
    def __init__(self, data_path=None, output_path="results_amemo.json", top_k=10, filter_memories=False, batch_size=2):
        self.amemo_client = Memory.from_config(config)
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        
        # 聊天 LLM（用于生成答案）
        self.chat_llm = self.amemo_client.llm
        
        if data_path:
            self.load_data()

    # ========== 加载数据 ==========
    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    # ========== 添加记忆 ==========
    def add_memory(self, user_id, messages, metadata):
        """
        添加记忆到 AMemo
        
        Args:
            user_id: 用户ID
            messages: 消息列表 [{"role": "user", "content": "...", ...}]
            metadata: 元数据字典
        """
        for attempt in range(3):
            try:
                self.amemo_client.add(messages=messages, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # 删除旧记忆
        self.amemo_client.delete_all(user_id=speaker_a_user_id)
        self.amemo_client.delete_all(user_id=speaker_b_user_id)

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
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}", "name": speaker_a})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}", "name": speaker_a})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}", "name": speaker_b})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}", "name": speaker_b})
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
        
        # 限制为第一个样本用于测试
        n_samples = min(1, len(self.data))
        
        for idx in tqdm(range(n_samples), desc="Processing conversations"):
            item = self.data[idx]
            self.process_conversation(item, idx)

    # ========== 搜索记忆 ==========
    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                memories = self.amemo_client.search(
                    user_id=user_id,
                    query=query,
                    limit=self.top_k
                )
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)
        end_time = time.time()

        # 格式化 AMemo 的返回结果，保留所有元数据
        formatted_memories = []
        for m in memories:
            # 保留所有字段，同时添加兼容的字段
            memory_item = m.copy()
            # 确保 text 字段存在
            if "text" in memory_item:
                memory_item["memory"] = memory_item["text"]
            # 确保 timestamp 存在（从 metadata 中获取）
            if "timestamp" not in memory_item:
                memory_item["timestamp"] = str(memory_item.get("timestamp", ""))
            # 确保 score 存在
            if "score" not in memory_item:
                memory_item["score"] = memory_item.get("score", 0)
            formatted_memories.append(memory_item)

        return formatted_memories, None, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category=None, is_adversarial=False, adversarial_answer=None):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(speaker_1_user_id, question)
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(speaker_2_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        # 调试：打印原始记忆信息
        print(f"\n=== DEBUG: Question: {question} ===")
        print(f"Speaker 1 memories count: {len(speaker_1_memories)}")
        print(f"Speaker 2 memories count: {len(speaker_2_memories)}")
        if speaker_1_memories:
            print(f"First memory from speaker 1:")
            print(f"  timestamp: {speaker_1_memories[0].get('timestamp')}")
            print(f"  text: {speaker_1_memories[0].get('text')[:100]}")
        print(f"Formatted search_1_memory: {search_1_memory[:1] if search_1_memory else 'empty'}")
        print(f"Formatted search_2_memory: {search_2_memory[:1] if search_2_memory else 'empty'}")
        print("=============================================\n")

        template = Template(ANSWER_PROMPT)
        
        # 准备渲染参数
        render_params = {
            "speaker_1_user_id": speaker_1_user_id.split("_")[0],
            "speaker_2_user_id": speaker_2_user_id.split("_")[0],
            "speaker_1_memories": json.dumps(search_1_memory, indent=4),
            "speaker_2_memories": json.dumps(search_2_memory, indent=4),
            "question": question,
        }
        
        # 调试：检查参数
        print(f"DEBUG: Rendering template with params:")
        print(f"  speaker_1_user_id: {render_params['speaker_1_user_id']}")
        print(f"  speaker_2_user_id: {render_params['speaker_2_user_id']}")
        print(f"  question: {question[:50]}...")
        print(f"  speaker_1_memories length: {len(render_params['speaker_1_memories'])}")
        print(f"  speaker_2_memories length: {len(render_params['speaker_2_memories'])}")
        
        # 渲染模板
        answer_prompt = template.render(**render_params)
        
        # 调试：检查渲染结果
        print(f"DEBUG: Rendered answer_prompt length: {len(answer_prompt)}")
        print(f"DEBUG: First 200 chars of answer_prompt:")
        print(answer_prompt[:200])
        
        # 保存 prompt 用于调试
        with open("/root/nfs/hmj/proj/mem0/evaluation/src/Amemo/1.txt", "w") as f:
            f.write(answer_prompt)

        t1 = time.time()
        response = self.chat_llm.generate_response(messages=[{"role": "user", "content": answer_prompt}])
        t2 = time.time()
        response_time = t2 - t1
        
        # 清理响应
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
            "answer": answer if not is_adversarial else None,
            "adversarial_answer": adversarial_answer if is_adversarial else None,
            "category": category,
            "is_adversarial": is_adversarial
        }

    def process_data_file(self, file_path):
        # 限制为第一个样本用于测试
        n_samples = min(1, len(self.data))
        
        for idx in tqdm(range(n_samples), total=n_samples, desc="Processing Q&A"):
            item = self.data[idx]
            qa = item["qa"]
            
            speaker_a_user_id = f"{item['conversation']['speaker_a']}_{idx}"
            speaker_b_user_id = f"{item['conversation']['speaker_b']}_{idx}"

            for question_item in qa:
                category = question_item.get("category", -1)
                is_adversarial = (category == 5)
                
                # 对抗性问题使用 adversarial_answer，普通问题使用 answer
                if is_adversarial:
                    answer = question_item.get("adversarial_answer")
                    adversarial_answer = answer
                else:
                    answer = question_item.get("answer")
                    adversarial_answer = None
                
                result = self.answer_question(
                    speaker_a_user_id, 
                    speaker_b_user_id, 
                    question_item["question"], 
                    answer,
                    category=category,
                    is_adversarial=is_adversarial,
                    adversarial_answer=adversarial_answer
                )
                self.results[idx].append(result)

                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    manager = MemoryManager(
        data_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1.json",
        output_path="/root/nfs/hmj/proj/mem0/evaluation/src/Amemo/results/amemo_results.json",
        top_k=10,
        batch_size=2
    )
    
    # 如果 output 文件不存在，mkdir 创建它
    os.makedirs(os.path.dirname(manager.output_path) or ".", exist_ok=True)
    
    manager.process_all_conversations()  # 添加记忆
    manager.process_data_file("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1.json")  # 搜索问答
