import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("/root/nfs/hmj/proj/mem0")
from dotenv import load_dotenv
from tqdm import tqdm
from mem0 import Memory
from mem0 import MemoryClient

# from logger import logger

load_dotenv()

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:7b",
            "temperature": 0.7,
            "max_tokens": 2000,
        }
    },
    # "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
}

# Configuration for vLLM integration
config = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0.7,
            "max_tokens": 100,
        },
    },
    "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    },
}

# Update custom instructions
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


class MemoryADD_MINI:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        self.mem0_client = Memory.from_config(config)

        # self.mem0_client.update_project(custom_instructions=custom_instructions)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
              # For testing, limit to first 2 conversations
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3, prompt=None):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    message, user_id=user_id, metadata=metadata,prompt=prompt
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp},prompt=custom_instructions)

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
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
                
            # logger.info(f"加载对话，内容为: {messages}")
            # logger.info(f"加载对话，颠倒内容为: {messages_reverse}")
            # logger.info(f"Timestamp: {timestamp}, Speaker A ID: {speaker_a_user_id}, Speaker B ID: {speaker_b_user_id}")
            
            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
                # args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
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

if __name__ == "__main__":
    # logger.info("Starting Memory ADD MINI Process-----------------------------Mytest")
    memory_manager = MemoryADD_MINI(data_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1.json", is_graph=False)
    memory_manager.process_all_conversations()