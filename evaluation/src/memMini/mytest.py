import sys
sys.path.append("/root/nfs/hmj/proj/mem0")
from dotenv import load_dotenv
from tqdm import tqdm
from jinja2 import Template
from mem0 import Memory
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.ollama import OllamaLLM
answer_prompt="""
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

    Memories for user Dave:

    [
    "9:19 am on 2 September, 2023: Came back from San Francisco with great insights and knowledge on car modification",
    "9:19 am on 2 September, 2023: Want to share car modification knowledge with Calvin",
    "9:19 am on 2 September, 2023: Tied up with car stuff lately"
]

    Memories for user Calvin:

    [
    "9:19 am on 2 September, 2023: Came back from San Francisco with insights on car modification",
    "9:19 am on 2 September, 2023: Giving an old car a new life is satisfying",
    "9:19 am on 2 September, 2023: Tied up with car stuff lately"
]

    Question: What was Dave doing in San Francisco?

    Answer:
    """
import time
import re
from mem0.utils.factory import LlmFactory
def main():
    
    # 1️⃣ 定义要测试的模型列表
    models = [
        "qwen2.5:7b",
        "qwen2.5:3b",
        "qwen3:4b",
        "qwen3:8b",
        "gpt-oss:20b"
    ]

    # 2️⃣ 输入提示
    # answer_prompt = "请在这里写你的系统提示或问题"  # 替换成实际测试内容

    # 3️⃣ 循环测试每个模型
    for model_name in models:
        print("="*50)
        print(f"Testing model: {model_name}")

        # 配置模型
        config_chat = BaseLlmConfig(
            model=model_name,
            temperature=0,
            max_tokens=8192,
            top_p=1.0
        )
        
        # 初始化 LLM
        llm = OllamaLLM(config_chat)
        
        # 开始计时
        start_time = time.time()
        
        # 生成回答
        response = llm.generate_response(messages=[{"role": "system", "content": answer_prompt}])
        
        # 去掉 <think> 标签内容
        response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        
        # 结束计时
        end_time = time.time()
        
        # 输出结果
        print("Response:\n", response_clean.strip())
        print("Time taken:", round(end_time - start_time, 2), "seconds")
def LLM():
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

    chat_llm =Memory.from_config(config_chat)
    print(chat_llm)
if __name__ == "__main__":
    # main()    
    LLM()

# 4b -> 190s --4096
# 8b -> 397s --4096
# 8b(8192) ->
# qwen3-8b -- 11.4s
# qwen3-4b -- 38.4 s
# qwen2.5-7b -- 4.4 s
# qwen2.5-3b --  24s
# gpt-oss:20b - 12.7s