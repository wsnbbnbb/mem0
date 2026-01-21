import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from jinja2 import Template
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool, create_search_memory_tool
from openai import OpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from prompts import ANSWER_PROMPT
from tqdm import tqdm
# import langgraph
from langchain_openai import ChatOpenAI
import sys
sys.path.append("/root/nfs/hmj/proj/mem0")
from logger import get_logger

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = get_logger(__name__, filename="lgm.log")
# Default client variable (may be set to OpenAI client if available)
client = None
# load_dotenv()
from langchain_ollama import ChatOllama

model_langmem = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
)
# langmem_model = ChatOpenAI(base_url="http://127.0.0.1:8088/v1",api_key="vllm")
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )
# llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen2.5-1.5B-Instruct",
#     task="text-generation",
#     device_map="auto",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
#     model_kwargs={"quantization_config": quantization_config}, 
# )
# langmem_model = ChatHuggingFace(llm=llm)
# client = OpenAI()
# client = OpenAI(base_url="http://127.0.0.1:8008/v1",api_key="vllm")

# # 创建 llm，需要避免使用 'auto' tool_choice，后端不支持 --enable-auto-tool-choice
# # 改为使用 tool_choice='none' 强制使用工具调用或 bind_tools 时指定
# llm = ChatOpenAI(
#     model="Qwen/Qwen2.5-7B-Instruct",
#     base_url="http://localhost:8008/v1",
#     api_key="vllm",
#     temperature=0,
#     model_kwargs={"tool_choice": "none"}  # 设置为 'none' 来避免 'auto' 导致的错误
# )
ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)

def get_answer(question, speaker_1_user_id, speaker_1_memories, speaker_2_user_id, speaker_2_memories):
    prompt = ANSWER_PROMPT_TEMPLATE.render(
        question=question,
        speaker_1_user_id=speaker_1_user_id,
        speaker_1_memories=speaker_1_memories,
        speaker_2_user_id=speaker_2_user_id,
        speaker_2_memories=speaker_2_memories,
    )

    t1 = time.time()
    answer_text = ""
    try:
        # Prefer OpenAI-like client if available
        if "client" in globals() and client is not None:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct", messages=[{"role": "system", "content": prompt}], temperature=0.0
            )
            # OpenAI-compatible response structure
            answer_text = response.choices[0].message.content
        else:
            # Fallback: use local ChatOllama model (or any model assigned to model_langmem)
            if "model_langmem" in globals() and model_langmem is not None:
                try:
                    res = model_langmem.invoke([{"role": "system", "content": prompt}])
                except Exception:
                    try:
                        res = model_langmem.generate([{"role": "system", "content": prompt}])
                    except Exception as e:
                        raise

                # Normalize response content from various return types
                if isinstance(res, dict):
                    answer_text = res.get("content") or res.get("text") or str(res)
                else:
                    # res might be an AI message object or a LangChain Generation
                    answer_text = getattr(res, "content", None) or getattr(res, "text", None) or str(res)
            else:
                raise RuntimeError("No LLM client available: neither 'client' nor 'model_langmem' is defined.")
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer_text = ""
    t2 = time.time()
    return answer_text, t2 - t1


def prompt(state):
    """Prepare the messages for the LLM.
    
    Limits retrieved memories to prevent context overflow:
    - Only retrieves top 3 most relevant memories
    - Truncates each memory to max 500 chars
    - Total memory section capped at ~2000 chars to avoid exceeding token limit
    """
    store = get_store()
    try:
        # Retrieve only the top 1 most relevant memory to aggressively save tokens
        all_memories = store.search(
            ("memories",),
            query=state["messages"][-1].content,
            limit=1,  # Aggressively limit to top 1 result
        )

        # Truncate and filter memories to stay within a much smaller token budget
        memory_items = []
        total_chars = 0
        max_total_chars = 800  # much smaller budget (~200 tokens) for memory section
        max_per_item = 200
        
        for memory in all_memories:
            if isinstance(memory, dict):
                mem_text = str(memory.get("text", "") or memory.get("content", ""))
            else:
                mem_text = str(memory)
            
            # Truncate individual memory
            if len(mem_text) > max_per_item:
                mem_text = mem_text[:max_per_item] + "..."
            
            # Check total budget
            if total_chars + len(mem_text) > max_total_chars:
                break
            
            memory_items.append(mem_text)
            total_chars += len(mem_text)
        
        memories_text = "\n".join(memory_items) if memory_items else "(No relevant memories)"
    except Exception as e:
        logger.warning(f"Error retrieving memories in prompt: {e}")
        memories_text = "(Error retrieving memories)"
    
    system_msg = f"""You are a helpful assistant with memory management capabilities.

When a user message contains information that should be remembered, use the memory management tools provided.
When a user asks a question, search the memory tools to find relevant past information.

## Current Memories
<memories>
{memories_text}
</memories>

IMPORTANT TOOL CALL GUIDELINES:
- Always return valid JSON for tool parameters
- Keep memory updates concise and focused
- If updating memory, provide a clear summary of what changed
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]

from sentence_transformers import SentenceTransformer

# 初始化 embedding 模型 (开源)
# embed_model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda:2")  # 小巧快速，也能处理中英文本

# def embed_texts(texts: list[str]) -> list[list[float]]:
    # 过滤掉空文本
    # clean = [t if isinstance(t, str) and t.strip() != "" else " " for t in texts]
    # vecs = embed_model.encode(clean, show_progress_bar=False, convert_to_numpy=True)
    # return vecs.tolist()

def validate_and_fix_json(json_str):
    """Validate JSON and attempt to fix common issues from LLM output."""
    if not isinstance(json_str, str):
        return json_str
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON from LLM: {json_str[:200]}... Error: {e}")
        # Try common fixes
        # Remove trailing incomplete JSON
        if json_str.count('{') > json_str.count('}'):
            json_str = json_str.rsplit('}', 1)[0] + '}'
        
        try:
            return json.loads(json_str)
        except:
            # Last resort: return empty dict
            logger.error(f"Could not parse JSON: {json_str[:200]}")
            return {"action": "skip", "reason": "invalid_json"}

class LangMem:
    def __init__(
        self,
    ):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda:2")  # 小巧快速，也能处理中英文本
        # self.store = InMemoryStore(
        #     index={
        #         "dims": 1536,
        #         "embed": f"openai:{os.getenv('EMBEDDING_MODEL')}",
        #     }
        # )
        self.store = InMemoryStore( #
            index={
               "dims":self.embed_model.get_sentence_embedding_dimension(),
               "embed": self.embed_texts,
            }
        )
        self.checkpointer = MemorySaver()  # Checkpoint graph state

        self.agent = create_react_agent(
            model=model_langmem,
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
            checkpointer=self.checkpointer,
        )
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = None
        if tiktoken:
            try:
                # Try to get encoding for the model; fallback to cl100k_base if not available
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    def embed_texts(self,texts: list[str]) -> list[list[float]]:
    # 过滤掉空文本
        clean = [t if isinstance(t, str) and t.strip() != "" else " " for t in texts]
        vecs = self.embed_model.encode(clean, show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()

    def _truncate_text_for_model(self, text: str, max_tokens: int = 32768, safety: float = 0.70) -> str:
        """Truncate `text` to stay within `max_tokens` using accurate tokenizer if available.

        Very aggressive: reserves ~9800 tokens for system prompts, tools, chat history, and all overhead.
        This ensures even with system prompt + tool definitions + agent processing, we stay under limit.
        Keeps the last `allowed_tokens` worth of text (most recent context).
        Falls back to heuristic if tiktoken unavailable.
        """
        if not isinstance(text, str):
            return text

        # Reserve aggressively for ALL overhead: system prompt (~500t), tools (~1500t), 
        # agent processing overhead (~2000t), chat history management (~5000t)
        # Total conservative estimate: ~9800 tokens reserved, leaving ~23k for user input
        system_overhead = int(max_tokens * 0.30)  # Reserve 30% of budget for system overhead
        allowed_tokens = int((max_tokens - system_overhead) * safety)

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) <= allowed_tokens:
                    return text
                # Keep the last `allowed_tokens` worth of text (most recent context)
                truncated_tokens = tokens[-allowed_tokens:]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                logger.warning(
                    f"Input truncated from {len(tokens)} to {len(truncated_tokens)} tokens "
                    f"(model limit {max_tokens}, reserved {system_overhead}t overhead, safety {safety})."
                )
                return truncated_text
            except Exception as e:
                logger.warning(f"Tokenizer error: {e}. Falling back to heuristic.")
        
        # Fallback: ~4 chars per token heuristic
        approx_tokens = len(text) / 4
        if approx_tokens <= allowed_tokens:
            return text

        allowed_chars = int(allowed_tokens * 4)
        logger.warning(
            f"Input too long (approx {int(approx_tokens)} tokens). "
            f"Truncating to last ~{allowed_tokens} tokens (~{allowed_chars} chars)."
        )
        return text[-allowed_chars:]

    def add_memory(self, message, config):
        # Ensure message fits model context limits (heuristic truncation)
        try:
            truncated = self._truncate_text_for_model(message)
        except Exception:
            truncated = message
        
        logger.debug(f"Adding memory加入的记忆: {truncated}，长度: {len(self.tokenizer.encode(truncated)) if self.tokenizer else 'unknown'}")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = self.agent.invoke({"messages": [{"role": "user", "content": truncated}]}, config=config)
                return result
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for add_memory: {error_msg[:200]}")
                
                # Check if it's a JSON parsing error from Ollama
                if "error parsing tool call" in error_msg or "invalid character" in error_msg:
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying add_memory after JSON parsing error...")
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    else:
                        # Last attempt failed, return graceful fallback
                        logger.error(f"add_memory failed after {max_retries} attempts: {error_msg[:200]}")
                        return {"messages": [{"role": "assistant", "content": "Memory update skipped due to format error"}]}
                else:
                    # Non-JSON error, raise immediately
                    raise

    def search_memory(self, query, config):
        t1 = time.time()
        try:
            try:
                truncated_q = self._truncate_text_for_model(query)
            except Exception:
                truncated_q = query
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.agent.invoke({"messages": [{"role": "user", "content": truncated_q}]}, config=config)
                    t2 = time.time()
                    logger.info(f"response:{response}")
                    return response["messages"][-1].content, t2 - t1
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Attempt {attempt + 1} failed for search_memory: {error_msg[:200]}")
                    
                    # Check if it's a JSON parsing error from Ollama
                    if "error parsing tool call" in error_msg or "invalid character" in error_msg:
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying search_memory after JSON parsing error...")
                            time.sleep(0.5)
                            continue
                        else:
                            # Return empty result instead of failing completely
                            logger.error(f"search_memory failed after {max_retries} attempts")
                            t2 = time.time()
                            return "", t2 - t1
                    else:
                        raise
        except Exception as e:
            t2 = time.time()
            logger.error(f"Error in search_memory: {e}")
            return "", t2 - t1


class LangMemManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)
    @staticmethod
    def process_conversation(key_value_pair , data):
            key, value = key_value_pair
            result = defaultdict(list)

            chat_history = value["conversation"]
            questions = value["question"]

            agent1 = LangMem()
            agent2 = LangMem()
            config = {"configurable": {"thread_id": f"thread-{key}"}}
            speakers = set()

            # Identify speakers
            for conv in chat_history:
                speakers.add(conv["speaker"])

            if len(speakers) != 2:
                raise ValueError(f"Expected 2 speakers, got {len(speakers)}")

            speaker1 = list(speakers)[0]
            speaker2 = list(speakers)[1]

            # Add memories for each message
            for conv in tqdm(chat_history, desc=f"Processing messages {key}", leave=False):
                message = f"{conv['timestamp']} | {conv['speaker']}: {conv['text']}"
                try:
                    if conv["speaker"] == speaker1:
                        agent1.add_memory(message, config)
                    elif conv["speaker"] == speaker2:
                        agent2.add_memory(message, config)
                    else:
                        raise ValueError(f"Expected speaker1 or speaker2, got {conv['speaker']}")
                except Exception as e:
                    # Log the error but continue processing
                    logger.error(f"Error adding memory for speaker {conv['speaker']}: {str(e)[:200]}")
                    continue

            # Process questions
            for q in tqdm(questions, desc=f"Processing questions {key}", leave=False):
                try:
                    category = q["category"]

                    if int(category) == 5:
                        continue

                    answer = q["answer"]
                    question = q["question"]
                    response1, speaker1_memory_time = agent1.search_memory(question, config)
                    response2, speaker2_memory_time = agent2.search_memory(question, config)

                    generated_answer, response_time = get_answer(question, speaker1, response1, speaker2, response2)

                    result[key].append(
                        {
                            "question": question,
                            "answer": answer,
                            "response1": response1,
                            "response2": response2,
                            "category": category,
                            "speaker1_memory_time": speaker1_memory_time,
                            "speaker2_memory_time": speaker2_memory_time,
                            "response_time": response_time,
                            "response": generated_answer,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)[:200]}")
                    continue

            return result

    # def process_all_conversations(self, output_file_path):
    #     OUTPUT = defaultdict(list)

    #     # Process conversations in parallel with multiple workers
       

    #     # 准备参数列表
    #     tasks = [(kv, self.data) for kv in self.data.items()]

    #     with mp.Pool(processes=10) as pool:
    #         results = list(
    #             tqdm(
    #                 pool.starmap(LangMemManager.process_conversation, tasks),
    #                 total=len(tasks),
    #                 desc="Processing conversations",
    #             )
    #         )
    #     # Combine results from all workers
    #     for result in results:
    #         for key, items in result.items():
    #             OUTPUT[key].extend(items)

    #     # Save final results
    #     with open(output_file_path, "w") as f:
    #         json.dump(OUTPUT, f, indent=4)
            
    def process_all_conversations(self, output_file_path):
        OUTPUT = defaultdict(list)

        for key, value in tqdm(self.data.items(), desc="Processing conversations serially"):
            result = LangMemManager.process_conversation((key, value), self.data)
            for k, items in result.items():
                OUTPUT[k].extend(items)

        with open(output_file_path, "w") as f:
            json.dump(OUTPUT, f, indent=4)
        
def main():
    output_file_path = os.path.join('/root/nfs/hmj/proj/mem0/evaluation/src/other/results', "langmem_results.json")
    langmem_manager = LangMemManager(dataset_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1_rag.json")
    langmem_manager.process_all_conversations(output_file_path)
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()