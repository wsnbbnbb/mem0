"""
Alternative implementation of LangMem that avoids problematic tool calls.
This version uses simpler prompts and manual memory management instead of
relying on LangChain's ReAct agent which has issues with Ollama's JSON parsing.
"""

import json
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from tqdm import tqdm
import sys
sys.path.append("/root/nfs/hmj/proj/mem0")
from logger import get_logger

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = get_logger(__name__, filename="lgm_alternative.log")
load_dotenv()

from jinja2 import Template
from prompts import ANSWER_PROMPT

ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)

# Initialize LLM
model_langmem = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
)


def get_answer(question, speaker_1_user_id, speaker_1_memories, speaker_2_user_id, speaker_2_memories):
    """Generate an answer based on question and retrieved memories."""
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
        if model_langmem is not None:
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
                answer_text = getattr(res, "content", None) or getattr(res, "text", None) or str(res)
        else:
            raise RuntimeError("LLM model not available")
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer_text = ""
    
    t2 = time.time()
    return answer_text, t2 - t1


class SimpleLangMem:
    """Simplified memory manager that avoids problematic ReAct tool calls."""
    
    def __init__(self):
        self.memories = []  # Simple in-memory storage
        self.tokenizer = None
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    
    def _truncate_text_for_model(self, text: str, max_tokens: int = 32768, safety: float = 0.70) -> str:
        """Truncate text to stay within token limits."""
        if not isinstance(text, str):
            return text

        system_overhead = int(max_tokens * 0.30)
        allowed_tokens = int((max_tokens - system_overhead) * safety)

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) <= allowed_tokens:
                    return text
                truncated_tokens = tokens[-allowed_tokens:]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                logger.warning(
                    f"Input truncated from {len(tokens)} to {len(truncated_tokens)} tokens "
                    f"(model limit {max_tokens}, reserved {system_overhead}t overhead, safety {safety})."
                )
                return truncated_text
            except Exception as e:
                logger.warning(f"Tokenizer error: {e}. Falling back to heuristic.")
        
        approx_tokens = len(text) / 4
        if approx_tokens <= allowed_tokens:
            return text

        allowed_chars = int(allowed_tokens * 4)
        logger.warning(
            f"Input too long (approx {int(approx_tokens)} tokens). "
            f"Truncating to last ~{allowed_tokens} tokens (~{allowed_chars} chars)."
        )
        return text[-allowed_chars:]
    
    def add_memory(self, message):
        """Add a memory without using ReAct agent."""
        try:
            truncated = self._truncate_text_for_model(message)
        except Exception:
            truncated = message
        
        logger.debug(f"Adding memory: {truncated[:100]}... (length: {len(truncated)})")
        
        # Simple memory addition
        self.memories.append({"timestamp": time.time(), "content": truncated})
        return {"status": "success", "memory_count": len(self.memories)}
    
    def search_memory(self, query):
        """Search memories using simple string matching."""
        t1 = time.time()
        try:
            try:
                truncated_q = self._truncate_text_for_model(query)
            except Exception:
                truncated_q = query
            
            # Simple search: find memories that contain query words
            query_words = set(truncated_q.lower().split())
            
            matches = []
            for mem in self.memories:
                mem_content = mem["content"].lower()
                matching_words = len(query_words.intersection(set(mem_content.split())))
                if matching_words > 0:
                    matches.append((mem, matching_words))
            
            # Sort by relevance and return top results
            matches.sort(key=lambda x: x[1], reverse=True)
            
            if matches:
                # Format retrieved memories
                retrieved = "\n".join([m[0]["content"][:200] for m in matches[:3]])
            else:
                retrieved = "(No relevant memories found)"
            
            t2 = time.time()
            return retrieved, t2 - t1
        except Exception as e:
            t2 = time.time()
            logger.error(f"Error in search_memory: {e}")
            return "", t2 - t1


class SimpleLangMemManager:
    """Manager for processing conversations without ReAct agents."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)
    
    @staticmethod
    def process_conversation(key_value_pair, data):
        """Process a conversation with simplified memory management."""
        key, value = key_value_pair
        result = defaultdict(list)

        chat_history = value["conversation"]
        questions = value["question"]

        agent1 = SimpleLangMem()
        agent2 = SimpleLangMem()

        speakers = set()
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
                    agent1.add_memory(message)
                elif conv["speaker"] == speaker2:
                    agent2.add_memory(message)
                else:
                    raise ValueError(f"Expected speaker1 or speaker2, got {conv['speaker']}")
            except Exception as e:
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
                response1, speaker1_memory_time = agent1.search_memory(question)
                response2, speaker2_memory_time = agent2.search_memory(question)

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

    def process_all_conversations(self, output_file_path):
        """Process all conversations and save results."""
        OUTPUT = defaultdict(list)

        for key, value in tqdm(self.data.items(), desc="Processing conversations serially"):
            try:
                result = SimpleLangMemManager.process_conversation((key, value), self.data)
                for k, items in result.items():
                    OUTPUT[k].extend(items)
            except Exception as e:
                logger.error(f"Error processing conversation {key}: {str(e)[:200]}")
                continue

        with open(output_file_path, "w") as f:
            json.dump(OUTPUT, f, indent=4)


def main():
    output_file_path = os.path.join('./result_lgm/', "langmem_results_alternative.json")
    os.makedirs('./result_lgm/', exist_ok=True)
    
    langmem_manager = SimpleLangMemManager(
        dataset_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1_rag.json"
    )
    langmem_manager.process_all_conversations(output_file_path)
    logger.info(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
