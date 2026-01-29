import json
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

# 添加路径 (请确保路径正确)
sys.path.append("/root/nfs/hmj/proj/mem0/evaluation")

from AgentMem.configs.llms.base import BaseLlmConfig
from AgentMem.llms.openai_structured import OpenAIStructuredLLM

load_dotenv()

# LLM 配置
config_llm = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "arcee-ai/trinity-large-preview:free",
            "openai_base_url": "https://openrouter.ai/api/v1",
            "api_key": "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",
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
}
config_data = config_llm["llm"]["config"]

# ==================== 针对时间换算优化的提示词 ====================
SUFFICIENCY_CHECK_PROMPT = """You are a professional evaluator for a RAG system. 

## Task
Determine if the **Retrieved Memories** contain sufficient information to deduce the **Expected Answer**.

## Input Data
- **Question**: {question}
- **Expected Answer**: {expected_answer}
- **Memories (Caroline)**:
{speaker_1_memories}
- **Memories (Melanie)**:
{speaker_2_memories}

## CRITICAL RULE: Temporal Reasoning
- Many memories use relative time (e.g., "yesterday", "today"). 
- You **MUST** use the `Timestamp` provided in each memory to calculate the absolute date.
- Example: If a memory from "May 8, 2023" says "I went yesterday", the calculated date is "May 7, 2023". If the Expected Answer is "7 May 2023", this is **SUFFICIENT**.

## Output Requirement
Output ONLY a valid JSON object:
{{
    "is_sufficient": true/false,
    "reasoning": "If sufficient, explain how the answer is derived (including time calculations). If insufficient, explain exactly what fact is missing."
}}

## Strict Guidelines
1. Combine info from all speakers.
2. If the answer can be logically calculated using the Timestamp and the text, mark as `true`.
3. Do not include any other fields in JSON.
"""

def format_memories(memories: List[Dict]) -> str:
    """格式化记忆及其时间戳"""
    if not memories:
        return "[No memories retrieved]"
    
    formatted = []
    for i, mem in enumerate(memories, 1):
        text = mem.get('text', mem.get('memory', 'N/A'))
        timestamp = mem.get('timestamp', 'N/A')
        formatted.append(f"[Memory {i}]")
        formatted.append(f"  Time: {timestamp}")
        formatted.append(f"  Content: {text}")
        formatted.append("")
    
    return "\n".join(formatted)


class MemorySufficiencyEvaluator:
    """评估检索记忆是否足够回答问题"""
    
    def __init__(self, results_file: str, output_dir: str = "evaluation_results"):
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = defaultdict(list)
        self.stats = defaultdict(int)
        
        config_obj = BaseLlmConfig(**config_data)
        self.llm = OpenAIStructuredLLM(config=config_obj)
        
        os.makedirs(output_dir, exist_ok=True)
        self.load_results()
    
    def load_results(self):
        if not os.path.exists(self.results_file):
            print(f"❌ 错误: 找不到文件 {self.results_file}")
            sys.exit(1)
        with open(self.results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, value in data.items():
                self.results[key] = value
        print(f"✓ 加载了 {len(self.results)} 个样本")
    
    def check_memory_sufficiency(self, question: str, expected_answer: str, memories_1: List[Dict], memories_2: List[Dict]) -> Dict[str, Any]:
        mem_1_str = format_memories(memories_1)
        mem_2_str = format_memories(memories_2)
        
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            speaker_1_memories=mem_1_str,
            speaker_2_memories=mem_2_str
        )
        
        try:
            response = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
            response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"is_sufficient": False, "reasoning": "JSON parse error"}
        except Exception as e:
            return {"is_sufficient": False, "reasoning": f"LLM error: {str(e)}"}
    
    def evaluate_all(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        all_details = []
        sample_keys = list(self.results.keys())[:max_samples] if max_samples else list(self.results.keys())
        
        print(f"\n开始评估...")
        for sample_idx in tqdm(sample_keys, desc="Samples"):
            for qa_idx, item in enumerate(self.results[sample_idx]):
                if item.get("is_adversarial", False):
                    continue

                res = self.check_memory_sufficiency(
                    question=item.get("question", ""),
                    expected_answer=item.get("answer"),
                    memories_1=item.get("speaker_1_memories", []),
                    memories_2=item.get("speaker_2_memories", [])
                )
                
                is_sufficient = res.get("is_sufficient", False)
                self.stats["sufficient" if is_sufficient else "insufficient"] += 1
                
                all_details.append({
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "question": item.get("question"),
                    "expected_answer": item.get("answer"),
                    "is_sufficient": is_sufficient,
                    "reasoning": res.get("reasoning", "")
                })
        
        # 合并输出为一个文件
        output_file = os.path.join(self.output_dir, "sufficiency_evaluation_final.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_details, f, indent=2, ensure_ascii=False)
        
        return all_details

    def print_report(self, total: int):
        s = self.stats["sufficient"]
        i = self.stats["insufficient"]
        print("\n" + "=" * 50)
        print(f"评估完成！总计: {total}")
        print(f"✅ 充分: {s} | ❌ 不足: {i}")
        print(f"召回率: {round(s/total*100, 2) if total>0 else 0}%")
        print("=" * 50)


def main():
    results_file = "/root/nfs/hmj/proj/mem0/evaluation/src/Amemo/results/amemo_results.json"
    output_dir = "/root/nfs/hmj/proj/mem0/evaluation/src/Amemo/results"
    
    evaluator = MemorySufficiencyEvaluator(results_file, output_dir)
    details = evaluator.evaluate_all()
    evaluator.print_report(len(details))

if __name__ == "__main__":
    main()