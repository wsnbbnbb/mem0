# 需要的额外依赖
# pip install sentence-transformers transformers

import json
import os
import time
from collections import defaultdict
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

sys.path.append("/root/nfs/hmj/proj/mem0")
from logger import get_logger

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""

client = OpenAI(base_url="http://127.0.0.1:8008/v1", api_key="vllm")
flogger = get_logger(__name__, filename="rag.log")


class RAGManager:
    def __init__(self,
                 data_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1_rag.json",
                 chunk_size=500,
                 k=1,
                 embedding_model_name="all-MiniLM-L6-v2",
                 device: str = None):
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.client = client
        self.data_path = data_path
        self.chunk_size = chunk_size  # 以 tokenizer token 为单位（见 create_chunks）
        self.k = k

        # Device for sentence-transformers: 'cuda' or 'cpu' or None (auto)
        if device is None:
            # auto choose: use cuda if available
            try:
                import torch
                device = "cuda:2" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        # sentence-transformers embedding model (返回 numpy 向量)
        # 这里用小写模型 id，也可以用 "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)

        # Tokenizer for chunking — 使用 transformers tokenizer 对文本做 token 切分/解码
        # 使用与 embedding 模型一致的 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast=True)

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                                       "questions based on the provided context. "
                                       "If the question involves timing, use the conversation date for reference. "
                                       "Provide the shortest possible answer. "
                                       "Use words directly from the conversation when possible. "
                                       "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception as e:
                retries += 1
                flogger.exception("generate_response error, retry %s/%s", retries, max_retries)
                if retries > max_retries:
                    raise e
                time.sleep(1)

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"
        return cleaned_chat_history

    def calculate_embedding(self, document: str) -> np.ndarray:
        """
        Use SentenceTransformer to get embedding (returns numpy array).
        """
        # sentence-transformers accepts single str or list[str]
        emb = self.embedding_model.encode(document, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # cosine similarity
        if embedding1 is None or embedding2 is None:
            return -1.0
        # protect against zero vectors
        denom = (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if denom == 0:
            return -1.0
        return float(np.dot(embedding1, embedding2) / denom)

    def search(self, query: str, chunks: list[str], embeddings: list[np.ndarray], k: int = 1):
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, emb) for emb in embeddings]

        if k == 1:
            top_indices = [int(np.argmax(similarities))]
        else:
            top_indices = list(np.argsort(similarities)[-k:][::-1])

        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size: int = None):
        """
        Create chunks by token count using self.tokenizer.
        chunk_size is token count per chunk (default uses self.chunk_size)
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            # no chunking
            embeddings = [self.calculate_embedding(documents)]
            return [documents], embeddings

        # tokenize (ids)
        token_ids = self.tokenizer.encode(documents, add_special_tokens=False)
        chunks = []
        for i in range(0, len(token_ids), chunk_size):
            chunk_ids = token_ids[i: i + chunk_size]
            # decode back to text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)

        # compute embeddings (can batch for speed)
        embeddings = []
        # batch encode to speed up
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embs = self.embedding_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            for emb in embs:
                embeddings.append(emb.astype(np.float32))

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
                response, response_time = self.generate_response(question, context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "search_time": search_time,
                        "response_time": response_time,
                    }
                )
                # flush to disk periodically (减少丢失)
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)


def main():
    output_folder = './results/'
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, f"rag_results_1000_k1.json")
    rag_manager = RAGManager(data_path="/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo1_rag.json",
                             chunk_size=1000,
                             k=1,
                             embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                             device=None)
    rag_manager.process_all_conversations(output_file_path)


if __name__ == "__main__":
    main()
