import importlib
from typing import Optional

from AgentMem.configs.embeddings.base import BaseEmbedderConfig
from AgentMem.configs.llms.base import BaseLlmConfig
from AgentMem.embeddings.mock import MockEmbeddings


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "AgentMem.llms.ollama.OllamaLLM",
        "openai": "AgentMem.llms.openai.OpenAILLM",
        "groq": "AgentMem.llms.groq.GroqLLM",
        "together": "AgentMem.llms.together.TogetherLLM",
        "aws_bedrock": "AgentMem.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "AgentMem.llms.litellm.LiteLLM",
        "azure_openai": "AgentMem.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "AgentMem.llms.openai_structured.OpenAIStructuredLLM",
        "anthropic": "AgentMem.llms.anthropic.AnthropicLLM",
        "azure_openai_structured": "AgentMem.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
        "gemini": "AgentMem.llms.gemini.GeminiLLM",
        "deepseek": "AgentMem.llms.deepseek.DeepSeekLLM",
        "xai": "AgentMem.llms.xai.XAILLM",
        "sarvam": "AgentMem.llms.sarvam.SarvamLLM",
        "lmstudio": "AgentMem.llms.lmstudio.LMStudioLLM",
        "vllm": "AgentMem.llms.vllm.VllmLLM",
        "langchain": "AgentMem.llms.langchain.LangchainLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "AgentMem.embeddings.openai.OpenAIEmbedding",
        "ollama": "AgentMem.embeddings.ollama.OllamaEmbedding",
        "huggingface": "AgentMem.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "AgentMem.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "AgentMem.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "AgentMem.embeddings.vertexai.VertexAIEmbedding",
        "together": "AgentMem.embeddings.together.TogetherEmbedding",
        "lmstudio": "AgentMem.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "AgentMem.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "AgentMem.embeddings.aws_bedrock.AWSBedrockEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "AgentMem.vector_stores.qdrant.Qdrant",
        "chroma": "AgentMem.vector_stores.chroma.ChromaDB",
        "pgvector": "AgentMem.vector_stores.pgvector.PGVector",
        "milvus": "AgentMem.vector_stores.milvus.MilvusDB",
        "upstash_vector": "AgentMem.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "AgentMem.vector_stores.azure_ai_search.AzureAISearch",
        "pinecone": "AgentMem.vector_stores.pinecone.PineconeDB",
        "mongodb": "AgentMem.vector_stores.mongodb.MongoDB",
        "redis": "AgentMem.vector_stores.redis.RedisDB",
        "elasticsearch": "AgentMem.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "AgentMem.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "AgentMem.vector_stores.opensearch.OpenSearchDB",
        "supabase": "AgentMem.vector_stores.supabase.Supabase",
        "weaviate": "AgentMem.vector_stores.weaviate.Weaviate",
        "faiss": "AgentMem.vector_stores.faiss.FAISS",
        "langchain": "AgentMem.vector_stores.langchain.Langchain",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance
