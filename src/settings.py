import os
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from dotenv import load_dotenv

load_dotenv()

SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY")
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME")
AOAI_ENDPOINT=os.environ.get("AOAI_ENDPOINT")
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") 
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") 
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME")
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME")

Settings.llm = AzureOpenAI(
    model="4o-mini",
    deployment_name=AOAI_CHAT_MODEL_NAME,
    api_key=AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=AOAI_EMBEDDING_MODEL_NAME,
    api_key=AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)