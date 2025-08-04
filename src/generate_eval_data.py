import os
import sys
import csv
import settings
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.core.credentials import AzureKeyCredential

# ====== LlamaIndex 設定 ======

# System Prompt
system_message_chat_conversation = """
あなたはユーザーの質問に回答するチャットボットです。
以下のコンテキストを参考に、簡潔かつ正確に回答してください。
Sourcesに記載がない場合は「すみません、わかりません」と答えてください。
回答には情報源を表示しないでください。

コンテキスト:
{context_str}

質問:
{query_str}
"""

Settings.llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=settings.AOAI_CHAT_MODEL_NAME,
    api_key=settings.AOAI_API_KEY,
    azure_endpoint=settings.AOAI_ENDPOINT,
    api_version=settings.AOAI_API_VERSION,
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=settings.AOAI_EMBEDDING_MODEL_NAME,
    api_key=settings.AOAI_API_KEY,
    azure_endpoint=settings.AOAI_ENDPOINT,
    api_version=settings.AOAI_API_VERSION,
)

# Azure Search 接続
search_client = SearchClient(
    endpoint=settings.SEARCH_SERVICE_ENDPOINT,
    index_name=settings.SEARCH_SERVICE_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_SERVICE_API_KEY)
)
async_search_client = AsyncSearchClient(
    endpoint=settings.SEARCH_SERVICE_ENDPOINT,
    index_name=settings.SEARCH_SERVICE_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_SERVICE_API_KEY),
)

# VectorStore 設定
vector_store = AzureAISearchVectorStore(
    search_or_index_client=search_client,
    async_search_or_index_client=async_search_client,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="contentVector",
    metadata_string_field_key="metadata_json",
    doc_id_field_key="doc_id"
)

# Index + QueryEngine 準備
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
qa_template = PromptTemplate(system_message_chat_conversation, prompt_type="text_qa")
query_engine = index.as_query_engine(similarity_top_k=3, text_qa_template=qa_template)

# ====== 評価データ生成関数 ======

def search(question: str):
    """LlamaIndex QueryEngineを使って質問に回答"""
    response = query_engine.query(question)
    answer_text = response.response if hasattr(response, "response") else str(response)
    
    # コンテキスト（Sources）をまとめる
    sources = [node.node.get_content() for node in getattr(response, "source_nodes", [])]
    context_text = " ".join(sources)
    
    return answer_text, context_text


def load_questions(filepath):
    """CSVから質問と正解データをロード"""
    questions = []
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append((row["question"], row["ground_truth"]))
    return questions


def generate_evaluation_dataset(questions, file_name):
    """質問と回答を評価用CSVに出力"""
    with open(file_name, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["query", "response", "context", "ground_truth"])
        
        for question, ground_truth in questions:
            response, context = search(question)
            writer.writerow([
                question, 
                response.replace("\n", " "), 
                context.replace("\n", " "), 
                ground_truth
            ])

# ====== 実行部分 ======

if __name__ == "__main__":
    """
    使用方法:
    python generate_eval_data.py input.csv output.csv
    input.csv: question, ground_truthが列に含まれる
    output.csv: query, response, context, ground_truthが出力される
    """
    csv_file_path = sys.argv[1]
    output_file = sys.argv[2]
    
    questions = load_questions(csv_file_path)
    generate_evaluation_dataset(questions, output_file)