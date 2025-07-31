import os
import settings 
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.core.credentials import AzureKeyCredential
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore


# AIのキャラクターを決めるためのシステムメッセージを定義する。
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
    api_version=settings.AOAI_API_VERSION
)

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

vector_store = AzureAISearchVectorStore(
    search_or_index_client=search_client,
    async_search_or_index_client=async_search_client,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="contentVector",
    metadata_string_field_key="metadata_json",
    doc_id_field_key="doc_id"
)

# Query Engine 作成（System Prompt組み込み）
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

qa_template = PromptTemplate(
    system_message_chat_conversation,
    prompt_type="text_qa"
)

query_engine = index.as_query_engine(similarity_top_k=10, text_qa_template=qa_template)


# streamlit

# チャット履歴を初期化する
if "history" not in st.session_state:
    st.session_state["history"] = []
    
# チャット履歴を表示する
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# ユーザーが質問を入力した時の処理
if prompt := st.chat_input("質問を入力してください"):
    # ユーザーが入力した質問を表示する
    with st.chat_message("user"):
        st.write(prompt)
        
    # ユーザーの質問をチャット履歴に追加する
    st.session_state.history.append({"role": "user", "content": prompt})
    
    # 回答を生成
    response = query_engine.query(prompt)
    answer_text = response.response if hasattr(response, "response") else str(response)
    
    with st.chat_message("assistant"):
        st.write(answer_text)
        
    # 回答の追加
    st.session_state.history.append({"role": "assistant", "content": answer_text})