import os
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import streamlit as st
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv(verbose=True)

# 環境変数から接続情報を取得
SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY")
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME")
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") 
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") 
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") 
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME")
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME")

# AIのキャラクターを決めるためのシステムメッセージを定義する。
system_message_chat_conversation = """
あなたはユーザーの質問に回答するチャットボットです。
回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
「Sources:」に記載されている情報以外の回答はしないでください。
情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
また、ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「すみません、わかりません。」と回答してください。
回答の中に情報源の提示は含めないでください。例えば、回答の中に「[Source1]」や「Sources:」という形で情報源を示すことはしないでください。
"""

# ユーザーの質問に対して回答する関数
def search(history: list[dict]):
    """
    [{'role': 'user', 'content': '有給は何日取れますか？'},{'role': 'assistant', 'content': '10日です'},
    {'role': 'user', 'content': '一日の労働上限時間は？'}...]というJSON配列から
    最も末尾に格納されているJSONオブジェクトのcontent(=ユーザーの質問)を取得する。
    """
    question = history[-1].get("content")
    
    # それぞれのクライアントを作成
    search_client = SearchClient(
        endpoint=SEARCH_SERVICE_ENDPOINT,
        index_name=SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_SERVICE_API_KEY)
    )
    openai_client = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION
    )
    
    # ユーザーの質問をベクトル化する
    response = openai_client.embeddings.create(
        input=question,
        model=AOAI_EMBEDDING_MODEL_NAME
    )
    
    # AI Searchで検索できるようにクエリを生成
    vector_query = VectorizedQuery(
        vector=response.data[0].embedding,
        k_nearest_neighbors=3,
        fields="contentVector"
    )
    
    # AI Searchに対してベクトル検索を行う
    results = search_client.search(
        vector_queries=[vector_query],
        select=["id", "content"])
    
    messages = []
    
    # 先頭にキャラ付けのシステムメッセージを追加する
    messages.insert(0, {"role": "system", "content": system_message_chat_conversation})
    
    sources = ["[Source" + result["id"] + "]: " + result["content"] for result in results]
    source = "\n".join(sources)
    
    user_message = """
    {query}
    
    Sources:
    {source}
    """.format(query=question, source=source)
    
    messages.append({"role": "user", "content": user_message})
    
    # 回答を生成させる
    response = openai_client.chat.completions.create(
        model=AOAI_CHAT_MODEL_NAME,
        messages=messages
    )
    answer = response.choices[0].message.content
    
    return answer

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
    response = search(st.session_state.history)
    
    with st.chat_message("assistant"):
        st.write(response)
        
    # 回答の追加
    st.session_state.history.append({"role": "assistant", "content": response})