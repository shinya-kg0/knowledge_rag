import os
import sys
import glob

from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


# 環境変数の読み込み
load_dotenv(verbose=True)

SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY")
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME")
AOAI_ENDPOINT=os.environ.get("AOAI_ENDPOINT")
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") 
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") 
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME")

DATA_PATH = "../data"


# ドキュメントからテキストを抽出する
def extract_text_from_docs(filename: str):
    lines = []
    text = ""
    
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    
    text = "".join(lines)

    return text

# テキストを指定したサイズで分割する
def create_chunk(content: str):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    chunks = splitter.split_text(content)
    
    # デバック用
    for i, chunk in enumerate(chunks):
        print(f"-----チャンク{i}-----")
        print(chunk.page_content)

    return chunks

# チャンクをインデックスに登録する
def index_docs(chunks: list):
    # クライアントを作成
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
    
    for i, chunk in enumerate(chunks):
        print(f"{i+1}個目のチャンクを処理中...")
        
        response = openai_client.embeddings.create(
            input=chunk.page_content,
            model=AOAI_EMBEDDING_MODEL_NAME
        )
        
        document = {"id": str(i), "content": chunk.page_content, "contentVector": response.data[0].embedding}
        search_client.upload_documents([document])


if __name__ == "__main__":
    
    markdown_files = glob.glob(os.path.join(DATA_PATH, "*.md"))

    # ドキュメントからテキストを抽出する
    content = ""
    for filename in markdown_files:
        content += extract_text_from_docs(filename)
    
    # ドキュメントから抽出したテキストをチャンクに分解する
    chunks = create_chunk(content)
    
    # AI Searchにインデックスする
    index_docs(chunks)
    
    print("インデックスの作成完了")