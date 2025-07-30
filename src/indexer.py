import os
import sys
import glob
import json
import logging

import settings as st
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.core.credentials import AzureKeyCredential
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore

# ロギング設定
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.DEBUG
)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

DATA_PATH = "../data"
ID_FIELD_KEY = "id"
CHUNK_FIELD_KEY = "content"
EMBEDDING_FIELD_KEY = "contentVector"
METADATA_STRING_FIELD_KEY = "metadata_json"
DOC_ID_FIELD_KEY = "doc_id"

Settings.transformations = [
    MarkdownNodeParser(
        include_metadata=True,  # 見出しレベル、見出しテキストなどのメタデータを含める
        include_prev_next_rel=True,  # 前後の兄弟ノードへの関係を含める
        include_hierarchy_metadata=True,  # 親ノードのIDやレベルなど、階層情報を含める)
    ),
    AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=st.AOAI_EMBEDDING_MODEL_NAME,
        api_key=st.AOAI_API_KEY,
        azure_endpoint=st.AOAI_ENDPOINT,
        api_version=st.AOAI_API_VERSION,
    ),
]


# ドキュメントを取得して、チャンク分割、ベクトル化、AI Searchにアップロード
def create_and_upload_chunks(documents_path: list[str]):

    logging.info(f"Loading documents from: {documents_path}")
    # ドキュメントの読み込み
    documents = SimpleDirectoryReader(input_dir=documents_path).load_data()
    logging.info(f"Loaded {len(documents)} documents.")

    search_client = SearchClient(
        endpoint=st.SEARCH_SERVICE_ENDPOINT,
        index_name=st.SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(st.SEARCH_SERVICE_API_KEY),
    )

    # 警告を解消するためにインスタンス化
    async_search_client = AsyncSearchClient(
        endpoint=st.SEARCH_SERVICE_ENDPOINT,
        index_name=st.SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(st.SEARCH_SERVICE_API_KEY),
    )

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=search_client,
        async_search_or_index_client=async_search_client,
        id_field_key=ID_FIELD_KEY,
        chunk_field_key=CHUNK_FIELD_KEY,
        embedding_field_key=EMBEDDING_FIELD_KEY,
        metadata_string_field_key=METADATA_STRING_FIELD_KEY,
        doc_id_field_key=DOC_ID_FIELD_KEY,
        dim=1536,
    )

    # IngestionPipeline の設定
    pipeline = IngestionPipeline(
        transformations=Settings.transformations,
        vector_store=vector_store,
    )

    logging.info("Starting ingestion pipeline...")
    try:
        nodes = pipeline.run(documents=documents)
        output_nodes_path = "debug_nodes_for_azure.json"
        serializable_nodes_for_azure = []
        for node in nodes:
            # デバック用
            doc = {
                ID_FIELD_KEY: node.node_id,
                CHUNK_FIELD_KEY: node.text,
                EMBEDDING_FIELD_KEY: (
                    node.embedding if node.embedding is not None else None
                ),  # numpy array to list
                METADATA_STRING_FIELD_KEY: (
                    json.dumps(node.metadata, ensure_ascii=False)
                    if node.metadata
                    else None
                ),  # metadataをJSON文字列に変換
                DOC_ID_FIELD_KEY: node.ref_doc_id,
            }
            serializable_nodes_for_azure.append(doc)

        with open(output_nodes_path, "w", encoding="utf-8") as f:
            json.dump(serializable_nodes_for_azure, f, ensure_ascii=False, indent=2)
        logging.info(
            f"Generated nodes (Azure AI Search format) saved to {output_nodes_path}"
        )

    except Exception as e:
        logging.error(f"エラー発生: {e}")

    finally:
        logging.info(
            f"Successfully processed {len(nodes)} nodes and uploaded to Azure AI Search."
        )
        logging.info("Ingestion complete.")


if __name__ == "__main__":

    # Markdown ファイルのパスを取得
    markdown_files = glob.glob(os.path.join(DATA_PATH, "*.md"))

    if not markdown_files:
        logging.error(
            f"'{DATA_PATH}' ディレクトリに Markdown ファイルが見つかりませんでした。"
        )
        sys.exit(1)

    create_and_upload_chunks(DATA_PATH)
