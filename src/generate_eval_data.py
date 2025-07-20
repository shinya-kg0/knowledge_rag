import os
import sys
import csv
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
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
また、ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「すみません。わかりません。」と回答してください。
回答の中に情報源の提示は含めないでください。例えば、回答の中に「[Source1]」や「Sources:」という形で情報源を示すことはしないでください。
"""

# ユーザーの質問に対して回答する関数
# 引数はチャットの履歴（JSON配列）
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
    
    return answer, source

# ユーザーの質問を読み込むための関数を定義する
def load_questions(filepath):
    questions = []
    
    with open(filepath, mode="r", encoding="utf-8") as file:
        # csvを辞書形式で読み込む
        reader = csv.DictReader(file)
        # 各行の"question", "ground_truth"列の値をリストに追加していく
        for row in reader:
            questions.append((row["question"], row["ground_truth"]))
    
    return questions

def generate_evaluation_dataset(questions, file_name):
    # evaluation_dataset.csvというファイルを新規作成または上書きして開く
    with open(file_name, mode="w", newline="", encoding="utf-8") as f:
        # CSVライターを作成、すべての項目をダブルクオーテーションで囲む
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["query", "response", "context", "ground_truth"])
        
        # 質問ごとに処理
        for question, ground_truth in questions:
            history = [{"role": "user", "content": question}]
            response, context = search(history)

            writer.writerow(
                [question, 
                response.replace("\n", " "), 
                " ".join(context).replace("\n", ""), 
                ground_truth])

if __name__ == "__main__":
    csv_file_path = sys.argv[1]
    output_file = sys.argv[2]
    
    questions = load_questions(csv_file_path)
    
    generate_evaluation_dataset(questions, output_file)
            
