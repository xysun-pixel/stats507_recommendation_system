import os
from pathlib import Path
from fashion_recommender.recommender import FashionRecommender

if __name__ == "__main__":
    # === 参数配置 ===
    MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
    EMBEDDING_MODEL_ID = "BAAI/bge-small-en"
    QDRANT_URL = "https://b91a21f9-3b5e-4707-b398-e61ddb39a0a8.us-east-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CQSaMqv-PlbjAZb8li38pYePHVywekT6XaIMnhDU8CM"
    COLLECTION_NAME = "articles"
    CSV_PATH = "articles.csv"  # 请确保此文件存在于当前工作目录下

    # === 初始化推荐器 ===
    recommender = FashionRecommender(
        model_id=MODEL_ID,
        embedding_model_id=EMBEDDING_MODEL_ID,
        qdrant_url=QDRANT_URL,
        qdrant_key=QDRANT_KEY,
        collection_name=COLLECTION_NAME
    )

    # === 检查是否已建立索引（自动跳过） ===
    try:
        if not Path(".indexed_flag").exists():
            print("...初次运行，正在索引商品...")
            recommender.index_articles(CSV_PATH)
            Path(".indexed_flag").touch()
            print("索引完成U•ェ•*U!")
        else:
            print("🟢 已检测到索引数据，跳过索引步骤。")
    except Exception as e:
        print("/(ㄒoㄒ)/~~索引失败：", e)
        exit(1)

    # === 多轮交互式推荐 ===
    print("\n欢迎使用 AI 时尚推荐助手 φ(゜▽゜*)♪（输入 'exit' 退出）\n")
    while True:
        query = input("🧠 请输入你的穿搭需求：\n> ")
        if query.lower().strip() in {"exit", "quit"}:
            print("再见，期待下次为你服务！ヾ(￣▽￣)Bye~Bye~")
            break
        try:
            response = recommender.recommend(query, k=20, topn=5)
            print("\n🎁 推荐结果：\n")
            print(response)
            print("\n" + "=" * 60 + "\n")
        except Exception as err:
            print("❌ 推荐失败：", err)
# # 配置参数
# MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
# EMBEDDING_MODEL_ID = "BAAI/bge-small-en"
# QDRANT_URL = "https://b91a21f9-3b5e-4707-b398-e61ddb39a0a8.us-east-1-0.aws.cloud.qdrant.io:6333"         # 你的 Qdrant Cloud 或本地地址
# QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CQSaMqv-PlbjAZb8li38pYePHVywekT6XaIMnhDU8CM"             # Qdrant 授权 key
# COLLECTION_NAME = "articles"
# CSV_PATH = r"C:\Users\Daddy\Desktop\stats 507\final\fashion_recommender\articles.csv"                      # 含商品信息的 CSV 文件
#
# # 初始化推荐器
# recommender = FashionRecommender(
#     model_id=MODEL_ID,
#     embedding_model_id=EMBEDDING_MODEL_ID,
#     qdrant_url=QDRANT_URL,
#     qdrant_key=QDRANT_KEY,
#     collection_name=COLLECTION_NAME
# )
#
# # 步骤 1：索引 CSV 数据（只需要运行一次，已存在可以跳过）
# print("📦 Indexing articles...")
# recommender.index_articles(CSV_PATH)
#
# # 步骤 2：执行推荐
# query = "I want a black coat that makes me look like a movie star."
# print(f"\n🧠 User query: {query}")
# response = recommender.recommend(query, k=20, topn=5)
#
# print("\n🎁 Recommendation Result:\n")
# print(response)