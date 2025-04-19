import pandas as pd
import time
from .llm_wrapper import LLMWrapper
from .embedding_index import EmbeddingIndex
from .prompt import FashionPrompt

class FashionRecommender(LLMWrapper, EmbeddingIndex, FashionPrompt):
    def __init__(self, model_id, embedding_model_id, qdrant_url, qdrant_key, collection_name="articles"):
        super().__init__(
            model_id=model_id,
            embedding_model_id=embedding_model_id,
            qdrant_url=qdrant_url,
            qdrant_key=qdrant_key,
            collection_name=collection_name
        )

    def index_articles(self, csv_path):
        df = pd.read_csv(csv_path)
        docs = [f"{r['prod_name']} - {r['product_type_name']} - {r['colour_group_name']} - {r['index_name']} - {r['detail_desc']}" for _, r in df.iterrows()]
        self.create_or_reset_collection()
        self.index_documents(docs)

    def recommend(self, question, k=20, topn=10):
        T1 = time.time()
        results = self.retrieve(question, k)
        context = "\n\n".join([doc.page_content for doc, _ in results])
        T2 = time.time()

        response = (self.prompt | self.chain).invoke({"num": topn, "question": question, "context": context})
        answer = response.split("Answer:")[-1].strip()
        T3 = time.time()

        print(f"🔍 Retrieval time: {(T2-T1)*1000:.2f} ms")
        print(f"💬 Generation time: {(T3-T2)*1000:.2f} ms")
        return answer
