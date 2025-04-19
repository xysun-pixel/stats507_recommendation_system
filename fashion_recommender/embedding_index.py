from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
import uuid
from .base import BaseComponent

class EmbeddingIndex(BaseComponent):
    def __init__(self, embedding_model_id, qdrant_url, qdrant_key, collection_name, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        self.collection_name = collection_name
        self.qdrant = Qdrant(client=self.client, collection_name=collection_name, embeddings=self.embedding_model)

    def create_or_reset_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            quantization_config=models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=models.CompressionRatio.X8,
                    always_ram=True,
                )
            ),
        )

    def index_documents(self, docs):
        BATCH_SIZE = 256
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i + BATCH_SIZE]
            batch_vectors = self.embedding_model.embed_documents(batch_docs)
            points = [
                PointStruct(id=uuid.uuid4().int >> 64, vector=vec, payload={"text": doc})
                for vec, doc in zip(batch_vectors, batch_docs)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def retrieve(self, query, k):
        return self.qdrant.similarity_search_with_score(query, k=k)
