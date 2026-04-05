from pymilvus.model.sparse import BM25EmbeddingFunction
import json

bm25_ef = BM25EmbeddingFunction()
texts = ["hello world", "test message", "the quick brown fox"]
bm25_ef.fit(texts)

res = bm25_ef.encode_documents(texts)
results = []
for row in res:
    # row is a 1D csr_array
    row_dict = {int(k): float(v) for k, v in zip(row.indices, row.data)}
    results.append(row_dict)

print("First dict:", results[0])

# Let's verify with Milvus insert directly
from src.rag.retriever import MilvusVectorDB
db = MilvusVectorDB(collection_name="debug_insert")
fake_dense = [[0.1]*1024 for _ in range(3)]
try:
    db.insert(chunks=texts, embeddings=fake_dense, sparse_embeddings=results)
    print("Insert successful!")
except Exception as e:
    print("Insert failed:", e)

