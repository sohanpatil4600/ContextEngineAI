from pymilvus.model.sparse import BM25EmbeddingFunction
bm25_ef = BM25EmbeddingFunction()
bm25_ef.fit(["hello world", "test message"])
docs = bm25_ef.encode_documents(["hello world"])
print(docs)
