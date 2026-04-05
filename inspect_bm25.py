from pymilvus.model.sparse import BM25EmbeddingFunction

bm25_ef = BM25EmbeddingFunction()
texts = ["hello world", "test message"]
bm25_ef.fit(texts)
res = bm25_ef.encode_documents(texts)

# If res is a csr_array, iterating over it gives 1D arrays
results = []
for row in res:
    # row is a 1D csr_array
    row_dict = {int(k): float(v) for k, v in zip(row.indices, row.data)}
    results.append(row_dict)

print(f"Results: {results}")

