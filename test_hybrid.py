import random
from src.rag.retriever import MilvusVectorDB

def main():
    print("🚀 Initializing Milvus Vector DB for Hybrid Search Test...")
    # Initialize DB (forces collection to rebuild the hybrid schema)
    db = MilvusVectorDB(collection_name="hybrid_test_collection")
    
    # 1. Create Mock Data
    print("📦 Creating mock documents...")
    mock_chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Generative AI relies heavily on transformer models.",
        "To perform hybrid search, we need dense and sparse vectors."
    ]
    
    # Fake Dense Embeddings (Length 1024, random floats)
    mock_embeddings = [[random.random() for _ in range(1024)] for _ in range(3)]
    
    # Fake Sparse Embeddings (Dictionary representation for Milvus)
    # A sparse embedding is represented as a dict of {integer_index: float_weight}
    mock_sparse = [
        {101: 0.8, 202: 0.5},
        {303: 0.9, 404: 0.1},
        {505: 0.7, 606: 0.9}
    ]
    
    # 2. Insert into DB (this hits both dense and sparse fields)
    print("📥 Inserting mock records with both Dense and Sparse arrays...")
    db.insert(
        chunks=mock_chunks,
        embeddings=mock_embeddings,
        sparse_embeddings=mock_sparse
    )
    
    count = db.get_collection_count()
    print(f"✅ inserted! Total records in DB: {count}")
    
    # 3. Perform Hybrid Search
    print("🔍 Executing Hybrid Search Algorithm (Reciprocal Rank Fusion)...")
    
    # Fake Query Vectors
    fake_query_dense = [random.random() for _ in range(1024)]
    fake_query_sparse = {505: 0.9} # Should heavily match the 3rd document
    
    results = db.hybrid_search(
        query_dense_embedding=fake_query_dense,
        query_sparse_embedding=fake_query_sparse,
        limit=2
    )
    
    print("\n--- HYBRID SEARCH RESULTS ---")
    for r in results:
        print(f"Match: {r['text']}")
        print(f"Combined Score: {r['score']}")
    
    print("\n🎉 If you see scores and texts above, your Hybrid Vector Search correctly accepted both dense and sparse representations!")

if __name__ == '__main__':
    main()
