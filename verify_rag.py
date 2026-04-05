import os
from src.rag.rag_pipeline import RAGPipeline
import json

def test_pipeline():
    print("Initializing Pipeline against existing Milvus DB...")
    # Initialize RAGPipeline which connects to the existing milvus_lite.db
    # that your Streamlit app just populated!
    pipeline = RAGPipeline(
        tensorlake_api_key=os.getenv("TENSORLAKE_API_KEY"),
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    query = "what is the document about?"
    print(f"\nQuerying: '{query}'")
    
    try:
        results = pipeline.retrieve_context(query=query, top_k=3, hybrid_limit=10)
        
        print(f"\n✅ Retrieved {len(results)} chunks successfully!")
        for i, res in enumerate(results):
            score = res.get("score")
            text = res.get("text", "")[:150].replace('\n', ' ')
            print(f"[{i+1}] Score: {score:.4f} | Chunk: ...{text}...")
            
    except Exception as e:
        print("\n❌ Pipeline Error:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
