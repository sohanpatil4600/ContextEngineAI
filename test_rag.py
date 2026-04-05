from src.rag.rag_pipeline import RAGPipeline

def main():
    print("Initializing RAGPipeline...")
    pipeline = RAGPipeline()
    
    doc_path = "data/attention-is-all-you-need-Paper.pdf"
    print(f"Processing document: {doc_path}...")
    try:
        pipeline.process_documents([doc_path])
        print("Document processed successfully!")
        
        query = "What is the Transformer architecture?"
        print(f"Querying: {query}")
        
        results = pipeline.retrieve_context(query=query, top_k=2)
        print("\n--- RESULTS ---")
        for res in results:
            print(f"Score: {res.get('score')} | Chunk Index: {res.get('chunk_index')} | Text: {res.get('text')[:100]}...")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
