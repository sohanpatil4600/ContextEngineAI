import os
from typing import List, Dict, Any, Optional
from src.document_processing import TensorLakeClient, RESEARCH_PAPER_SCHEMA
from src.rag.embeddings import ContextualizedEmbeddings, SparseEmbeddings, VoyageReRanker
from src.rag.retriever import MilvusVectorDB
from src.generation import StructuredResponseGen

class RAGPipeline:
    """Unified RAG pipeline combining document parsing, dense/sparse embeddings, and reranked retrieval"""
    def __init__(
        self,
        tensorlake_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        milvus_db_path: str = "milvus_lite.db",
        collection_name: str = "research_assistant"
    ):
        self.doc_parser = TensorLakeClient(api_key=tensorlake_api_key)
        self.embeddings = ContextualizedEmbeddings(api_key=voyage_api_key)
        self.sparse_embeddings = SparseEmbeddings()
        self.reranker = VoyageReRanker(api_key=voyage_api_key)
        
        self.vector_db = MilvusVectorDB(db_path=milvus_db_path, collection_name=collection_name)
        self.generator = StructuredResponseGen(api_key=openai_api_key)
        
    def process_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        results = {
            "processed_docs": [],
            "total_chunks": 0,
            "structured_data": []
        }
        
        # Upload and parse documents
        file_ids = self.doc_parser.upload(document_paths)
        
        for i, (path, file_id) in enumerate(zip(document_paths, file_ids)):
            parse_id = self.doc_parser.parse_structured(
                file_id=file_id,
                json_schema=RESEARCH_PAPER_SCHEMA,
                labels={"source": path, "doc_index": i}
            )
            
            parse_result = self.doc_parser.get_result(parse_id)
            if parse_result is None:
                raise Exception(f"TensorLake parsing failed for {path}: No result returned")
            
            # Extract chunks and structured data
            chunks = []
            if hasattr(parse_result, 'chunks') and parse_result.chunks:
                for chunk in parse_result.chunks:
                    if chunk and hasattr(chunk, 'content') and hasattr(chunk, 'page_number'):
                        chunks.append({
                            "page": chunk.page_number,
                            "text": chunk.content,
                            "source": path
                        })
            else:
                raise Exception(f"TensorLake parsing failed for {path}: No chunks found in result")
            
            if not chunks:
                raise Exception(f"TensorLake parsing failed for {path}: No valid chunks extracted")
            
            # Generate contextualized embeddings (Dense)
            flat_chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_texts = [flat_chunk_texts]
            embeddings_result = self.embeddings.embed_document_chunks(chunk_texts)
            
            if not embeddings_result or len(embeddings_result) == 0:
                raise Exception(f"Embedding generation failed for {path}: No embeddings returned")
            
            chunk_embeddings = embeddings_result[0]
            
            if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
                raise Exception(f"Embedding generation failed for {path}: Chunk count mismatch.")
            
            # Generate Sparse Embeddings (BM25)
            sparse_embeddings_result = self.sparse_embeddings.embed_documents(flat_chunk_texts)

            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "page_number": chunk.get("page", 0),
                    "chunk_index": i,
                    "source_file": chunk.get("source", path)
                }
                chunk_metadata.append(metadata)
            
            # Store in vector database with metadata and both embedding types
            self.vector_db.insert(
                chunks=flat_chunk_texts,
                embeddings=chunk_embeddings,
                sparse_embeddings=sparse_embeddings_result,
                metadata=chunk_metadata
            )
            
            results["processed_docs"].append({
                "path": path,
                "file_id": file_id,
                "chunks_count": len(chunks),
                "structured_data": parse_result.model_dump()
            })
            results["total_chunks"] += len(chunks)
            results["structured_data"].append(parse_result.model_dump())
            
        return results
    
    def retrieve_context(self, query: str, top_k: int = 3, hybrid_limit: int = 15) -> List[Dict[str, Any]]:
        # We need a trained BM25 model to query.
        if not self.sparse_embeddings.is_fit:
             # Fallback to pure dense search if no documents were processed in this session 
             # (Milvus Lite limits cross-session BM25 model persistence easily)
             query_dense_embedding = self.embeddings.embed_query(query)
             search_results = self.vector_db.search(query_dense_embedding, limit=hybrid_limit)
        else:
             # Generate embeddings for both Dense and Sparse
             query_dense_embedding = self.embeddings.embed_query(query)
             query_sparse_embedding = self.sparse_embeddings.embed_query(query)
             
             # Search vector database using Hybrid Search & RRF
             search_results = self.vector_db.hybrid_search(
                 query_dense_embedding=query_dense_embedding,
                 query_sparse_embedding=query_sparse_embedding,
                 limit=hybrid_limit
             )
        
        if not search_results:
            return []
            
        # Cross-Encoder Re-Ranking using Voyage
        documents_to_rerank = [hit["text"] for hit in search_results]
        reranked_results = self.reranker.rerank(query=query, documents=documents_to_rerank, top_k=top_k)
        
        # Match back to original chunks to restore metadata
        final_results = []
        for rank in reranked_results:
            original_hit = search_results[rank["index"]]
            original_hit["score"] = rank["relevance_score"] # Inject cross-encoder confidence
            final_results.append(original_hit)
            
        return final_results
    
    def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]],
        source_used: str = "RAG"
    ) -> Dict[str, Any]:
        context_blocks = [result["text"] for result in context]
        
        response = self.generator.generate(
            query=query,
            context_blocks=context_blocks,
            source_used=source_used
        )
        
        return response
    
    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        context_results = self.retrieve_context(query, top_k=top_k)
        response = self.generate_response(query, context_results)
        
        # Add retrieval metadata for citations
        response["retrieval_metadata"] = {
            "retrieved_chunks": len(context_results),
            "top_scores": [r["score"] for r in context_results]
        }
        
        return response

