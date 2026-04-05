import os
import voyageai
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pymilvus.model.sparse import BM25EmbeddingFunction

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")


class ContextualizedEmbeddings:
    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-context-3"):
        self.client = voyageai.Client(api_key=api_key or VOYAGE_API_KEY)
        self.model = model

    def embed_document_chunks(
        self,
        docs_chunks: List[List[str]],
        *,
        output_dimension = 1024,
        output_dtype = "float",
    ) -> List[List[List[float]]]:

        resp = self.client.contextualized_embed(
            inputs=docs_chunks,
            model=self.model,
            input_type="document",               
            output_dimension=output_dimension,   
            output_dtype=output_dtype,          
        )
        return [r.embeddings for r in resp.results]

    def embed_query(
        self,
        query,
        *,
        output_dimension = None,
        output_dtype: Literal["float", "int8", "uint8", "binary", "ubinary"] = "float",
    ) -> List[float]:
        
        resp = self.client.contextualized_embed(
            inputs=[[query]],
            model=self.model,
            input_type="query",
            output_dimension=output_dimension,
            output_dtype=output_dtype,
        )
        return resp.results[0].embeddings[0]


class SparseEmbeddings:
    def __init__(self):
        self.bm25_ef = BM25EmbeddingFunction()
        self.is_fit = False

    def fit(self, texts: List[str]):
        """Fits the BM25 model vocabulary onto the input texts."""
        if not texts:
            return
        self.bm25_ef.fit(texts)
        self.is_fit = True

    def embed_documents(self, texts: List[str]):
        """Generates sparse embeddings for documents."""
        if not texts:
            return []
        if not self.is_fit:
            self.fit(texts)
            
        import scipy.sparse
        res = self.bm25_ef.encode_documents(texts)
        # res is usually a csr_array or csr_matrix of shape (N, M).
        if not isinstance(res, (scipy.sparse.csr_matrix, scipy.sparse.csr_array)):
            res = scipy.sparse.csr_matrix(res)
            
        results = []
        # We must extract exactly 1xM chunks (2D) for Milvus. (M,) flattening throws 'expect 1 row'.
        for i in range(res.shape[0]):
            slice_2d = res[i:i+1, :]
            results.append(slice_2d)
            
        return results
        
    def embed_query(self, query: str):
        """Generates sparse embedding for a query."""
        if not self.is_fit:
            raise ValueError("BM25 model not fitted yet. Ensure documents are embedded first.")
        
        import scipy.sparse
        res = self.bm25_ef.encode_queries([query])
        if not isinstance(res, (scipy.sparse.csr_matrix, scipy.sparse.csr_array)):
            res = scipy.sparse.csr_matrix(res)
            
        return res[0:1, :]


class VoyageReRanker:
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-2"):
        self.client = voyageai.Client(api_key=api_key or VOYAGE_API_KEY)
        self.model = model
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[dict]:
        """
        Re-ranks the documents according to their relevance to the query.
        Returns a list of dicts with 'index', 'relevance_score', and 'text'.
        """
        if not documents:
            return []
            
        resp = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k
        )
        
        results = []
        for r in resp.results:
            results.append({
                "index": r.index,
                "relevance_score": r.relevance_score,
                "text": r.document
            })
        return results