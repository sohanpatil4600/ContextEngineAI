from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

class MilvusVectorDB:
    def __init__(self, db_path: str = "milvus_lite.db", collection_name: str = "research_assistant"):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self, dim: int = 1024):
        if self.client.has_collection(collection_name=self.collection_name):
            return

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("source_file", DataType.VARCHAR, max_length=500)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="IVF_FLAT", metric_type="COSINE")
        index_params.add_index("sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def insert(self, chunks: List[str], embeddings: List[List[float]], sparse_embeddings: List[Any] = None, metadata: List[Dict[str, Any]] = None):
        assert len(chunks) == len(embeddings), "Mismatch between chunks and embeddings"
        
        if metadata:
            assert len(chunks) == len(metadata), "Mismatch between chunks and metadata"
            
        if sparse_embeddings is not None:
            assert len(chunks) == len(sparse_embeddings), "Mismatch between chunks and sparse_embeddings"

        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            entry = {
                "text": chunk,
                "embedding": emb,
            }
            if sparse_embeddings is not None:
                entry["sparse_embedding"] = sparse_embeddings[i]
                
            if metadata and i < len(metadata):
                meta = metadata[i]
                entry["page_number"] = meta.get("page_number", 0)
                entry["chunk_index"] = meta.get("chunk_index", i)
                entry["source_file"] = meta.get("source_file", "unknown")
            else:
                entry["page_number"] = 0
                entry["chunk_index"] = i
                entry["source_file"] = "unknown"
            
            data.append(entry)

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        self.client.flush(collection_name=self.collection_name)

    def get_collection_count(self) -> int:
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return stats.get('row_count', 0)
        except:
            return 0

    def search(
        self,
        query_embedding: List[float],
        limit: int = 3,
        nprobe: int = 10,
        metric: str = "COSINE"
    ) -> List[Dict[str, Any]]:
        """Legacy dense-only search"""
        search_params = {"metric_type": metric, "params": {"nprobe": nprobe}}

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=limit,
            output_fields=["text", "page_number", "chunk_index", "source_file"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "page_number": hit.entity.get("page_number", 0),
                "chunk_index": hit.entity.get("chunk_index", 0),
                "source_file": hit.entity.get("source_file", "unknown")
            })

        return hits

    def hybrid_search(
        self,
        query_dense_embedding: List[float],
        query_sparse_embedding: Any,
        limit: int = 5,
        nprobe: int = 10,
    ) -> List[Dict[str, Any]]:
        """Performs a hybrid search combining dense and sparse embeddings via RRF"""
        
        dense_req = AnnSearchRequest(
            data=[query_dense_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
            limit=limit
        )

        sparse_req = AnnSearchRequest(
            data=[query_sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=limit
        )

        reqs = [dense_req, sparse_req]
        # Use Reciprocal Rank Fusion
        ranker = RRFRanker()

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=["text", "page_number", "chunk_index", "source_file"]
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "page_number": hit.entity.get("page_number", 0),
                "chunk_index": hit.entity.get("chunk_index", 0),
                "source_file": hit.entity.get("source_file", "unknown")
            })

        return hits