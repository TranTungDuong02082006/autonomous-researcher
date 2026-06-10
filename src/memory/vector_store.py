import logging
import uuid
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

logger = logging.getLogger(__name__)

@dataclass
class Evidence:
    text: str
    source_url: str
    title: str
    score: float

class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function quản lý nạp mô hình dạng Thread-Safe Singleton.
    Đảm bảo mô hình chỉ load ĐÚNG 1 LẦN LÊN RAM, triệt tiêu 100% lỗi Meta Tensor khi chạy đa luồng!
    """
    _shared_model = None  # Biến tĩnh lưu cache mô hình dùng chung giữa các luồng
    _lock = threading.Lock() # Khóa luồng cưỡng chế

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Cắm khóa luồng: Chỉ cho phép duy nhất 1 luồng xử lý khởi tạo tại 1 thời điểm
        with LocalSentenceTransformerEmbeddingFunction._lock:
            if LocalSentenceTransformerEmbeddingFunction._shared_model is None:
                from sentence_transformers import SentenceTransformer
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"📦 [Thread-Safe] Đang nạp ĐỘC NHẤT mô hình nhúng vào RAM: {self.model_name} trên thiết bị: [{device.upper()}]...")
                
                # Khởi tạo bản model golden dùng chung cho toàn bộ hệ thống
                LocalSentenceTransformerEmbeddingFunction._shared_model = SentenceTransformer(self.model_name, device=device)
        
        # Gọi bản model đã cache trong RAM ra encode dữ liệu, tốc độ nhanh gấp bách lần
        embeddings = LocalSentenceTransformerEmbeddingFunction._shared_model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

class EvidenceMemory:
    def __init__(self, embedding_model: str = "BAAI/bge-m3", collection_name: str = "research_evidence", reranker: Any = None):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.reranker = reranker
        
        # Initialize in-memory chromadb client for session-based isolation
        self.chroma_client = chromadb.EphemeralClient()
        self.embedding_fn = LocalSentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def add(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        """Add chunks with corresponding metadata to the memory store."""
        if not chunks:
            return
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Ensure metadata is in a safe format for Chroma (simple types)
        safe_metadata = []
        for meta in metadata:
            safe_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            safe_metadata.append(safe_meta)

        self.collection.add(
            documents=chunks,
            metadatas=safe_metadata,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} chunks to vector memory.")

    def search(self, query: str, top_k: int = 5) -> List[Evidence]:
        """Query memory and perform semantic validation to clean data output."""
        count = self.collection.count()
        if count == 0:
            logger.info("Vector collection is empty. Returning empty evidence list.")
            return []
            
        n_candidates = top_k * 4 if self.reranker is not None else top_k * 2
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_candidates, count) # Lấy tập ứng viên rộng hơn
        )
        
        evidence_list = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            # Dynamically extract significant keywords from query for context guardrail
            # instead of hard-coding domain-specific terms
            query_lower = query.lower()
            _stopwords = {
                "what", "how", "why", "when", "where", "which", "who", "does",
                "the", "and", "for", "are", "but", "not", "you", "all", "can",
                "had", "her", "was", "one", "our", "out", "has", "have", "been",
                "from", "with", "they", "this", "that", "will", "would", "there",
                "their", "about", "between", "through", "during", "before", "after",
                "above", "below", "than", "each", "some", "such", "into", "over",
                "also", "most", "other", "just", "more", "very", "only", "then",
                "compare", "discuss", "explain", "describe", "analyze", "evaluate",
                "versus", "difference", "differences", "between", "impact", "effects",
            }
            query_words = query_lower.split()
            mandatory_keywords = [
                w for w in query_words
                if len(w) > 3 and w not in _stopwords
            ]

            for doc, meta, dist in zip(docs, metas, distances):
                score = 1.0 / (1.0 + dist)
                doc_lower = doc.lower()
                
                # Semantic guardrail: if query contains specific technical terms,
                # require at least one to appear in the retrieved chunk
                if mandatory_keywords and not any(kw in doc_lower for kw in mandatory_keywords):
                    continue
                    
                evidence_list.append(Evidence(
                    text=doc,
                    source_url=meta.get("source_url", "Unknown"),
                    title=meta.get("title", "Unknown"),
                    score=float(score)
                ))
        
        evidence_list.sort(key=lambda x: x.score, reverse=True)
        
        if self.reranker is not None and evidence_list:
            candidate_texts = [ev.text for ev in evidence_list]
            logger.info(f"Reranking {len(candidate_texts)} candidates using custom CrossEncoder...")
            try:
                top_passages = self.reranker.rerank(query, candidate_texts, top_k=top_k)
                
                # Map back to Evidence objects to preserve metadata
                evidence_by_text = {ev.text: ev for ev in evidence_list}
                reranked_evidence = []
                for i, passage in enumerate(top_passages):
                    if passage in evidence_by_text:
                        ev = evidence_by_text[passage]
                        # Set a sequence-based score to keep descending sorted order
                        ev.score = 1.0 - (i * 0.01)
                        reranked_evidence.append(ev)
                return reranked_evidence
            except Exception as e:
                logger.warning(f"Error during reranking: {e}. Falling back to default embedding distance ranking.")
                
        return evidence_list[:top_k] # Trả về đúng top_k tinh khiết nhất

    def clear_session(self):
        """Reset / delete current memory collection to start a clean research session."""
        logger.info("Clearing memory session...")
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
