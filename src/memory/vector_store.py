import logging
import uuid
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
    """Custom embedding function to wrap sentence-transformers locally."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def __call__(self, input: Documents) -> Embeddings:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

class EvidenceMemory:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", collection_name: str = "research_evidence"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
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
        """Query memory and return list of Evidence."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        evidence_list = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            # Chroma scores are distances (L2 square by default), so lower distance means closer match.
            # We map this to a pseudo score for sorting.
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Convert L2 distance to dynamic score (e.g. cosine similarity approx)
                score = 1.0 / (1.0 + dist)
                evidence_list.append(Evidence(
                    text=doc,
                    source_url=meta.get("source_url", "Unknown"),
                    title=meta.get("title", "Unknown"),
                    score=float(score)
                ))
        
        # Sort evidence by similarity score descending
        evidence_list.sort(key=lambda x: x.score, reverse=True)
        return evidence_list

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
