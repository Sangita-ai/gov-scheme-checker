import os

_chroma_client = None
_collection = None
_embedding_model = None

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "schemes"


def _get_embedding_model():
    """Lazy-load sentence-transformer model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("⏳ Loading embedding model (first time ~30 seconds)...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded.")
        except Exception as e:
            print(f"⚠️  Could not load sentence-transformers: {e}")
            _embedding_model = None
    return _embedding_model


def _get_collection():
    """Lazy-load ChromaDB collection. Compatible with v0.4.x and v0.5.x."""
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb
            os.makedirs(CHROMA_PATH, exist_ok=True)

            
            try:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            except AttributeError:
                
                from chromadb.config import Settings
                _chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=CHROMA_PATH,
                    anonymized_telemetry=False
                ))

            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"✅ ChromaDB collection ready ({_collection.count()} embeddings).")
        except Exception as e:
            print(f"⚠️  ChromaDB init failed: {e}")
            print("   → Eligibility checking will still work via SQL rules.")
            _collection = None
    return _collection


def add_scheme_to_vector_store(scheme: dict) -> bool:
    """Add or update a scheme in ChromaDB. Returns True on success."""
    model = _get_embedding_model()
    collection = _get_collection()
    if model is None or collection is None:
        return False

    try:
        text = (
            f"{scheme.get('name_en', '')} "
            f"{scheme.get('benefit_summary', '')} "
            f"{scheme.get('benefit_type', '')} "
            f"{scheme.get('ministry', '')} "
            f"state:{scheme.get('state', 'all')} "
            f"occupation:{scheme.get('occupation_list', '')} "
            f"gender:{scheme.get('gender', 'any')}"
        )
        embedding = model.encode(text).tolist()

        
        collection.upsert(
            documents=[text],
            embeddings=[embedding],
            ids=[scheme["id"]],
            metadatas=[{
                "scheme_id": scheme["id"],
                "name": scheme.get("name_en", "")
            }],
        )
        return True
    except Exception as e:
        print(f"⚠️  Failed to add scheme {scheme.get('id')} to vector store: {e}")
        return False


def find_relevant_schemes(user_query: str, n_results: int = 10) -> list:
    """Semantic search — returns list of scheme_ids. Falls back to [] on failure."""
    model = _get_embedding_model()
    collection = _get_collection()
    if model is None or collection is None:
        return []

    try:
        count = collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)
        query_embedding = model.encode(user_query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        if results and results.get("metadatas") and results["metadatas"][0]:
            return [meta["scheme_id"] for meta in results["metadatas"][0]]
        return []
    except Exception as e:
        print(f"⚠️  Vector search failed: {e}")
        return []


def get_vector_store_count() -> int:
    """Return number of documents in vector store (0 on failure)."""
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        return collection.count()
    except Exception:
        return 0
