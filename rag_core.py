import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain Imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.docstore.document import Document

# Intelligence Imports
from sentence_transformers import CrossEncoder

# Local Imports
from configs.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Extension to Language Mapping
EXTENSION_TO_LANG = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.JS,
    ".tsx": Language.JS,
    ".java": Language.JAVA,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".md": Language.MARKDOWN,
}


class RagEngineException(Exception):
    """Base exception for RAG Engine errors."""
    pass


class IngestionError(RagEngineException):
    """Raised when codebase ingestion fails."""
    pass


class RetrievalError(RagEngineException):
    """Raised when retrieval/generation fails."""
    pass


class RagEngine:
    """    
    Features:
    - Language-aware code chunking
    - Two-stage retrieval (vector + cross-encoder)
    - Confidence thresholds to prevent hallucinations
    - Rich metadata tracking
    - Duplicate detection
    """
    
    def __init__(self):
        logger.info("Initializing RAG Engine...")
        
        try:
            # 1. Embeddings
            logger.info(f" Loading Embedding Model: {config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            # 2. Vector Store
            logger.info(f"Connecting to Vector DB at {config.VECTOR_DB_PATH}")
            self._initialize_vector_store()
            
            # 3. Re-ranker
            logger.info(f"Loading Re-ranker: {config.RERANKER_MODEL}")
            self.reranker = CrossEncoder(config.RERANKER_MODEL)
            
            # 4. LLM
            logger.info(f"Connecting to Ollama: {config.OLLAMA_MODEL}")
            self.llm = Ollama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_MODEL,
                temperature=config.LLM_TEMPERATURE,
            )
            
            logger.info("RAG Engine Ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}")
            raise RagEngineException(f"Initialization failed: {str(e)}")
    
    def _initialize_vector_store(self):
        self.vector_store = Chroma(
            persist_directory=config.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="dev_assistant_collection"
        )
    
    # Ingestion Logic

    def ingest_codebase(self, directory_path: str, reset: bool = False) -> Dict[str, Any]:
        start_time = datetime.now()
        
        if not os.path.exists(directory_path):
            raise IngestionError(f"Path does not exist: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise IngestionError(f"Path is not a directory: {directory_path}")
        
        repo_name = os.path.basename(os.path.normpath(directory_path))
        logger.info(f"Ingesting Repo: {repo_name} from {directory_path}")
        
        if reset:
            logger.info("Resetting database...")
            self.reset_database()
        
        try:
            # Load documents with exclusions
            logger.info("Loading files...")
            raw_docs = self._load_documents(directory_path)
            
            if not raw_docs:
                raise IngestionError("No files loaded. Check directory path and permissions.")
            
            logger.info(f"Loaded {len(raw_docs)} raw documents")
            
            # Filter and enrich documents
            logger.info("Filtering and enriching documents...")
            docs_by_lang = self._filter_and_group_documents(raw_docs, directory_path, repo_name)
            
            if not docs_by_lang:
                raise IngestionError(
                    "No supported code files found. "
                    f"Supported extensions: {', '.join(EXTENSION_TO_LANG.keys())}"
                )
            
            total_files = sum(len(docs) for docs in docs_by_lang.values())
            logger.info(f"Processing {total_files} supported files")
            
            # Chunking
            all_chunks = self._chunk_documents(docs_by_lang)
            logger.info(f"Created {len(all_chunks)} chunks")
            
            # Storing in vector database
            logger.info("Embedding and storing in ChromaDB...")
            stored_count = self._store_chunks(all_chunks)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                "status": "success",
                "repo_name": repo_name,
                "files_processed": total_files,
                "chunks_created": len(all_chunks),
                "chunks_stored": stored_count,
                "duration_seconds": round(duration, 2),
                "languages": {
                    lang.value: len(docs) 
                    for lang, docs in docs_by_lang.items()
                }
            }
            
            logger.info(f"Ingestion Complete! {len(all_chunks)} chunks in {duration:.2f}s")
            return stats
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            raise IngestionError(f"Failed to ingest codebase: {str(e)}")
    
    def _load_documents(self, directory_path: str) -> List[Document]:
        all_docs = []
        
        loader_kwargs = {
            "autodetect_encoding": True,
            "encoding": "utf-8"
        }

        for ext in EXTENSION_TO_LANG.keys():
            loader = DirectoryLoader(
                directory_path,
                glob=f"**/*{ext}",
                loader_cls=TextLoader,
                loader_kwargs=loader_kwargs,
                show_progress=False,
                silent_errors=True,
                exclude=config.EXCLUDE_PATTERNS,
                use_multithreading=True
            )
            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load files for extension {ext}: {str(e)}")
                continue
                
        return all_docs

    def _filter_and_group_documents(self, raw_docs: List[Document], directory_path: str, repo_name: str) -> Dict[Language, List[Document]]:
        docs_by_lang = {}
        
        for doc in raw_docs:
            source = doc.metadata.get("source", "")
            ext = os.path.splitext(source)[1].lower()
            
            # Check if extension is supported
            if ext not in EXTENSION_TO_LANG:
                continue
            
            # Skip empty files
            if not doc.page_content.strip():
                continue
            
            # Enrich metadata
            doc.metadata.update({
                "repo": repo_name,
                "file_name": os.path.basename(source),
                "relative_path": os.path.relpath(source, directory_path),
                "language": EXTENSION_TO_LANG[ext].value,
                "extension": ext,
                "ingestion_timestamp": datetime.now().isoformat()
            })
            
            # Group by language
            lang = EXTENSION_TO_LANG[ext]
            if lang not in docs_by_lang:
                docs_by_lang[lang] = []
            docs_by_lang[lang].append(doc)
        
        return docs_by_lang
    
    def _chunk_documents(self, docs_by_lang: Dict[Language, List[Document]]) -> List[Document]:
        
        all_chunks = []
        
        for lang, docs in docs_by_lang.items():
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
            except ValueError:
                # Fallback for languages not supported by specific splitters
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
                
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            logger.info(f"      ├─ {lang.value}: {len(chunks)} chunks from {len(docs)} files")
        
        return all_chunks
    
    def _store_chunks(self, chunks: List[Document]) -> int:
        chunk_ids = []
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            content_for_hash = (
                f"{chunk.metadata.get('relative_path', '')}:"
                f"{chunk.page_content}"
            )
            content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                chunk_ids.append(content_hash)
                unique_chunks.append(chunk)
                seen_hashes.add(content_hash)
        
        # Batch insertion
        stored_count = 0
        for i in range(0, len(unique_chunks), config.BATCH_SIZE):
            batch_chunks = unique_chunks[i:i + config.BATCH_SIZE]
            batch_ids = chunk_ids[i:i + config.BATCH_SIZE]
            
            self.vector_store.add_documents(batch_chunks, ids=batch_ids)
            stored_count += len(batch_chunks)
            
            if stored_count % 500 == 0:
                logger.info(f"      ├─ Stored {stored_count}/{len(unique_chunks)} chunks...")
        
        return stored_count
    
    # Retrieval & Generation Logic
    
    def ask_question(self, query: str, return_debug_info: bool = False) -> Dict[str, Any]:
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        logger.info(f"Question: {query}")
        
        try:
            # Stage 1: Vector Search
            logger.info("Vector similarity search...")
            candidates, vector_scores = self._vector_search(query)
            
            if not candidates:
                return self._create_response(
                    answer="I couldn't find any relevant code in the indexed codebase for your question.",
                    sources=[],
                    confidence="none"
                )
            
            logger.info(f"Found {len(candidates)} candidates")
            
            # Stage 2: Re-ranking
            logger.info("Re-ranking with cross-encoder...")
            ranked_docs, rerank_scores = self._rerank_documents(query, candidates)
            
            # Stage 3: Confidence Check
            top_score = rerank_scores[0] if rerank_scores else -999
            confidence = self._determine_confidence(top_score)
            
            if confidence == "none":
                return self._create_response(
                    answer="I found some code, but it doesn't seem relevant enough to confidently answer your question.",
                    sources=[],
                    confidence="none"
                )
            
            # Stage 4: Context Assembly
            logger.info(f"Assembling context from top {config.FINAL_K} results...")
            context_text, sources = self._assemble_context(
                ranked_docs[:config.FINAL_K],
                rerank_scores[:config.FINAL_K]
            )
            
            # Stage 5: Generation
            logger.info("Generating answer with LLM...")
            answer = self._generate_answer(query, context_text)
            
            response = self._create_response(
                answer=answer,
                sources=sources,
                confidence=confidence
            )
            
            # Add debug info if requested
            if return_debug_info:
                response["debug"] = {
                    "vector_scores": vector_scores[:5],
                    "rerank_scores": rerank_scores[:5],
                    "top_confidence_score": float(top_score)
                }
            
            logger.info("Answer generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise RetrievalError(f"Failed to process question: {str(e)}")
    
    def _vector_search(self, query: str) -> Tuple[List[Document], List[float]]:
        results_with_score = self.vector_store.similarity_search_with_score(
            query,
            k=config.INITIAL_K
        )
        
        candidates = [doc for doc, _ in results_with_score]
        scores = [float(score) for _, score in results_with_score]
        
        return candidates, scores
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> Tuple[List[Document], List[float]]:
        pairs = [[query, doc.page_content] for doc in documents]
        rerank_scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(
            zip(documents, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        ranked_docs = [doc for doc, _ in scored_docs]
        sorted_scores = [float(score) for _, score in scored_docs]
        
        return ranked_docs, sorted_scores
    
    def _determine_confidence(self, score: float) -> str:
        if score < config.RERANKER_SCORE_THRESHOLD:
            return "none"
        elif score < 0:
            return "low"
        elif score < 2:
            return "medium"
        else:
            return "high"
    
    def _assemble_context(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        context_parts = []
        sources = []
        
        for doc, score in zip(documents, scores):
            fname = doc.metadata.get("file_name", "unknown")
            path = doc.metadata.get("relative_path", "unknown")
            lang = doc.metadata.get("language", "unknown")
            
            context_parts.append(
                f"\n--- FILE: {fname} ({path}) | Language: {lang} ---\n"
                f"{doc.page_content}\n"
            )
            
            sources.append({
                "file": fname,
                "path": path,
                "language": lang,
                "relevance_score": float(score),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        context_text = "\n".join(context_parts)
        return context_text, sources
    
    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are an expert Senior Developer Assistant with deep knowledge of software architecture and best practices.

Your task is to answer the developer's question using ONLY the code context provided below.

IMPORTANT RULES:
1. Base your answer ONLY on the provided code context
2. If the answer isn't in the context, say "I cannot answer this based on the provided code"
3. Cite specific file names when referencing code
4. Be concise but thorough
5. Use technical language appropriate for developers
6. If you see partial implementations, mention what's shown vs what might be missing
7. If code is incomplete, explicitly say what is missing instead of guessing

CODE CONTEXT:
{context}

DEVELOPER'S QUESTION: 
{query}

YOUR ANSWER:"""
        
        response = self.llm.invoke(prompt)
        answer = response if isinstance(response, str) else response.content
        return answer.strip()
    
    def _create_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: str
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "source_count": len(sources)
        }
    
    # Utility Methods
    
    def reset_database(self):
        logger.info("Resetting vector database...")
        
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            try:
                del self.vector_store._client
            except:
                pass
        
        if os.path.exists(config.VECTOR_DB_PATH):
            shutil.rmtree(config.VECTOR_DB_PATH)
        
        self._initialize_vector_store()
        logger.info("Database reset complete")
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_chunks": count,
                "vector_db_path": config.VECTOR_DB_PATH,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.OLLAMA_MODEL
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def debug_retrieval(self, query: str, k: int = 10):
        logger.info(f"\n{'='*60}")
        logger.info(f"DEBUG RETRIEVAL FOR: {query}")
        logger.info(f"{'='*60}")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        logger.info("\nVECTOR SEARCH SCORES (L2 Distance):")
        for i, (doc, score) in enumerate(results, 1):
            fname = doc.metadata.get('file_name', 'unknown')
            logger.info(f"  {i}. {score:.4f} | {fname}")
        
        docs = [doc for doc, _ in results]
        pairs = [[query, doc.page_content] for doc in docs]
        rerank_scores = self.reranker.predict(pairs)
        
        logger.info("\nRE-RANKER SCORES (Relevance):")
        scored = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
        for i, (doc, score) in enumerate(scored, 1):
            fname = doc.metadata.get('file_name', 'unknown')
            confidence = self._determine_confidence(score)
            logger.info(f"  {i}. {score:.4f} ({confidence}) | {fname}")
        
        logger.info(f"\n{'='*60}\n")