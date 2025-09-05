#!/usr/bin/env python3
"""
Collection Query Tool

A dedicated tool for querying ChromaDB collections created by the PDF processing pipeline.
Supports flexible querying with PDF filtering, HyDE preprocessing, and detailed result analysis.

Features:
- Query entire collections or specific PDFs
- HyDE (Hypothetical Document Embeddings) preprocessing for improved search
- Interactive query mode with advanced commands
- PDF comparison and overview capabilities
- Result export to JSON

Requirements:
- chromadb: pip install chromadb
- openai: pip install openai

Usage:
    # Basic querying
    python query_collection.py "machine learning" --collection research_papers
    
    # Query specific PDFs
    python query_collection.py "neural networks" --collection research_papers --pdfs paper1.pdf paper2.pdf
    
    # Enhanced search with HyDE preprocessing
    python query_collection.py "how do transformers work?" --collection research_papers --hyde
    
    # Get collection information
    python query_collection.py --info --collection research_papers
    
    # Interactive query mode
    python query_collection.py --interactive --collection research_papers
    
    # PDF comparison
    python query_collection.py --compare "attention mechanism" paper1.pdf paper2.pdf --collection research_papers
"""

import argparse
import json
import pickle
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys
from dataclasses import dataclass
from openai import OpenAI

# Import the vector store component
from vector_store import DocumentVectorStore

# Optional hybrid search imports
try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str  # 'dense', 'sparse', 'hybrid', 'reranked'


class CollectionQueryTool:
    def __init__(self, collection_name: str, vector_store_dir: str = "./chroma_db", 
                 gpt_token_path: str = "gpt_token.txt", enable_hyde: bool = False, 
                 enable_auto_toc: bool = False, enable_hybrid_search: bool = False):
        """
        Initialize the collection query tool.
        
        Args:
            collection_name: Name of the ChromaDB collection to query
            vector_store_dir: Directory containing the ChromaDB data
            gpt_token_path: Path to GPT token file for OpenAI embeddings
            enable_hyde: Whether to enable HyDE (Hypothetical Document Embeddings) preprocessing
            enable_auto_toc: Whether to enable automatic TOC section routing
            enable_hybrid_search: Whether to enable hybrid search (dense + sparse + reranking)
        """
        self.collection_name = collection_name
        self.vector_store_dir = Path(vector_store_dir)
        self.gpt_token_path = Path(gpt_token_path)
        self.enable_hyde = enable_hyde
        self.enable_auto_toc = enable_auto_toc
        self.enable_hybrid_search = enable_hybrid_search
        
        # Initialize hybrid search components
        self.bm25_index = None
        self.documents = []
        self.doc_metadata = []
        self.reranker = None
        
        # Initialize vector store
        try:
            self.vector_store = DocumentVectorStore(
                collection_name=collection_name,
                persist_directory=str(vector_store_dir),
                gpt_token_path=str(gpt_token_path)
            )
            print(f"Connected to collection: {collection_name}")
            if enable_hyde:
                print("HyDE preprocessing enabled")
            if enable_auto_toc:
                print("Automatic TOC routing enabled")
            if enable_hybrid_search:
                if HYBRID_SEARCH_AVAILABLE:
                    print("Hybrid search enabled - initializing BM25 and reranker...")
                    self._initialize_hybrid_search()
                else:
                    print("Warning: Hybrid search requested but dependencies not available.")
                    print("Install: pip install rank_bm25 sentence-transformers")
                    self.enable_hybrid_search = False
        except Exception as e:
            print(f"Error connecting to collection '{collection_name}': {e}")
            sys.exit(1)
        
        # Initialize OpenAI client for HyDE and auto-TOC if enabled
        self.openai_client = None
        if enable_hyde or enable_auto_toc:
            try:
                with open(gpt_token_path, 'r') as f:
                    api_key = f.readline().strip()
                self.openai_client = OpenAI(api_key=api_key)
                features = []
                if enable_hyde:
                    features.append("HyDE preprocessing")
                if enable_auto_toc:
                    features.append("automatic TOC routing")
                print(f"OpenAI client initialized for {', '.join(features)}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                if enable_hyde:
                    print("HyDE preprocessing will be disabled")
                    self.enable_hyde = False
                if enable_auto_toc:
                    print("Automatic TOC routing will be disabled")
                    self.enable_auto_toc = False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the collection."""
        try:
            stats = self.vector_store.get_collection_stats()
            available_pdfs = self.vector_store.get_available_pdfs()
            toc_info = self.get_toc_sections()
            
            return {
                "collection_name": self.collection_name,
                "storage_location": str(self.vector_store_dir),
                "statistics": stats,
                "available_pdfs": available_pdfs,
                "toc_sections": toc_info
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def get_toc_sections(self) -> Dict[str, List[Dict]]:
        """Get table of contents sections for all PDFs in the collection."""
        try:
            # Get all chunks with TOC information
            all_results = self.vector_store.collection.get(include=["metadatas"])
            
            toc_sections = {}
            
            for metadata in all_results["metadatas"]:
                pdf_filename = metadata.get("pdf_filename", "unknown")
                # TOC info is now stored as individual fields
                toc_title = metadata.get("toc_title", "")
                corrected_toc_page = metadata.get("corrected_toc_page", 0)
                
                if toc_title:
                    if pdf_filename not in toc_sections:
                        toc_sections[pdf_filename] = {}
                    
                    # Store as dict with title and page info, avoid duplicates
                    toc_key = toc_title.strip()
                    if toc_key not in toc_sections[pdf_filename]:
                        toc_sections[pdf_filename][toc_key] = {
                            "title": toc_title,
                            "page": corrected_toc_page or 0
                        }
            
            # Convert dicts to sorted lists (by page number)
            for pdf in toc_sections:
                sections_list = list(toc_sections[pdf].values())
                # Sort by page number, then by title
                sections_list.sort(key=lambda x: (x["page"], x["title"]))
                toc_sections[pdf] = sections_list
            
            return toc_sections
            
        except Exception as e:
            print(f"Error getting TOC sections: {e}")
            return {}
    
    def _initialize_hybrid_search(self):
        """Initialize BM25 index and reranker for hybrid search."""
        if not HYBRID_SEARCH_AVAILABLE:
            return
        
        cache_path = self.vector_store_dir / f"{self.collection_name}_bm25.pkl"
        
        # Try to load cached BM25 index
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25_index = cache_data['bm25_index']
                    self.documents = cache_data['documents']
                    self.doc_metadata = cache_data['doc_metadata']
                print(f"Loaded BM25 index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Failed to load cached BM25 index: {e}")
                self._build_bm25_index()
        else:
            self._build_bm25_index()
            
        # Initialize reranker
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            print("Loaded cross-encoder reranker")
        except Exception as e:
            print(f"Failed to load reranker: {e}")
    
    def _build_bm25_index(self):
        """Build BM25 index from ChromaDB collection."""
        if not HYBRID_SEARCH_AVAILABLE:
            return
            
        print("Building BM25 index from collection...")
        results = self.vector_store.collection.get(include=["documents", "metadatas"])
        
        if not results["documents"]:
            print("No documents found in collection")
            return
        
        self.documents = []
        self.doc_metadata = []
        
        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            # Tokenize document for BM25
            tokens = re.findall(r'\b\w+\b', doc.lower())
            self.documents.append(tokens)
            
            # Store metadata with index
            metadata_with_idx = metadata.copy()
            metadata_with_idx['_chroma_idx'] = i
            metadata_with_idx['_content'] = doc
            self.doc_metadata.append(metadata_with_idx)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(self.documents)
        
        # Cache the index
        cache_path = self.vector_store_dir / f"{self.collection_name}_bm25.pkl"
        try:
            cache_data = {
                'bm25_index': self.bm25_index,
                'documents': self.documents, 
                'doc_metadata': self.doc_metadata
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached BM25 index to {cache_path}")
        except Exception as e:
            print(f"Failed to cache BM25 index: {e}")
            
        print(f"Built BM25 index with {len(self.documents)} documents")
    
    def _sparse_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform sparse retrieval using BM25."""
        if not self.bm25_index or not HYBRID_SEARCH_AVAILABLE:
            return []
        
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                metadata = self.doc_metadata[idx]
                result = SearchResult(
                    chunk_id=metadata.get("chunk_id", f"chunk_{idx}"),
                    content=metadata.get("_content", ""),
                    metadata=metadata,
                    score=float(bm25_scores[idx]),
                    retrieval_method="sparse"
                )
                results.append(result)
        
        return results
    
    def _dense_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform dense retrieval using embeddings."""
        try:
            # Generate query embedding using OpenAI (same as stored embeddings)
            query_embedding = self.vector_store._get_openai_embeddings([query])[0]
            
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.vector_store.collection.count())
            )
            
            search_results = []
            for i in range(len(results["ids"][0])):
                result = SearchResult(
                    chunk_id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                    retrieval_method="dense"
                )
                search_results.append(result)
            
            return search_results
        except Exception as e:
            print(f"Dense search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, dense_results: List[SearchResult], 
                               sparse_results: List[SearchResult], k: int = 60) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        # Create score maps
        dense_scores = {result.chunk_id: 1.0 / (k + rank + 1) for rank, result in enumerate(dense_results)}
        sparse_scores = {result.chunk_id: 1.0 / (k + rank + 1) for rank, result in enumerate(sparse_results)}
        
        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Create combined results
        combined_results = []
        result_lookup = {result.chunk_id: result for result in dense_results + sparse_results}
        
        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            
            # Weighted combination (70% dense, 30% sparse)
            combined_score = 0.7 * dense_score + 0.3 * sparse_score
            
            result = result_lookup[chunk_id]
            hybrid_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata,
                score=combined_score,
                retrieval_method="hybrid"
            )
            combined_results.append(hybrid_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results
    
    def _rerank_results(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = [[query, result.content] for result in results]
            
            # Get reranker scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update results with reranker scores
            reranked_results = []
            for result, score in zip(results, rerank_scores):
                reranked_result = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    score=float(score),
                    retrieval_method="reranked"
                )
                reranked_results.append(reranked_result)
            
            # Sort by reranker scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """Perform exact keyword search using vector store's keyword functionality."""
        try:
            # Extract keywords for search
            keywords = self.vector_store.extract_keywords_from_query(query)
            print(f"🔍 Extracted keywords for search: {keywords}")
            
            if not keywords:
                print("⚠️  No keywords extracted from query")
                return []
            
            # Use vector store's keyword search
            keyword_results = self.vector_store.search_exact_keywords(keywords, chunk_type="child")
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(keyword_results[:top_k]):
                search_result = SearchResult(
                    chunk_id=result.get("id", ""),
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                    score=result.get("keyword_score", 0.0),
                    retrieval_method="keyword"
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []
    
    def _three_way_reciprocal_rank_fusion(self, dense_results: List[SearchResult], 
                                         sparse_results: List[SearchResult], 
                                         keyword_results: List[SearchResult], k: int = 60) -> List[SearchResult]:
        """Combine three result sets using Reciprocal Rank Fusion."""
        # Create score maps for each method
        dense_scores = {result.chunk_id: 1.0 / (k + rank + 1) for rank, result in enumerate(dense_results)}
        sparse_scores = {result.chunk_id: 1.0 / (k + rank + 1) for rank, result in enumerate(sparse_results)}
        keyword_scores = {result.chunk_id: 1.0 / (k + rank + 1) for rank, result in enumerate(keyword_results)}
        
        # Collect all unique chunk IDs
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys()) | set(keyword_scores.keys())
        
        # Calculate combined scores
        combined_scores = {}
        result_map = {}
        
        # Build result map from all sources
        for result in dense_results + sparse_results + keyword_results:
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
        
        # Calculate RRF scores
        for chunk_id in all_chunk_ids:
            combined_score = (
                dense_scores.get(chunk_id, 0) + 
                sparse_scores.get(chunk_id, 0) + 
                keyword_scores.get(chunk_id, 0)
            )
            combined_scores[chunk_id] = combined_score
        
        # Sort by combined score and return results
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        combined_results = []
        for chunk_id in sorted_chunk_ids:
            if chunk_id in result_map:
                result = result_map[chunk_id]
                result.score = combined_scores[chunk_id]
                result.retrieval_method = "rrf_combined"
                combined_results.append(result)
        
        return combined_results
    
    def hybrid_search(self, query: str, top_k: int = 10, retrieval_k: int = 50) -> List[SearchResult]:
        """
        Perform hybrid search combining dense, sparse, keyword, and reranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            retrieval_k: Number of results to retrieve from each method
            
        Returns:
            List of SearchResult objects
        """
        if not self.enable_hybrid_search:
            # Fallback to dense search only
            return self._dense_search(query, top_k)
        
        print(f"Performing hybrid search for: '{query}'")
        
        # Step 1: Dense retrieval
        dense_results = self._dense_search(query, retrieval_k)
        print(f"Dense retrieval: {len(dense_results)} results")
        
        # Step 2: Sparse retrieval  
        sparse_results = self._sparse_search(query, retrieval_k)
        print(f"Sparse retrieval: {len(sparse_results)} results")
        
        # Step 3: Keyword retrieval
        keyword_results = self._keyword_search(query, retrieval_k)
        print(f"Keyword retrieval: {len(keyword_results)} results")
        
        # Step 4: Three-way reciprocal rank fusion
        combined_results = self._three_way_reciprocal_rank_fusion(dense_results, sparse_results, keyword_results)
        print(f"RRF combination: {len(combined_results)} results")
        
        # Step 5: Reranking
        final_results = self._rerank_results(query, combined_results[:20], top_k)
        print(f"Final results after reranking: {len(final_results)}")
        
        return final_results
    
    def generate_hyde_passage(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical document passage using HyDE technique.
        
        Args:
            query: The user's search query
            
        Returns:
            Generated hypothetical passage or None if generation fails
        """
        if not self.enable_hyde or not self.openai_client:
            return None
        
        hyde_system_prompt = """You are a helpful assistant that writes concise technical passages. Your job is to generate a short, realistic passage that could appear in a document relevant to a user's query. The passage should look like part of an academic paper, report, or manual, depending on the query. Do not mention the query itself. Do not speculate about document titles or authors. Just write plausible content."""
        
        hyde_user_prompt = f"""Write a short passage (≤120 words) that would plausibly appear in a document answering the following question:
"{query}"

The passage should directly contain the kind of information such a document would provide, expressed in neutral, factual style."""
        
        try:
            print(f"🔄 Generating HyDE passage for improved search...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": hyde_system_prompt},
                    {"role": "user", "content": hyde_user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            hyde_passage = response.choices[0].message.content.strip()
            print(f"📝 Generated passage ({len(hyde_passage)} chars): {hyde_passage[:100]}...")
            print("-" * 60)
            return hyde_passage
            
        except Exception as e:
            print(f"Warning: HyDE generation failed: {e}")
            print("Falling back to original query")
            return None
    
    def suggest_relevant_toc_sections(self, query: str, pdf_filenames: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Use LLM to suggest which TOC sections are most relevant for a query.
        
        Args:
            query: The user's search query
            pdf_filenames: Optional list of PDF filenames to restrict suggestions
            
        Returns:
            Dictionary mapping PDF filenames to lists of relevant section names
        """
        if not self.enable_auto_toc or not self.openai_client:
            return {}
        
        try:
            # Get available TOC sections
            all_toc_sections = self.get_toc_sections()
            
            # Filter by PDF if specified
            if pdf_filenames:
                filtered_toc = {pdf: sections for pdf, sections in all_toc_sections.items() 
                               if pdf in pdf_filenames}
            else:
                filtered_toc = all_toc_sections
            
            if not filtered_toc:
                return {}
            
            # Prepare section lists for LLM analysis
            section_context = ""
            for pdf, sections in filtered_toc.items():
                if sections:
                    section_context += f"\n{pdf}:\n"
                    for i, section in enumerate(sections, 1):
                        if not isinstance(section, dict):
                            raise TypeError(f"Expected section to be dict, got {type(section)}: {section}")
                        if "title" not in section:
                            raise KeyError(f"Section dict missing 'title' key. Keys: {section.keys()}")
                        section_title = section["title"]
                        section_context += f"  {i}. {section_title}\n"
            
            if not section_context.strip():
                return {}
            
            system_prompt = """You are an expert at analyzing research questions and determining which sections of academic papers would be most relevant to answer them. Given a user's query and a list of table of contents sections from various documents, identify the 2-3 most relevant sections that would likely contain information to answer the query.

If the query is broad, general, or doesn't clearly relate to specific sections, you can suggest searching the "entire pdf" instead.

Return your response as a JSON object where keys are PDF filenames and values are either:
1. Arrays of the most relevant section names from that document, OR
2. The string "entire pdf" if the query is too general or doesn't clearly match specific sections

Only include sections that are highly relevant - it's better to suggest fewer sections or "entire pdf" than to include irrelevant ones."""
            
            user_prompt = f"""Query: "{query}"

Available sections from documents:
{section_context}

Which sections are most likely to contain relevant information for this query? Return as JSON format:
{{
  "document.pdf": ["relevant_section1", "relevant_section2"],
  "another.pdf": ["relevant_section"],
  "broad_document.pdf": "entire pdf"
}}

Use "entire pdf" for documents where the query is too general or doesn't clearly match specific sections."""
            
            print(f"🤖 Analyzing query for relevant TOC sections...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.3
            )
            
            suggested_sections = json.loads(response.choices[0].message.content)
            
            # Validate and clean suggestions
            validated_suggestions = {}
            for pdf, sections in suggested_sections.items():
                if pdf in filtered_toc:
                    # Handle "entire pdf" option
                    if isinstance(sections, str) and sections.lower() == "entire pdf":
                        validated_suggestions[pdf] = "entire pdf"
                    elif isinstance(sections, list):
                        valid_sections = []
                        for section in sections:
                            # Find matching sections (case-insensitive partial match)
                            for available_section in filtered_toc[pdf]:
                                if not isinstance(available_section, dict):
                                    raise TypeError(f"Expected available_section to be dict, got {type(available_section)}: {available_section}")
                                if "title" not in available_section:
                                    raise KeyError(f"Available section dict missing 'title' key. Keys: {available_section.keys()}")
                                
                                available_section_title = available_section["title"]
                                if (section.lower() in available_section_title.lower() or 
                                    available_section_title.lower() in section.lower()):
                                    if available_section_title not in valid_sections:
                                        valid_sections.append(available_section_title)
                                    break
                        if valid_sections:
                            validated_suggestions[pdf] = valid_sections
            
            if validated_suggestions:
                print(f"🎯 Suggested relevant sections:")
                for pdf, sections in validated_suggestions.items():
                    if isinstance(sections, str) and sections == "entire pdf":
                        print(f"  {pdf}: {sections}")
                    else:
                        print(f"  {pdf}: {', '.join(sections)}")
                print("-" * 60)
            
            return validated_suggestions
            
        except Exception as e:
            print(f"Warning: Auto-TOC routing failed: {e}")
            return {}
    
    def _query_auto_toc_sections(self, query: str, suggested_sections: Dict[str, List[str]], 
                                n_results: int, chunk_type: str, use_hyde: bool) -> Dict[str, Any]:
        """
        Query multiple TOC sections and combine results with global fallback.
        
        Args:
            query: Original search query
            suggested_sections: Dictionary mapping PDF names to relevant section names
            n_results: Total number of results to return
            chunk_type: Type of chunks to search
            use_hyde: Whether to use HyDE preprocessing
            
        Returns:
            Combined query results from all relevant sections with global fallback if needed
        """
        all_results = []
        section_info = []
        
        # Collect all sections and PDFs for batch processing
        all_sections = []
        pdf_list = []
        entire_pdf_list = []  # PDFs to search entirely
        
        for pdf, sections in suggested_sections.items():
            if isinstance(sections, str) and sections == "entire pdf":
                # This PDF should be searched entirely without section filtering
                entire_pdf_list.append(pdf)
                section_info.append({
                    "pdf": pdf,
                    "section": "entire pdf",
                    "results_found": 0
                })
            elif isinstance(sections, list):
                for section in sections:
                    all_sections.append(section)
                    pdf_list.append(pdf)
                    section_info.append({
                        "pdf": pdf,
                        "section": section,
                        "results_found": 0  # Will be updated after search
                    })
        
        if not all_sections and not entire_pdf_list:
            return {"results": [], "total_found": 0}
        
        # Prepare search query (with HyDE if needed)
        search_query = query
        if use_hyde:
            hyde_passage = self.generate_hyde_passage(query)
            if hyde_passage:
                search_query = hyde_passage
        
        try:
            # Search TOC sections if any are specified
            section_results = []
            if all_sections:
                section_search_results = self.vector_store.query_hybrid_search(
                    query=search_query,
                    n_results=n_results,  # Adjust based on split between section/entire PDF results
                    chunk_type=chunk_type,
                    pdf_filenames=list(set(pdf_list)),  # Unique PDFs with section filtering
                    toc_sections=all_sections,
                    use_keywords=True
                )
                section_results = section_search_results.get("results", [])
            
            # Search entire PDFs if any are specified
            entire_pdf_results = []
            if entire_pdf_list:
                entire_pdf_search_results = self.vector_store.query_hybrid_search(
                    query=search_query,
                    n_results=n_results,  # Adjust based on split between section/entire PDF results
                    chunk_type=chunk_type,
                    pdf_filenames=entire_pdf_list,  # PDFs to search entirely
                    toc_sections=None,  # No section filtering
                    use_keywords=True
                )
                entire_pdf_results = entire_pdf_search_results.get("results", [])
            
            # Combine results from both searches
            combined_results = section_results + entire_pdf_results
            
            # Add section context to each result and map back to specific sections
            for result in combined_results[:n_results]:
                result_pdf = result.get("metadata", {}).get("pdf_filename", "")
                result_toc = result.get("metadata", {}).get("toc_title", "")
                matched_section = None
                source_type = "toc_section"
                
                # Check if this result comes from an "entire pdf" search
                if result_pdf in entire_pdf_list:
                    matched_section = "entire pdf"
                    source_type = "entire_pdf"
                else:
                    # Find the matching section from our suggestions for section-specific searches
                    for pdf, sections in suggested_sections.items():
                        if pdf == result_pdf and isinstance(sections, list):
                            for section in sections:
                                if section.lower() in result_toc.lower() or result_toc.lower() in section.lower():
                                    matched_section = section
                                    break
                        if matched_section:
                            break
                
                result["auto_toc_section"] = matched_section or result_toc
                result["auto_toc_pdf"] = result_pdf
                result["source_type"] = source_type
                result["used_hyde"] = use_hyde and search_query != query
                all_results.append(result)
                
                # Update section info count
                for info in section_info:
                    if info["pdf"] == result_pdf and info["section"] == (matched_section or "entire pdf"):
                        info["results_found"] += 1
                        break
                        
        except Exception as e:
            print(f"Warning: Error in batch TOC search: {e}")
            return {"results": [], "total_found": 0}
        
        # Sort all results by similarity and limit
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        final_results = all_results[:n_results]
        
        # Check if we have enough results, if not, supplement with global search
        min_results_threshold = max(3, n_results // 2)  # At least 3 results or half of requested
        
        if len(final_results) < min_results_threshold:
            print(f"⚠️  TOC-specific search returned only {len(final_results)} results, supplementing with global search...")
            
            # Get PDF filenames from suggested sections
            pdf_filenames = list(suggested_sections.keys()) if suggested_sections else None
            
            # Perform global search for remaining results
            remaining_needed = n_results - len(final_results)
            
            try:
                global_results = self.vector_store.query_hybrid_search(
                    query=query,
                    n_results=remaining_needed * 2,  # Get more to avoid duplicates
                    chunk_type=chunk_type,
                    pdf_filenames=pdf_filenames,
                    use_keywords=True
                )
                
                # Filter out results we already have (by chunk ID)
                existing_ids = {result.get("id") for result in final_results}
                
                for result in global_results.get("results", []):
                    if result.get("id") not in existing_ids and len(final_results) < n_results:
                        result["source_type"] = "global_fallback"
                        final_results.append(result)
                
                # Re-sort combined results
                final_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                
                global_added = len([r for r in final_results if r.get("source_type") == "global_fallback"])
                if global_added > 0:
                    print(f"📄 Added {global_added} results from global search (total: {len(final_results)})")
                
            except Exception as e:
                print(f"Warning: Global fallback search failed: {e}")
        
        # Compile final response
        return {
            "query": query,
            "used_auto_toc": True,
            "used_global_fallback": any(r.get("source_type") == "global_fallback" for r in final_results),
            "suggested_sections": suggested_sections,
            "section_search_info": section_info,
            "results": final_results,
            "total_found": len(final_results),
            "total_sections_searched": len(all_sections) + len(entire_pdf_list)
        }
    
    def query_by_toc_section(self, query: str, toc_sections: List[str], pdf_filenames: Optional[List[str]] = None,
                            n_results: int = 10, chunk_type: str = "child", use_hyde: bool = None) -> Dict[str, Any]:
        """
        Query chunks that belong to specific table of contents sections.
        
        Args:
            query: Search query text
            toc_sections: List of TOC section titles to search within
            pdf_filenames: Optional list of PDF filenames to restrict search
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            use_hyde: Whether to use HyDE for this query (overrides global setting)
            
        Returns:
            Query results with metadata
        """
        # Determine if HyDE should be used for this query
        should_use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        
        # Generate HyDE passage if enabled
        search_query = query
        if should_use_hyde:
            hyde_passage = self.generate_hyde_passage(query)
            if hyde_passage:
                search_query = hyde_passage
        
        print(f"Searching TOC sections: {', '.join(toc_sections)}")
        print(f"Query: '{query}'")
        print(f"Collection: {self.collection_name}")
        if pdf_filenames:
            print(f"PDFs: {', '.join(pdf_filenames)}")
        print(f"Chunk type: {chunk_type}")
        if should_use_hyde and search_query != query:
            print(f"HyDE preprocessing: ✓")
        print("-" * 60)
        
        try:
            # Get all chunks from the collection with metadata
            all_results = self.vector_store.collection.get(include=["metadatas", "documents"])
            
            # Filter chunks that match any of the TOC sections
            matching_ids = []
            for i, metadata in enumerate(all_results["metadatas"]):
                # Check chunk type
                if metadata.get("chunk_type") != chunk_type:
                    continue
                
                # Check PDF filter if specified
                if pdf_filenames and metadata.get("pdf_filename") not in pdf_filenames:
                    continue
                
                # Check TOC section match against any of the provided sections
                toc_title = metadata.get("toc_title", "")
                if toc_title:
                    for toc_section in toc_sections:
                        # Case-insensitive partial match
                        if toc_section.lower() in toc_title.lower() or toc_title.lower() in toc_section.lower():
                            matching_ids.append(all_results["ids"][i])
                            break  # Found match, no need to check other sections
            
            if not matching_ids:
                print(f"No chunks found in TOC sections: {', '.join(toc_sections)}")
                return {"results": [], "total_found": 0, "toc_sections": toc_sections}
            
            print(f"Found {len(matching_ids)} chunks in sections: {', '.join(toc_sections)}")
            
            # Use the updated vector store method to search within the TOC sections
            search_results = self.vector_store.query_hybrid_search(
                query=search_query,
                n_results=n_results,
                chunk_type=chunk_type,
                pdf_filenames=pdf_filenames,
                toc_sections=toc_sections,
                use_keywords=True
            )
            
            filtered_results = search_results.get("results", [])
            print(f"Debug: Retrieved {len(filtered_results)} chunks using vector store TOC filtering")
            
            results = {
                "query": query,
                "search_query": search_query,
                "used_hyde": should_use_hyde and search_query != query,
                "toc_sections": toc_sections,
                "results": filtered_results,
                "total_found": len(filtered_results),
                "filtered_by_pdfs": pdf_filenames if pdf_filenames else None
            }
            
            self._display_results(results, show_pdf_names=True)
            return results
            
        except Exception as e:
            print(f"Error querying TOC sections: {e}")
            return {"results": [], "total_found": 0, "toc_sections": toc_sections}
    
    def query_entire_collection(self, query: str, n_results: int = 10, 
                               chunk_type: str = "child", use_hyde: bool = None, 
                               use_auto_toc: bool = None) -> Dict[str, Any]:
        """
        Query the entire collection without PDF filtering.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            use_hyde: Whether to use HyDE for this query (overrides global setting)
            use_auto_toc: Whether to use auto-TOC routing (overrides global setting)
            
        Returns:
            Query results with metadata
        """
        # Determine settings
        should_use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        should_use_auto_toc = use_auto_toc if use_auto_toc is not None else self.enable_auto_toc
        
        # Try auto-TOC routing first if enabled
        if should_use_auto_toc:
            suggested_sections = self.suggest_relevant_toc_sections(query)
            if suggested_sections:
                print(f"🔍 Using auto-TOC routing - searching relevant sections...")
                toc_results = self._query_auto_toc_sections(query, suggested_sections, n_results, chunk_type, should_use_hyde)
                # If TOC routing was successful (with or without fallback), return results
                if toc_results.get("results"):
                    return toc_results
        
        # Fallback to regular search
        # Generate HyDE passage if enabled
        search_query = query
        if should_use_hyde:
            hyde_passage = self.generate_hyde_passage(query)
            if hyde_passage:
                search_query = hyde_passage
        
        print(f"Searching entire collection for: '{query}'")
        print(f"Collection: {self.collection_name}")
        print(f"Chunk type: {chunk_type}")
        if should_use_hyde and search_query != query:
            print(f"HyDE preprocessing: ✓")
        if should_use_auto_toc:
            print(f"Auto-TOC routing: No relevant sections found, using full collection search")
        print("-" * 60)
        
        try:
            if self.enable_hybrid_search:
                # Use advanced hybrid search (dense + sparse + keyword + reranking)
                hybrid_results = self.hybrid_search(search_query, top_k=n_results)
                
                # Convert to expected format
                results = {
                    "results": [
                        {
                            "id": result.chunk_id,
                            "content": result.content,
                            "metadata": result.metadata,
                            "similarity_score": result.score,
                            "retrieval_method": result.retrieval_method
                        }
                        for result in hybrid_results
                    ],
                    "total_found": len(hybrid_results)
                }
            else:
                # Fall back to traditional hybrid search
                results = self.vector_store.query_hybrid_search(
                    query=search_query,
                    n_results=n_results,
                    chunk_type=chunk_type,
                    pdf_filenames=None,  # No filtering
                    use_keywords=True
                )
            
            # Add metadata
            results["original_query"] = query
            results["search_query"] = search_query
            results["used_hyde"] = should_use_hyde and search_query != query
            results["used_auto_toc"] = False  # Fallback case
            
            self._display_results(results, show_pdf_names=True)
            return results
            
        except Exception as e:
            print(f"Error querying collection: {e}")
            return {"results": [], "total_found": 0}
    
    def query_specific_pdfs(self, query: str, pdf_filenames: List[str], 
                           n_results: int = 10, chunk_type: str = "child", 
                           use_hyde: bool = None, use_auto_toc: bool = None) -> Dict[str, Any]:
        """
        Query only specific PDFs within the collection.
        
        Args:
            query: Search query text
            pdf_filenames: List of PDF filenames to search within
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            use_hyde: Whether to use HyDE for this query (overrides global setting)
            use_auto_toc: Whether to use auto-TOC routing (overrides global setting)
            
        Returns:
            Query results with metadata
        """
        # Determine settings
        should_use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        should_use_auto_toc = use_auto_toc if use_auto_toc is not None else self.enable_auto_toc
        
        # Validate PDF filenames
        available_pdfs = self.vector_store.get_available_pdfs()
        invalid_pdfs = [pdf for pdf in pdf_filenames if pdf not in available_pdfs]
        
        if invalid_pdfs:
            print(f"Warning: The following PDFs are not in the collection: {invalid_pdfs}")
            valid_pdfs = [pdf for pdf in pdf_filenames if pdf in available_pdfs]
            if not valid_pdfs:
                print("No valid PDFs to search. Aborting query.")
                return {"results": [], "total_found": 0}
            pdf_filenames = valid_pdfs
        
        # Try auto-TOC routing first if enabled
        if should_use_auto_toc:
            suggested_sections = self.suggest_relevant_toc_sections(query, pdf_filenames)
            if suggested_sections:
                print(f"🔍 Using auto-TOC routing for specific PDFs - searching relevant sections...")
                toc_results = self._query_auto_toc_sections(query, suggested_sections, n_results, chunk_type, should_use_hyde)
                # If TOC routing was successful (with or without fallback), return results
                if toc_results.get("results"):
                    return toc_results
        
        # Fallback to regular search
        # Generate HyDE passage if enabled
        search_query = query
        if should_use_hyde:
            hyde_passage = self.generate_hyde_passage(query)
            if hyde_passage:
                search_query = hyde_passage
        
        print(f"Searching specific PDFs for: '{query}'")
        print(f"Collection: {self.collection_name}")
        print(f"PDFs: {', '.join(pdf_filenames)}")
        print(f"Chunk type: {chunk_type}")
        if should_use_hyde and search_query != query:
            print(f"HyDE preprocessing: ✓")
        if should_use_auto_toc:
            print(f"Auto-TOC routing: No relevant sections found, using regular search")
        print("-" * 60)
        
        try:
            results = self.vector_store.query_hybrid_search(
                query=search_query,
                n_results=n_results,
                chunk_type=chunk_type,
                pdf_filenames=pdf_filenames,
                use_keywords=True
            )
            
            # Add metadata
            results["original_query"] = query
            results["search_query"] = search_query
            results["used_hyde"] = should_use_hyde and search_query != query
            results["used_auto_toc"] = False  # Fallback case
            
            self._display_results(results, show_pdf_names=True)
            return results
            
        except Exception as e:
            print(f"Error querying specific PDFs: {e}")
            return {"results": [], "total_found": 0}
        
    def query_by_page_range(self, query: str, pdf_filename: str, start_page: int, end_page: int,
                           n_results: int = 10, chunk_type: str = "child", use_hyde: bool = None, 
                           max_context_tokens: int = 8000) -> Dict[str, Any]:
        """
        Query chunks within a specific page range of a PDF.
        For small ranges, returns full page text directly. For larger ranges, uses RAG.
        
        Args:
            query: Search query text
            pdf_filename: Name of the PDF to search within
            start_page: Starting page number (inclusive)
            end_page: Ending page number (inclusive)
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            use_hyde: Whether to use HyDE for this query (overrides global setting if not None)
            max_context_tokens: Maximum tokens for context window
            
        Returns:
            Query results with metadata compatible with citation system
        """
        # First, check if we can retrieve full page text for small ranges
        full_pages_result = self._try_get_full_pages_from_files(pdf_filename, start_page, end_page, max_context_tokens)
        if full_pages_result and chunk_type == "parent":
            return full_pages_result
        
        # Fallback to regular RAG with optimized query
        # Determine settings
        should_use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        
        # Validate PDF filename
        available_pdfs = self.vector_store.get_available_pdfs()
        if pdf_filename not in available_pdfs:
            print(f"Warning: PDF '{pdf_filename}' is not in the collection")
            return {"results": [], "total_found": 0}
        
        try:
            print(f"🔍 Querying page range {start_page}-{end_page} in {pdf_filename}")
            
            # Use HyDE if enabled
            search_query = query
            if should_use_hyde and hasattr(self, 'hyde_generator') and self.hyde_generator:
                search_query = self.hyde_generator.generate_hypothetical_document(query)
                print(f"🧠 HyDE enhanced query generated")
            
            # Optimized ChromaDB query with proper where clause structure including chunk_type
            where_clause = {
                "$and": [
                    {"pdf_filename": {"$eq": pdf_filename}},
                    {"page_number": {"$gte": start_page}},
                    {"page_number": {"$lte": end_page}},
                    {"chunk_type": {"$eq": chunk_type}}
                ]
            }
            
            # Query using embeddings for semantic search
            query_embedding = self.vector_store._get_openai_embeddings([search_query])[0]
            
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results with proper citation-compatible structure
            processed_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Structure results for citation compatibility
                    result = {
                        "content": doc,
                        "metadata": {
                            "page_number": metadata.get("page_number", 0),
                            "pdf_filename": pdf_filename,  # Ensure this field exists
                            "pdf_name": pdf_filename,      # Also include pdf_name for citation compatibility
                            "toc_title": metadata.get("toc_title", ""),
                            "corrected_toc_page": metadata.get("corrected_toc_page", 0),
                            "chunk_id": metadata.get("chunk_id", ""),
                        },
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    }
                    processed_results.append(result)
            
            print(f"📄 Found {len(processed_results)} results in page range {start_page}-{end_page}")
            
            return {
                "results": processed_results,
                "total_found": len(processed_results),
                "query_metadata": {
                    "pdf_filename": pdf_filename,
                    "page_range": f"{start_page}-{end_page}",
                    "used_hyde": should_use_hyde,
                    "chunk_type": chunk_type,
                    "method": "rag"
                }
            }
            
        except Exception as e:
            print(f"Error querying page range {start_page}-{end_page}: {e}")
            return {"results": [], "total_found": 0}
    
    def _try_get_full_pages_from_files(self, pdf_filename: str, start_page: int, end_page: int, 
                                      max_context_tokens: int = 8000) -> Optional[Dict[str, Any]]:
        """
        Try to retrieve full page text from layout files for small page ranges.
        
        Args:
            pdf_filename: PDF filename
            start_page: Start page number
            end_page: End page number  
            max_context_tokens: Maximum context tokens allowed
            
        Returns:
            Full page results if successful and within token limit, None otherwise
        """
        page_count = end_page - start_page + 1
        
        # Only try for small page ranges (max 5 pages)
        if page_count > 10:
            return None
            
        # Estimate tokens (rough approximation: 1 page ≈ 500-1000 tokens)
        estimated_tokens = page_count * 750  # Conservative estimate
        if estimated_tokens > max_context_tokens:
            return None
            
        try:
            # Look for layout file in processed_output directory
            layout_file = Path("processed_output") / f"{pdf_filename.replace('.pdf', '')}_layout.json"
            
            if not layout_file.exists():
                print(f"Layout file not found: {layout_file}")
                return None
            
            # Load and parse the JSON layout file
            import json
            try:
                with open(layout_file, 'r', encoding='utf-8') as f:
                    layout_data = json.load(f)
            except Exception as e:
                print(f"Error loading layout file: {e}")
                return None
            
            # Collect full page texts for requested page range
            full_pages = []
            
            # Look through the pages array
            if 'pages' in layout_data:
                for page_data in layout_data['pages']:
                    page_num = page_data.get('page_number', 0)
                    
                    # Check if this page is in our requested range
                    if start_page <= page_num <= end_page:
                        # Extract full page text
                        if 'full_page_text' in page_data:
                            full_text = page_data['full_page_text']
                            if full_text and full_text.strip():
                                full_pages.append({
                                    "content": full_text,
                                    "metadata": {
                                        "page_number": page_num,
                                        "pdf_filename": pdf_filename,
                                        "pdf_name": pdf_filename,  # For citation compatibility
                                        "chunk_id": f"{pdf_filename}:page:{page_num}",
                                        "is_full_page": True
                                    },
                                    "similarity_score": 1.0,  # Perfect score for full pages
                                    "rank": page_num - start_page + 1
                                })
            
            if full_pages:
                print(f"📄 Retrieved full page text for {len(full_pages)} pages (pages {start_page}-{end_page})")
                return {
                    "results": full_pages,
                    "total_found": len(full_pages),
                    "query_metadata": {
                        "pdf_filename": pdf_filename,
                        "page_range": f"{start_page}-{end_page}",
                        "used_hyde": False,
                        "method": "full_page_text",
                        "pages_retrieved": len(full_pages)
                    }
                }
            
        except Exception as e:
            print(f"Error retrieving full page text: {e}")
            
        return None
    
    def get_pdf_overview(self, pdf_filename: str, n_chunks: int = 5) -> Dict[str, Any]:
        """
        Get an overview of a specific PDF's content.
        
        Args:
            pdf_filename: Name of the PDF to analyze
            n_chunks: Number of representative chunks to show
            
        Returns:
            Overview information about the PDF
        """
        print(f"PDF Overview: {pdf_filename}")
        print("-" * 60)
        
        try:
            # Get chunks from this PDF
            results = self.vector_store.query_by_pdf(
                pdf_filename=pdf_filename,
                chunk_type="child",
                n_results=n_chunks
            )
            
            if not results["results"]:
                print(f"No chunks found for PDF: {pdf_filename}")
                return results
            
            # Display overview
            print(f"Total chunks found: {results['total_found']}")
            print(f"Showing first {len(results['results'])} chunks:")
            
            for i, chunk in enumerate(results["results"], 1):
                print(f"\n{i}. {chunk['section_title']} (Page {chunk['page_number']})")
                print(f"   Content: {chunk['content'][:150]}...")
            
            return results
            
        except Exception as e:
            print(f"Error getting PDF overview: {e}")
            return {"results": [], "total_found": 0}
    
    def compare_pdfs(self, query: str, pdf_filenames: List[str], n_results_per_pdf: int = 3) -> Dict[str, Any]:
        """
        Compare how different PDFs address a specific query.
        
        Args:
            query: Search query text
            pdf_filenames: List of PDF filenames to compare
            n_results_per_pdf: Number of results to show per PDF
            
        Returns:
            Comparison results organized by PDF
        """
        print(f"PDF Comparison for query: '{query}'")
        print(f"Comparing PDFs: {', '.join(pdf_filenames)}")
        print("-" * 60)
        
        comparison_results = {}
        
        for pdf in pdf_filenames:
            print(f"\n📄 Results from {pdf}:")
            print("-" * 40)
            
            try:
                results = self.vector_store.query_hybrid_search(
                    query=query,
                    n_results=n_results_per_pdf,
                    chunk_type="child",
                    pdf_filenames=[pdf],
                    use_keywords=True
                )
                
                comparison_results[pdf] = results
                
                if results["results"]:
                    for i, result in enumerate(results["results"], 1):
                        print(f"{i}. {result['section_title']} (Page {result['page_number']}) - Score: {result['similarity_score']:.3f}")
                        print(f"   {result['content'][:200]}...")
                        print()
                else:
                    print(f"   No relevant results found in {pdf}")
                    
            except Exception as e:
                print(f"   Error querying {pdf}: {e}")
                comparison_results[pdf] = {"results": [], "total_found": 0}
        
        return comparison_results
    
    def interactive_mode(self) -> None:
        """Start an interactive query session."""
        print(f"\n🔍 Interactive Query Mode for Collection: {self.collection_name}")
        print("=" * 60)
        
        # Show collection info
        info = self.get_collection_info()
        if info:
            stats = info["statistics"]
            print(f"Collection Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  PDFs: {stats['pdf_count']}")
            print(f"  Available PDFs: {', '.join(info['available_pdfs'])}")
        
        print("\nCommands:")
        print("  query <text>                    - Search entire collection")
        print("  query <text> --pdfs <pdf1> <pdf2> - Search specific PDFs")
        print("  query <text> --hyde             - Force HyDE preprocessing")
        print("  query <text> --no-hyde          - Disable HyDE preprocessing")
        print("  info                           - Show collection info")
        print("  pdfs                           - List available PDFs")
        print("  overview <pdf>                 - Get PDF overview")
        print("  compare <text> <pdf1> <pdf2>   - Compare PDFs")
        print("  compare <text> <pdf1> <pdf2> --hyde - Compare with HyDE")
        print("  help                           - Show this help")
        print("  quit                           - Exit interactive mode")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == "quit" or command == "exit":
                    print("Goodbye!")
                    break
                
                elif command == "help":
                    print("\nCommands:")
                    print("  query <text>                    - Search entire collection")
                    print("  query <text> --pdfs <pdf1> <pdf2> - Search specific PDFs")
                    print("  info                           - Show collection info")
                    print("  pdfs                           - List available PDFs")
                    print("  overview <pdf>                 - Get PDF overview")
                    print("  compare <text> <pdf1> <pdf2>   - Compare PDFs")
                    print("  help                           - Show this help")
                    print("  quit                           - Exit interactive mode")
                
                elif command == "info":
                    self._display_collection_info()
                
                elif command == "pdfs":
                    available_pdfs = self.vector_store.get_available_pdfs()
                    print(f"\nAvailable PDFs ({len(available_pdfs)}):")
                    for i, pdf in enumerate(available_pdfs, 1):
                        print(f"  {i}. {pdf}")
                
                elif command == "query" and len(parts) > 1:
                    if "--pdfs" in parts:
                        pdfs_index = parts.index("--pdfs")
                        query_text = " ".join(parts[1:pdfs_index])
                        pdf_list = parts[pdfs_index + 1:]
                        self.query_specific_pdfs(query_text, pdf_list)
                    else:
                        query_text = " ".join(parts[1:])
                        self.query_entire_collection(query_text)
                
                elif command == "overview" and len(parts) > 1:
                    pdf_name = parts[1]
                    self.get_pdf_overview(pdf_name)
                
                elif command == "compare" and len(parts) >= 4:
                    query_text = parts[1]
                    pdf_list = parts[2:]
                    self.compare_pdfs(query_text, pdf_list)
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _display_results(self, results: Dict[str, Any], show_pdf_names: bool = True) -> None:
        """Display query results in a formatted way."""
        if not results["results"]:
            print("No results found.")
            return
        
        # Show HyDE info if available
        if results.get("used_hyde"):
            print(f"🔍 HyDE preprocessing was used for enhanced search")
            print(f"Original query: {results.get('original_query', 'N/A')}")
            print()
        
        # Show auto-TOC info if available
        if results.get("used_auto_toc"):
            print(f"🤖 Auto-TOC routing was used - searched relevant sections:")
            suggested = results.get("suggested_sections", {})
            for pdf, sections in suggested.items():
                if isinstance(sections, str) and sections == "entire pdf":
                    print(f"  {pdf}: {sections}")
                else:
                    print(f"  {pdf}: {', '.join(sections)}")
            
            # Show global fallback info if used
            if results.get("used_global_fallback"):
                global_count = sum(1 for r in results["results"] if r.get("source_type") == "global_fallback")
                print(f"📄 Global search supplemented {global_count} additional results")
            print()
        
        # Show hybrid search info if available
        if results.get("search_info"):
            search_info = results["search_info"]
            if search_info.get("used_keywords"):
                keyword_count = sum(1 for r in results["results"] if r.get("source_type") == "keyword_match")
                boosted_count = sum(1 for r in results["results"] if r.get("keyword_boost"))
                print(f"🔍 Hybrid search: {keyword_count} keyword matches, {boosted_count} semantic results boosted")
                if search_info.get("keywords"):
                    keywords_display = ", ".join(search_info["keywords"][:3])
                    if len(search_info["keywords"]) > 3:
                        keywords_display += "..."
                    print(f"🔑 Keywords used: {keywords_display}")
                print()
        
        print(f"Found {results['total_found']} results:")
        print()
        
        for i, result in enumerate(results["results"], 1):
            # Handle both metadata formats: promoted to top-level vs nested in 'metadata' field
            metadata = result.get('metadata', {})
            title = result.get('section_title', metadata.get('section_title', 'Untitled'))
            page = result.get('page_number', metadata.get('page_number', '?'))
            score = result.get('similarity_score', 0)
            content = result.get('content', '')[:200]
            
            # Add source indicator for different result types
            source_indicator = ""
            if result.get("source_type") == "global_fallback":
                source_indicator = " [Global]"
            elif result.get("source_type") == "entire_pdf":
                source_indicator = " [Entire PDF]"
            elif result.get("source_type") == "toc_section":
                source_indicator = f" [TOC: {result.get('auto_toc_section', '').split(':')[0]}]"
            elif result.get("source_type") == "keyword_match":
                source_indicator = " [Keyword]"
            elif result.get("keyword_boost"):
                source_indicator = " [Semantic+Keywords]"
            elif result.get("source_type") == "semantic":
                source_indicator = " [Semantic]"
            
            if show_pdf_names:
                pdf_name = result.get('pdf_filename', metadata.get('pdf_filename', 'Unknown'))
                print(f"{i}. {title} (Page {page}) - {pdf_name}{source_indicator}")
            else:
                print(f"{i}. {title} (Page {page}){source_indicator}")
            
            print(f"   Chunk ID {result.get('chunk_id')}")
            print(f"   Similarity: {score:.3f}")
            
            # Show matched keywords if available
            if result.get("matched_keywords"):
                keywords_str = ", ".join(result["matched_keywords"][:3])
                if len(result["matched_keywords"]) > 3:
                    keywords_str += "..."
                print(f"   Keywords: {keywords_str}")
            
            print(f"   Content: {content}...")
            
            # Show parent chunk info if available
            if 'chunk_id' in result:
                try:
                    parent = self.vector_store.get_parent_chunk(result['chunk_id'])
                    if parent:
                        token_count = parent['metadata'].get('token_count', 0)
                        print(f"   Parent chunk: {parent['chunk_id']} ({token_count} tokens)")
                except:
                    pass  # Ignore errors in parent lookup
            
            print()
    
    def _display_collection_info(self) -> None:
        """Display detailed collection information."""
        info = self.get_collection_info()
        if not info:
            return
        
        stats = info["statistics"]
        print(f"\n📊 Collection Information:")
        print(f"Collection Name: {info['collection_name']}")
        print(f"Storage Location: {info['storage_location']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Child Chunks: {stats['child_chunks']}")
        print(f"Parent Chunks: {stats['parent_chunks']}")
        print(f"Unique Pages: {stats['unique_pages']}")
        print(f"Unique Sections: {stats['unique_sections']}")
        print(f"PDF Count: {stats['pdf_count']}")
        
        if info["available_pdfs"]:
            print(f"\nAvailable PDFs:")
            for i, pdf in enumerate(info["available_pdfs"], 1):
                print(f"  {i}. {pdf}")
        
        if stats.get('chunks_per_pdf'):
            print(f"\nChunks per PDF:")
            for pdf, count in stats['chunks_per_pdf'].items():
                print(f"  {pdf}: {count} chunks")


def main():
    """Main function for the collection query tool."""
    parser = argparse.ArgumentParser(description="Query PDF processing pipeline collections")
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--collection", "-c", required=True, 
                       help="ChromaDB collection name to query")
    parser.add_argument("--vector-store-dir", default="./chroma_db",
                       help="Directory containing ChromaDB data")
    parser.add_argument("--gpt-token", default="gpt_token.txt",
                       help="Path to GPT token file for OpenAI embeddings")
    parser.add_argument("--pdfs", nargs="+", 
                       help="Specific PDF filenames to search (optional)")
    parser.add_argument("--n-results", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--chunk-type", choices=["child", "parent"], default="child",
                       help="Type of chunks to search")
    parser.add_argument("--info", action="store_true",
                       help="Show collection information only")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive query mode")
    parser.add_argument("--overview", 
                       help="Get overview of a specific PDF")
    parser.add_argument("--compare", nargs="+", metavar=("QUERY", "PDF"),
                       help="Compare how different PDFs address a query")
    parser.add_argument("--toc-section", 
                       help="Search within a specific table of contents section")
    parser.add_argument("--list-toc", action="store_true",
                       help="List all table of contents sections in the collection")
    parser.add_argument("--save-results", 
                       help="Save query results to JSON file")
    parser.add_argument("--hyde", action="store_true",
                       help="Enable HyDE (Hypothetical Document Embeddings) preprocessing")
    parser.add_argument("--no-hyde", action="store_true",
                       help="Explicitly disable HyDE preprocessing")
    parser.add_argument("--auto-toc", action="store_true",
                       help="Enable automatic TOC section routing")
    parser.add_argument("--no-auto-toc", action="store_true",
                       help="Explicitly disable automatic TOC routing")
    parser.add_argument("--hybrid", action="store_true",
                       help="Enable hybrid search (dense + sparse + reranking)")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Explicitly disable hybrid search")
    parser.add_argument("--test-hybrid", action="store_true",
                       help="Test hybrid search performance (requires query)")
    
    args = parser.parse_args()
    
    # Handle HyDE flag conflicts
    if args.hyde and args.no_hyde:
        print("Error: Cannot specify both --hyde and --no-hyde")
        return 1
    
    # Handle auto-TOC flag conflicts
    if args.auto_toc and args.no_auto_toc:
        print("Error: Cannot specify both --auto-toc and --no-auto-toc")
        return 1
    
    # Handle hybrid search flag conflicts
    if args.hybrid and args.no_hybrid:
        print("Error: Cannot specify both --hybrid and --no-hybrid")
        return 1
    
    # Set preferences
    if args.no_hyde:
        args.hyde = False
    if args.no_auto_toc:
        args.auto_toc = False
    if args.no_hybrid:
        args.hybrid = False
    
    try:
        # Initialize the query tool
        query_tool = CollectionQueryTool(
            collection_name=args.collection,
            vector_store_dir=args.vector_store_dir,
            gpt_token_path=args.gpt_token,
            enable_hyde=args.hyde,
            enable_auto_toc=args.auto_toc,
            enable_hybrid_search=args.hybrid
        )
        
        # Handle different modes
        if args.interactive:
            query_tool.interactive_mode()
        
        elif args.info:
            query_tool._display_collection_info()
        
        elif args.overview:
            query_tool.get_pdf_overview(args.overview)
        
        elif args.compare and len(args.compare) >= 2:
            query_text = args.compare[0]
            pdf_list = args.compare[1:]
            results = query_tool.compare_pdfs(query_text, pdf_list)
            
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nComparison results saved to: {args.save_results}")
        
        elif args.test_hybrid and args.query:
            if not HYBRID_SEARCH_AVAILABLE:
                print("Error: Hybrid search dependencies not available.")
                print("Install: pip install rank_bm25 sentence-transformers")
                return 1
            
            print("🔬 Testing Hybrid Search Performance")
            print("=" * 60)
            
            # Test with hybrid search enabled
            print("\n🔍 Hybrid Search Results:")
            query_tool.enable_hybrid_search = True
            if not query_tool.bm25_index:
                query_tool._initialize_hybrid_search()
            
            hybrid_results = query_tool.hybrid_search(args.query, args.n_results)
            for i, result in enumerate(hybrid_results, 1):
                print(f"{i}. [{result.retrieval_method.upper()}] Score: {result.score:.4f}")
                print(f"   Content: {result.content[:150]}...")
                if result.metadata.get("toc_title"):
                    print(f"   TOC: {result.metadata['toc_title']}")
                print()
            
            # Test with dense search only
            print("\n🧠 Dense Search Only (baseline):")
            dense_only = query_tool._dense_search(args.query, args.n_results)
            for i, result in enumerate(dense_only, 1):
                print(f"{i}. [DENSE] Score: {result.score:.4f}")
                print(f"   Content: {result.content[:150]}...")
                if result.metadata.get("toc_title"):
                    print(f"   TOC: {result.metadata['toc_title']}")
                print()
        
        elif args.list_toc:
            toc_sections = query_tool.get_toc_sections()
            if toc_sections:
                print("\nTable of Contents Sections:")
                print("=" * 50)
                for pdf, sections in toc_sections.items():
                    print(f"\n📄 {pdf}:")
                    if sections:
                        for i, section in enumerate(sections, 1):
                            print(f"  {i}. {section}")
                    else:
                        print("  No TOC sections detected")
            else:
                print("No table of contents information found in collection.")
        
        elif args.toc_section and args.query:
            # Search within a specific TOC section
            results = query_tool.query_by_toc_section(
                query=args.query,
                toc_sections=[args.toc_section],  # Convert single section to list
                pdf_filenames=args.pdfs,
                n_results=args.n_results,
                chunk_type=args.chunk_type,
                use_hyde=args.hyde
            )
            
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nTOC section query results saved to: {args.save_results}")
        
        elif args.query:
            # Perform the query
            if args.pdfs:
                results = query_tool.query_specific_pdfs(
                    args.query, args.pdfs, args.n_results, args.chunk_type, 
                    use_hyde=args.hyde, use_auto_toc=args.auto_toc
                )
            else:
                results = query_tool.query_entire_collection(
                    args.query, args.n_results, args.chunk_type, use_hyde=args.hyde, use_auto_toc=args.auto_toc
                )
            
            # Save results if requested
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nQuery results saved to: {args.save_results}")
        
        else:
            if args.toc_section and not args.query:
                print("Error: --toc-section requires a query. Use: --query 'your search' --toc-section 'section name'")
            else:
                print("Please provide a query, use --info, --interactive, or see --help for options")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())