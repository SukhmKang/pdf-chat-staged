#!/usr/bin/env python3
"""
Vector Store

Creates embeddings for semantic chunks using ChromaDB and OpenAI embeddings.
Stores both child and parent chunks with proper linking and metadata.
Supports multi-PDF indexing and filtering by PDF source.

Requirements:
- chromadb: pip install chromadb
- openai: pip install openai

Usage:
    python vector_store.py semantic_chunks.json --pdf-path document.pdf --gpt-token gpt_token.txt
    
    # Query with PDF filtering:
    python vector_store.py semantic_chunks.json --query "machine learning" --filter-pdfs doc1.pdf doc2.pdf
"""

import json
import uuid
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import argparse


class DocumentVectorStore:
    def __init__(self, collection_name: str = "document_chunks", persist_directory: str = "./chroma_db", 
                 gpt_token_path: str = "gpt_token.txt"):
        """Initialize the vector store with ChromaDB and OpenAI."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize OpenAI client
        print("Initializing OpenAI client...")
        try:
            with open(gpt_token_path, 'r') as f:
                api_key = f.readline().strip()
            self.openai_client = OpenAI(api_key=api_key)
            print("OpenAI client initialized.")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")
        
        # Get or create collection without embedding function (use raw embeddings)
        # Since we generate OpenAI embeddings manually, we don't need ChromaDB to do embedding
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks with hierarchical structure using OpenAI embeddings"},
            embedding_function=None
        )
        
        self.semantic_data: Optional[Dict[str, Any]] = None
        self.pdf_path: Optional[str] = None
    
    def load_semantic_chunks(self, semantic_chunks_path: str, pdf_path: Optional[str] = None) -> None:
        """Load semantic chunks from JSON file."""
        try:
            with open(semantic_chunks_path, 'r', encoding='utf-8') as f:
                self.semantic_data = json.load(f)
            
            # Extract PDF filename for tracking
            if pdf_path:
                self.pdf_path = pdf_path
                # Get just the filename without path for cleaner identification
                self.pdf_filename = Path(pdf_path).name
            else:
                self.pdf_path = None
                self.pdf_filename = Path(semantic_chunks_path).stem  # Use JSON filename as fallback
            
            semantic_pages = self.semantic_data.get("semantic_pages", [])
            total_child_chunks = sum(len(page.get("child_chunks", [])) for page in semantic_pages)
            total_parent_chunks = sum(len(page.get("parent_chunks", [])) for page in semantic_pages)
            
            print(f"Loaded {len(semantic_pages)} semantic pages from {self.pdf_filename}")
            print(f"Total child chunks: {total_child_chunks}")
            print(f"Total parent chunks: {total_parent_chunks}")
            
        except Exception as e:
            raise ValueError(f"Failed to load semantic chunks: {e}")
    
    def _get_openai_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings using OpenAI's text-embedding-3-small model."""
        embeddings = []
        
        print(f"Generating OpenAI embeddings for {len(texts)} texts...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
                
                # Rate limiting - OpenAI has limits on requests per minute
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Small delay between batches
                    
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings for failed batch to maintain alignment
                batch_embeddings = [[0.0] * 1536 for _ in batch]  # text-embedding-3-small is 1536 dimensions
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def create_embeddings(self) -> None:
        """Create embeddings for all chunks and store in ChromaDB."""
        if not self.semantic_data:
            raise ValueError("No semantic data loaded. Call load_semantic_chunks first.")
        
        print("Creating embeddings for chunks...")
        
        # Only clear chunks from the current PDF being processed (not entire collection)
        if hasattr(self, 'pdf_filename') and self.pdf_filename:
            try:
                # Get all items with this PDF filename
                existing_items = self.collection.get(
                    where={"pdf_filename": self.pdf_filename},
                    include=["metadatas"]
                )
                if existing_items["ids"]:
                    self.collection.delete(ids=existing_items["ids"])
                    print(f"Cleared {len(existing_items['ids'])} existing items for {self.pdf_filename}")
                else:
                    print(f"No existing items found for {self.pdf_filename} (new PDF)")
            except Exception as e:
                print(f"Note: Could not clear existing PDF chunks (this is normal for new PDFs): {e}")
        else:
            print("Warning: No PDF filename set, skipping existing chunk cleanup")
        
        # Prepare batches for efficient insertion
        child_docs = []
        child_metadatas = []
        child_ids = []
        
        parent_docs = []
        parent_metadatas = []
        parent_ids = []
        
        semantic_pages = self.semantic_data.get("semantic_pages", [])
        
        for page in semantic_pages:
            page_id = page.get("page_id", "unknown")
            page_type = page.get("page_type", "unknown")
            page_title = page.get("title", "Untitled")
            original_page_numbers = page.get("original_page_numbers", [])
            page_metadata = page.get("metadata", {})
            
            # Process child chunks
            child_chunks = page.get("child_chunks", [])
            for child_chunk in child_chunks:
                chunk_id = child_chunk.get("chunk_id", str(uuid.uuid4()))
                content = child_chunk.get("content", "")
                
                if not content.strip():
                    continue
                
                # Create comprehensive metadata for child chunk
                # Convert complex objects to strings for ChromaDB compatibility
                toc_info = page_metadata.get("toc_info", {})
                metadata = {
                    "chunk_type": "child",
                    "chunk_id": chunk_id,
                    "page_id": page_id,
                    "page_type": page_type,
                    "page_title": page_title,
                    "section_title": child_chunk.get("section_title", page_title),
                    "page_number": child_chunk.get("page_number", original_page_numbers[0] if original_page_numbers else 1),
                    "original_page_numbers": ",".join(map(str, original_page_numbers)) if original_page_numbers else "",
                    "chunk_index": child_chunk.get("chunk_index", 0),
                    "token_count": child_chunk.get("token_count", 0),
                    "pdf_path": self.pdf_path or "",
                    "pdf_filename": self.pdf_filename,
                    "toc_title": toc_info.get("toc_title", "") if isinstance(toc_info, dict) else "",
                    "toc_level": toc_info.get("toc_level", 0) if isinstance(toc_info, dict) else 0,
                    "corrected_toc_page": toc_info.get("corrected_toc_page", 0) if isinstance(toc_info, dict) else 0,
                    "has_parent": True  # All child chunks have parents
                }
                
                child_docs.append(content)
                child_metadatas.append(metadata)
                child_ids.append(chunk_id)  # Use chunk_id directly since it already has prefix
            
            # Process parent chunks
            parent_chunks = page.get("parent_chunks", [])
            for parent_chunk in parent_chunks:
                chunk_id = parent_chunk.get("chunk_id", str(uuid.uuid4()))
                content = parent_chunk.get("content", "")
                
                if not content.strip():
                    continue
                
                # Get child chunk IDs that belong to this parent
                child_chunk_ids = parent_chunk.get("metadata", {}).get("child_chunks", [])
                
                # Create comprehensive metadata for parent chunk
                # Convert complex objects to strings for ChromaDB compatibility
                toc_info = page_metadata.get("toc_info", {})
                metadata = {
                    "chunk_type": "parent",
                    "chunk_id": chunk_id,
                    "page_id": page_id,
                    "page_type": page_type,
                    "page_title": page_title,
                    "section_title": parent_chunk.get("section_title", page_title),
                    "page_number": parent_chunk.get("page_number", original_page_numbers[0] if original_page_numbers else 1),
                    "original_page_numbers": ",".join(map(str, original_page_numbers)) if original_page_numbers else "",
                    "chunk_index": parent_chunk.get("chunk_index", 0),
                    "token_count": parent_chunk.get("token_count", 0),
                    "pdf_path": self.pdf_path or "",
                    "pdf_filename": self.pdf_filename,
                    "toc_title": toc_info.get("toc_title", "") if isinstance(toc_info, dict) else "",
                    "toc_level": toc_info.get("toc_level", 0) if isinstance(toc_info, dict) else 0,
                    "corrected_toc_page": toc_info.get("corrected_toc_page", 0) if isinstance(toc_info, dict) else 0,
                    "child_chunk_ids": ",".join(child_chunk_ids) if child_chunk_ids else "",
                    "child_count": len(child_chunk_ids)
                }
                
                parent_docs.append(content)
                parent_metadatas.append(metadata)
                parent_ids.append(chunk_id)  # Use chunk_id directly since it already has prefix
        
        # Generate embeddings and add to collection
        if child_docs:
            child_embeddings = self._get_openai_embeddings(child_docs)
            print(f"Adding {len(child_docs)} child chunks to collection...")
            self.collection.add(
                documents=child_docs,
                embeddings=child_embeddings,
                metadatas=child_metadatas,
                ids=child_ids
            )
        
        if parent_docs:
            parent_embeddings = self._get_openai_embeddings(parent_docs)
            print(f"Adding {len(parent_docs)} parent chunks to collection...")
            self.collection.add(
                documents=parent_docs,
                embeddings=parent_embeddings,
                metadatas=parent_metadatas,
                ids=parent_ids
            )
        
        print(f"Successfully created embeddings for {len(child_docs + parent_docs)} chunks")
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract keywords from query using GPT, correcting typos and identifying key terms."""
        if not hasattr(self, 'openai_client') or not self.openai_client:
            # Fallback to simple keyword extraction if no OpenAI client
            
            return self._simple_keyword_extraction(query)
        
        try:
            system_prompt = """You are a keyword extraction expert. Given a user query, extract the most important keywords and phrases that would be useful for exact text matching. Your tasks:

1. Correct any typos or misspellings
2. Extract key terms, proper nouns, and important phrases
3. Include both individual words and multi-word phrases
4. Focus on terms that are likely to appear exactly in documents
5. NEVER include the entire query as a keyword, especially if it's a question
6. AVOID question words (what, who, when, where, why, how, which)
7. AVOID overly general words that would match too many documents (like "book", "argument", "main", "health", "report", "analysis", "study", "data", "information", "research", "development", "project", "system", "program", "policy", "economic", "political", "social", "international", "global", "national", "regional", "local", "major", "important", "significant", "current", "recent", "new", "old", "good", "bad", "high", "low", "large", "small")
8. Prioritize specific terms, proper nouns, technical terms, and distinctive phrases
9. For compound terms like "Health Silk Road", prioritize the full phrase over individual words
10. If the query is very generic (e.g., "What's the main argument of the book?"), return an empty array to avoid irrelevant matches

Return a JSON object with "keywords" array, ordered by specificity (most specific first)."""

            user_prompt = f"Extract keywords from this query for exact text matching: \"{query}\""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            keywords = result.get("keywords", [])
            
            # Also include the original query terms as fallback
            original_keywords = self._simple_keyword_extraction(query)
            
            # Combine and deduplicate
            all_keywords = list(dict.fromkeys(keywords + original_keywords))
            
            return all_keywords
            
        except Exception as e:
            print(f"Warning: GPT keyword extraction failed: {e}")
            return self._simple_keyword_extraction(query)
    
    def _simple_keyword_extraction(self, query: str) -> List[str]:
        """Fallback simple keyword extraction with general word filtering."""
        import re
        
        # Clean and split the query
        original_query = query.strip()
        query = query.lower().strip()
        
        # Comprehensive list of overly general words to avoid
        general_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
            # Question words 
            'what', 'who', 'when', 'why', 'how', 'which', 'where', "what's", "who's", "when's", 
            "why's", "how's", "where's",
            # Overly general descriptive words
            'book', 'books', 'argument', 'arguments', 'main', 'primary', 'central', 'key', 'basic', 'fundamental',
            'health', 'report', 'analysis', 'study', 'data', 'information', 'research', 'development', 
            'project', 'system', 'program', 'policy', 'economic', 'political', 'social', 'international', 
            'global', 'national', 'regional', 'local', 'major', 'important', 'significant', 'current', 
            'recent', 'new', 'old', 'good', 'bad', 'high', 'low', 'large', 'small', 'big', 'great',
            'many', 'few', 'most', 'some', 'all', 'any', 'every', 'each', 'other', 'another', 'such',
            'more', 'less', 'much', 'little', 'very', 'quite', 'rather', 'pretty', 'really', 'just',
            'only', 'also', 'even', 'still', 'yet', 'already', 'now', 'then', 'here', 'there', 
            'that', 'this', 'these', 'those', 'find', 'show', 'tell', 'explain', 'describe'
        }
        
        # Split into words and phrases
        words = re.findall(r'\\b\\w+\\b', query)
        
        # Keep individual important words (not in general_words and length > 4 for higher specificity)
        keywords = []
        for word in words:
            if word not in general_words and len(word) > 4:
                keywords.append(word)
        
        # Always keep the original query as a phrase if it's multi-word
        if len(original_query.split()) > 1:
            keywords.insert(0, original_query)
        
        # Look for quoted phrases (these are explicitly important)
        phrases = re.findall(r'"([^"]*)"', original_query)
        keywords.extend(phrases)
        
        # Filter out keywords that are too general even if they passed initial filter
        filtered_keywords = []
        for keyword in keywords:
            # Multi-word phrases get priority
            if len(keyword.split()) > 1:
                filtered_keywords.append(keyword)
            # Single words need to be more specific (length > 4 and not common)
            elif len(keyword) > 4 and keyword.lower() not in general_words:
                filtered_keywords.append(keyword)
        
        return list(dict.fromkeys(filtered_keywords))  # Remove duplicates while preserving order
    
    def search_exact_keywords(self, keywords: List[str], chunk_type: str = "child", 
                             pdf_filenames: Optional[List[str]] = None,
                             toc_sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for exact keyword matches in document content."""
        # Get all documents from the collection
        conditions = [{"chunk_type": {"$eq": chunk_type}}]
        
        # Add PDF filename filtering if specified
        if pdf_filenames:
            if len(pdf_filenames) == 1:
                conditions.append({"pdf_filename": {"$eq": pdf_filenames[0]}})
            else:
                conditions.append({"pdf_filename": {"$in": pdf_filenames}})
        
        # Add TOC section filtering if specified
        if toc_sections:
            if len(toc_sections) == 1:
                conditions.append({"toc_title": {"$eq": toc_sections[0]}})
            else:
                conditions.append({"toc_title": {"$in": toc_sections}})
        
        # Build where clause
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}
        
        try:
            # Get all matching documents
            all_docs = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            keyword_matches = []
            
            for i, (doc_id, content, metadata) in enumerate(zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"])):
                content_lower = content.lower()
                matched_keywords = []
                total_score = 0
                
                # Define overly general words for penalty scoring
                general_words = {
                    'health', 'report', 'analysis', 'study', 'data', 'information', 'research', 'development', 
                    'project', 'system', 'program', 'policy', 'economic', 'political', 'social', 'international', 
                    'global', 'national', 'regional', 'local', 'major', 'important', 'significant', 'current', 
                    'recent', 'new', 'old', 'good', 'bad', 'high', 'low', 'large', 'small', 'big', 'great',
                    'china', 'chinese', 'country', 'countries', 'government', 'state', 'public', 'private'
                }
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in content_lower:
                        matched_keywords.append(keyword)
                        # Score based on keyword importance (first keywords are more important)
                        keyword_weight = 1.0 / (keywords.index(keyword) + 1)
                        # Count occurrences for additional scoring
                        occurrences = content_lower.count(keyword_lower)
                        
                        # Bonus for exact phrase matches (multi-word keywords)
                        if len(keyword.split()) > 1:
                            keyword_weight *= 3.0  # Triple weight for phrases (increased from 2.0)
                        else:
                            # Penalty for overly general single words
                            if keyword_lower in general_words:
                                keyword_weight *= 0.1  # Heavy penalty for general words
                            elif len(keyword) <= 4:
                                keyword_weight *= 0.3  # Penalty for short single words
                        
                        # Extra bonus for perfect phrase matches in exact case
                        if keyword in content:  # Case-sensitive exact match
                            keyword_weight *= 1.5
                        
                        # Special bonus for rare/specific terms (less than 10 occurrences in content)
                        if occurrences <= 2:
                            keyword_weight *= 1.5  # Boost rare terms
                        elif occurrences > 20:
                            keyword_weight *= 0.7  # Slight penalty for very common terms
                        
                        total_score += keyword_weight * occurrences
                
                if matched_keywords:
                    result = {
                        "id": doc_id,
                        "content": content,
                        "similarity_score": min(total_score, 1.0),  # Cap at 1.0 for consistency
                        "matched_keywords": matched_keywords,
                        "keyword_score": total_score,
                        "source_type": "keyword_match"
                    }
                    
                    # Add metadata
                    result.update({
                        "chunk_id": metadata.get("chunk_id"),
                        "page_id": metadata.get("page_id"),
                        "page_title": metadata.get("page_title"),
                        "section_title": metadata.get("section_title"),
                        "page_number": metadata.get("page_number"),
                        "token_count": metadata.get("token_count"),
                        "pdf_path": metadata.get("pdf_path"),
                        "pdf_filename": metadata.get("pdf_filename"),
                        "metadata": metadata
                    })
                    
                    keyword_matches.append(result)
            
            # Sort by keyword score (descending)
            keyword_matches.sort(key=lambda x: x["keyword_score"], reverse=True)
            
            return keyword_matches
            
        except Exception as e:
            print(f"Warning: Keyword search failed: {e}")
            return []
    
    def query_hybrid_search(self, query: str, n_results: int = 5, 
                          chunk_type: str = "child", include_metadata: bool = True,
                          pdf_filenames: Optional[List[str]] = None, 
                          use_keywords: bool = True,
                          toc_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Hybrid search combining semantic similarity and exact keyword matching.
        
        Args:
            query: The search query text
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            include_metadata: Whether to include metadata in results
            pdf_filenames: Optional list of PDF filenames to filter by
            use_keywords: Whether to include keyword search
            toc_sections: Optional list of TOC section titles to filter by
        """
        all_results = []
        search_info = {"used_semantic": True, "used_keywords": False}
        
        # Perform semantic search
        semantic_results = self.query_similar_chunks(
            query=query,
            n_results=n_results * 2,  # Get more for better hybrid ranking
            chunk_type=chunk_type,
            include_metadata=include_metadata,
            pdf_filenames=pdf_filenames,
            toc_sections=toc_sections
        )
        
        # Add source type to semantic results
        for result in semantic_results["results"]:
            result["source_type"] = "semantic"
        
        all_results.extend(semantic_results["results"])
        
        # Perform keyword search if enabled
        if use_keywords:
            try:
                keywords = self.extract_keywords_from_query(query)
                if keywords:
                    search_info["used_keywords"] = True
                    search_info["keywords"] = keywords
                    print(f"🔍 Extracted keywords: {keywords[:3]}..." if len(keywords) > 3 else f"🔍 Extracted keywords: {keywords}")
                    
                    keyword_results = self.search_exact_keywords(
                        keywords=keywords,
                        chunk_type=chunk_type,
                        pdf_filenames=pdf_filenames,
                        toc_sections=toc_sections
                    )
                    
                    # Add keyword results, but avoid duplicates
                    existing_ids = {result["id"] for result in all_results}
                    for result in keyword_results:
                        if result["id"] not in existing_ids:
                            all_results.append(result)
                        else:
                            # Boost semantic results that also match keywords
                            for semantic_result in all_results:
                                if semantic_result["id"] == result["id"]:
                                    semantic_result["keyword_boost"] = True
                                    semantic_result["matched_keywords"] = result.get("matched_keywords", [])
                                    # Boost similarity score slightly for keyword matches
                                    semantic_result["similarity_score"] = min(1.0, semantic_result["similarity_score"] + 0.1)
                                    break
                    
                    if keyword_results:
                        print(f"📝 Found {len(keyword_results)} keyword matches")
                
            except Exception as e:
                print(f"Warning: Keyword search failed: {e}")
        
        # Sort combined results by similarity score
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Limit to requested number of results
        final_results = all_results[:n_results]
        
        return {
            "query": query,
            "results": final_results,
            "total_found": len(final_results),
            "search_info": search_info,
            "filtered_by_pdfs": pdf_filenames if pdf_filenames else None
        }
    
    def query_similar_chunks(self, query: str, n_results: int = 5, 
                           chunk_type: str = "child", include_metadata: bool = True,
                           pdf_filenames: Optional[List[str]] = None,
                           toc_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query for similar chunks based on text similarity.
        
        Args:
            query: The search query text
            n_results: Number of results to return
            chunk_type: Type of chunks to search ("child" or "parent")
            include_metadata: Whether to include metadata in results
            pdf_filenames: Optional list of PDF filenames to filter by
            toc_sections: Optional list of TOC section titles to filter by
        """
        # Generate query embedding using OpenAI
        query_embedding = self._get_openai_embeddings([query])[0]
        
        # Prepare where clause for chunk type filtering
        conditions = [{"chunk_type": {"$eq": chunk_type}}]
        
        # Add PDF filename filtering if specified
        if pdf_filenames:
            if len(pdf_filenames) == 1:
                conditions.append({"pdf_filename": {"$eq": pdf_filenames[0]}})
            else:
                conditions.append({"pdf_filename": {"$in": pdf_filenames}})
        
        # Add TOC section filtering if specified
        if toc_sections:
            if len(toc_sections) == 1:
                conditions.append({"toc_title": {"$eq": toc_sections[0]}})
            else:
                conditions.append({"toc_title": {"$in": toc_sections}})
        
        # Build final where clause
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
        )
        
        # Format results for easier use
        formatted_results = []
        
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "similarity_score": max(0.0, (2 - results["distances"][0][i]) / 2),  # Convert cosine distance [0,2] to similarity [0,1]
            }
            
            if include_metadata:
                metadata = results["metadatas"][0][i]
                result.update({
                    "chunk_id": metadata.get("chunk_id"),
                    "page_id": metadata.get("page_id"),
                    "page_title": metadata.get("page_title"),
                    "section_title": metadata.get("section_title"),
                    "page_number": metadata.get("page_number"),
                    "token_count": metadata.get("token_count"),
                    "pdf_path": metadata.get("pdf_path"),
                    "pdf_filename": metadata.get("pdf_filename"),
                    "metadata": metadata
                })
            
            formatted_results.append(result)
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "filtered_by_pdfs": pdf_filenames if pdf_filenames else None
        }
    
    def get_parent_chunk(self, child_chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get the parent chunk for a given child chunk."""
        # First get the child chunk to find its page
        child_results = self.collection.get(
            ids=[f"child_{child_chunk_id}"],
            include=["metadatas"]
        )
        
        if not child_results["ids"]:
            return None
        
        child_metadata = child_results["metadatas"][0]
        page_id = child_metadata.get("page_id")
        
        # Find parent chunks in the same page
        parent_results = self.collection.query(
            query_embeddings=None,
            n_results=100,  # Get all parents
            where={"chunk_type": "parent", "page_id": page_id},
            include=["documents", "metadatas"]
        )
        
        # Find the parent that contains this child
        for i, parent_metadata in enumerate(parent_results["metadatas"][0]):
            child_chunk_ids = parent_metadata.get("child_chunk_ids", [])
            if child_chunk_id in child_chunk_ids:
                return {
                    "chunk_id": parent_metadata.get("chunk_id"),
                    "content": parent_results["documents"][0][i],
                    "metadata": parent_metadata
                }
        
        return None
    
    def get_chunk_by_id(self, chunk_id: str, chunk_type: str = "child") -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        full_id = f"{chunk_type}_{chunk_id}"
        
        results = self.collection.get(
            ids=[full_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        return {
            "chunk_id": chunk_id,
            "content": results["documents"][0],
            "metadata": results["metadatas"][0]
        }
    
    def get_available_pdfs(self) -> List[str]:
        """Get list of all PDF filenames in the collection."""
        all_results = self.collection.get(include=["metadatas"])
        pdf_filenames = set(metadata.get("pdf_filename", "") for metadata in all_results["metadatas"] if metadata.get("pdf_filename"))
        return sorted(list(pdf_filenames))
    
    def query_by_pdf(self, pdf_filename: str, chunk_type: str = "child", n_results: int = 10) -> Dict[str, Any]:
        """Get all chunks from a specific PDF."""
        results = self.collection.get(
            where={"pdf_filename": pdf_filename, "chunk_type": chunk_type},
            include=["documents", "metadatas"],
            limit=n_results
        )
        
        formatted_results = []
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            result = {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "chunk_id": metadata.get("chunk_id"),
                "page_number": metadata.get("page_number"),
                "section_title": metadata.get("section_title"),
                "metadata": metadata
            }
            formatted_results.append(result)
        
        return {
            "pdf_filename": pdf_filename,
            "chunk_type": chunk_type,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        # Get all items to count by type
        all_results = self.collection.get(include=["metadatas"])
        
        child_count = sum(1 for metadata in all_results["metadatas"] if metadata.get("chunk_type") == "child")
        parent_count = sum(1 for metadata in all_results["metadatas"] if metadata.get("chunk_type") == "parent")
        
        # Get unique pages, sections, and PDFs
        pages = set(metadata.get("page_id", "") for metadata in all_results["metadatas"])
        sections = set(metadata.get("section_title", "") for metadata in all_results["metadatas"])
        pdf_filenames = set(metadata.get("pdf_filename", "") for metadata in all_results["metadatas"] if metadata.get("pdf_filename"))
        
        # Count chunks per PDF
        pdf_chunk_counts = {}
        for metadata in all_results["metadatas"]:
            pdf_name = metadata.get("pdf_filename", "unknown")
            pdf_chunk_counts[pdf_name] = pdf_chunk_counts.get(pdf_name, 0) + 1
        
        return {
            "total_chunks": len(all_results["ids"]),
            "child_chunks": child_count,
            "parent_chunks": parent_count,
            "unique_pages": len(pages),
            "unique_sections": len(sections),
            "pdf_count": len(pdf_filenames),
            "pdf_filenames": sorted(list(pdf_filenames)),
            "chunks_per_pdf": pdf_chunk_counts,
            "pdf_path": next((metadata.get("pdf_path", "") for metadata in all_results["metadatas"] if metadata.get("pdf_path")), "")
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection.name)
        print(f"Deleted collection: {self.collection.name}")


def main():
    """Example usage of the Vector Store."""
    parser = argparse.ArgumentParser(description="Create vector embeddings for semantic chunks")
    parser.add_argument("semantic_chunks_json", help="Path to semantic chunks JSON file")
    parser.add_argument("--pdf-path", help="Path to the original PDF file")
    parser.add_argument("--gpt-token", default="gpt_token.txt", 
                       help="Path to GPT token file for OpenAI embeddings")
    parser.add_argument("--collection-name", default="document_chunks", 
                       help="ChromaDB collection name")
    parser.add_argument("--persist-dir", default="./chroma_db", 
                       help="Directory to persist ChromaDB data")
    parser.add_argument("--query", help="Test query to run after creating embeddings")
    parser.add_argument("--n-results", type=int, default=5, 
                       help="Number of results to return for test query")
    parser.add_argument("--filter-pdfs", nargs="+", 
                       help="Optional list of PDF filenames to filter query results")
    
    args = parser.parse_args()
    
    try:
        # Initialize vector store
        vector_store = DocumentVectorStore(
            collection_name=args.collection_name,
            persist_directory=args.persist_dir,
            gpt_token_path=args.gpt_token
        )
        
        # Load semantic chunks
        vector_store.load_semantic_chunks(args.semantic_chunks_json, args.pdf_path)
        
        # Create embeddings
        vector_store.create_embeddings()
        
        # Print statistics
        stats = vector_store.get_collection_stats()
        print(f"\nVector Store Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Child chunks: {stats['child_chunks']}")
        print(f"  Parent chunks: {stats['parent_chunks']}")
        print(f"  Unique pages: {stats['unique_pages']}")
        print(f"  Unique sections: {stats['unique_sections']}")
        print(f"  PDFs indexed: {stats['pdf_count']}")
        if stats['pdf_filenames']:
            print(f"  PDF files: {', '.join(stats['pdf_filenames'])}")
        if stats.get('chunks_per_pdf'):
            print(f"  Chunks per PDF:")
            for pdf, count in stats['chunks_per_pdf'].items():
                print(f"    {pdf}: {count} chunks")
        
        # Show available PDFs
        available_pdfs = vector_store.get_available_pdfs()
        if available_pdfs:
            print(f"\nAvailable PDFs: {', '.join(available_pdfs)}")
        
        # Run test query if provided
        if args.query:
            print(f"\nTest Query: '{args.query}'")
            if args.filter_pdfs:
                print(f"Filtering by PDFs: {args.filter_pdfs}")
            print("-" * 50)
            
            results = vector_store.query_similar_chunks(
                query=args.query,
                n_results=args.n_results,
                chunk_type="child",
                pdf_filenames=args.filter_pdfs
            )
            
            if results["filtered_by_pdfs"]:
                print(f"Results filtered by PDFs: {results['filtered_by_pdfs']}")
            
            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. {result['section_title']} (Page {result['page_number']}) - {result['pdf_filename']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
                
                # Show parent chunk info
                parent = vector_store.get_parent_chunk(result['chunk_id'])
                if parent:
                    print(f"   Parent chunk: {parent['chunk_id']} ({parent['metadata'].get('token_count', 0)} tokens)")
        
        print(f"\nVector store created successfully at: {args.persist_dir}")
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())