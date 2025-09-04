#!/usr/bin/env python3
"""
PDF Pipeline Orchestrator

Orchestrates the complete PDF processing pipeline:
1. PDF Layout Parser - Extract layout blocks and detect TOC
2. Reading Order Processor - Organize content blocks in reading order
3. Semantic Chunker - Create hierarchical chunks with GPT analysis
4. Vector Store - Generate embeddings and store in ChromaDB

Collection Auto-Detection:
- PDFs placed in pdfs/<subdirectory>/ are automatically grouped into collections named after the subdirectory
- Example: pdfs/Alliances/book1.pdf → "alliances" collection
- Example: pdfs/Philosophy/paper1.pdf → "philosophy" collection
- This enables automatic organization and cross-document search within topics

Multi-Document Support:
- Documents in the same subdirectory are automatically added to the same collection
- This enables cross-document search and unified knowledge bases within topics
- Use --vector-store-collection to override auto-detection if needed

Requirements:
- All pipeline components (pdf_layout_parser, reading_order_processor, semantic_chunker, vector_store)
- PyMuPDF: pip install PyMuPDF
- chromadb: pip install chromadb
- openai: pip install openai
- tiktoken: pip install tiktoken
- Pillow: pip install Pillow

Usage:
    # Process all PDFs in the pdfs directory (recommended for bulk processing)
    python pdf_pipeline_orchestrator.py --batch-process
    
    # Hard reset: Clear all collections and reprocess everything from scratch
    python pdf_pipeline_orchestrator.py --batch-process --hard-reset
    
    # Process all PDFs in a custom directory
    python pdf_pipeline_orchestrator.py --batch-process --pdfs-dir "my_pdfs"
    
    # Process document with auto-detected collection (single PDF)
    python pdf_pipeline_orchestrator.py pdfs/Alliances/document.pdf
    
    # Build multi-document collection automatically (individual PDFs)
    python pdf_pipeline_orchestrator.py pdfs/Research/doc1.pdf
    python pdf_pipeline_orchestrator.py pdfs/Research/doc2.pdf
    python pdf_pipeline_orchestrator.py pdfs/Research/doc3.pdf
    
    # Override collection name if needed
    python pdf_pipeline_orchestrator.py pdfs/Alliances/doc.pdf --vector-store-collection "custom_name"
    
    # Query across all documents in auto-detected collection
    python pdf_pipeline_orchestrator.py pdfs/Research/doc1.pdf --query-collection "machine learning"
    
    # Check collection status before processing
    python pdf_pipeline_orchestrator.py pdfs/Alliances/doc.pdf --collection-info
"""

import json
import argparse
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Import pipeline components
from pdf_layout_parser import PDFLayoutParser
from reading_order_processor import ReadingOrderProcessor
from semantic_chunker import SemanticChunker
from vector_store import DocumentVectorStore


class PDFPipelineOrchestrator:
    def __init__(self, 
                 pdf_path: str,
                 output_dir: str = "processed_output",
                 gpt_token_path: str = "gpt_token.txt",
                 vector_store_collection: str = None,
                 vector_store_dir: str = "./chroma_db",
                 keep_intermediate_files: bool = True,
                 figure_analysis_min_size: float = 0.15,
                 figure_analysis_max_count: int = 20):
        """
        Initialize the PDF processing pipeline orchestrator.
        
        Args:
            pdf_path: Path to the PDF file to process
            output_dir: Directory to save intermediate output files
            gpt_token_path: Path to GPT token file for OpenAI API
            vector_store_collection: Name of the ChromaDB collection (auto-determined if None)
            vector_store_dir: Directory for ChromaDB persistence
            keep_intermediate_files: Whether to keep intermediate JSON files
            figure_analysis_min_size: Minimum fraction of page area for figure GPT analysis (0.0-1.0)
            figure_analysis_max_count: Maximum number of figures to process with GPT per document
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.gpt_token_path = Path(gpt_token_path)
        self.vector_store_dir = Path(vector_store_dir)
        self.keep_intermediate_files = keep_intermediate_files
        self.figure_analysis_min_size = figure_analysis_min_size
        self.figure_analysis_max_count = figure_analysis_max_count
        
        # Auto-determine collection name from subdirectory structure if not provided
        if vector_store_collection is None:
            self.vector_store_collection = self._determine_collection_name()
        else:
            self.vector_store_collection = vector_store_collection
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate file paths based on PDF name
        pdf_stem = self.pdf_path.stem
        self.layout_output_path = self.output_dir / f"{pdf_stem}_layout.json"
        self.reading_order_output_path = self.output_dir / f"{pdf_stem}_reading_order.json"
        self.semantic_chunks_output_path = self.output_dir / f"{pdf_stem}_semantic_chunks.json"
        
        # Pipeline results
        self.layout_data: Optional[Dict[str, Any]] = None
        self.reading_order_data: Optional[Dict[str, Any]] = None
        self.semantic_chunks_data: Optional[Dict[str, Any]] = None
        self.vector_store: Optional[DocumentVectorStore] = None
        
        print(f"Orchestrator initialized for: {self.pdf_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Collection name: {self.vector_store_collection}")
        print(f"Keep intermediate files: {self.keep_intermediate_files}")
    
    def _determine_collection_name(self) -> str:
        """
        Automatically determine collection name from PDF's subdirectory structure.
        Expects PDFs to be in pdfs/<subdirectory>/<pdf_file.pdf>
        
        Returns:
            Collection name based on subdirectory, or "default" if not in expected structure
        """
        pdf_path_parts = self.pdf_path.parts
        
        # Look for 'pdfs' directory in the path
        if 'pdfs' in pdf_path_parts:
            pdfs_index = pdf_path_parts.index('pdfs')
            
            # Check if there's a subdirectory after 'pdfs'
            if pdfs_index + 1 < len(pdf_path_parts):
                subdirectory = pdf_path_parts[pdfs_index + 1]
                
                # Make sure it's not the PDF file itself (in case PDF is directly in pdfs/)
                if not subdirectory.endswith('.pdf'):
                    collection_name = subdirectory.lower().replace(' ', '_').replace('-', '_')
                    print(f"Auto-determined collection name from subdirectory: {collection_name}")
                    return collection_name
        
        # Fallback: use "default" if not in expected pdfs/<subdirectory>/ structure
        print("PDF not in expected pdfs/<subdirectory>/ structure, using 'default' collection")
        return "default"
    
    @staticmethod
    def _determine_collection_name_static(pdf_path: str) -> str:
        """
        Static version of collection name determination that doesn't require orchestrator instance.
        """
        pdf_path_obj = Path(pdf_path)
        pdf_path_parts = pdf_path_obj.parts
        
        # Look for 'pdfs' directory in the path
        if 'pdfs' in pdf_path_parts:
            pdfs_index = pdf_path_parts.index('pdfs')
            
            # Check if there's a subdirectory after 'pdfs'
            if pdfs_index + 1 < len(pdf_path_parts):
                subdirectory = pdf_path_parts[pdfs_index + 1]
                
                # Make sure it's not the PDF file itself (in case PDF is directly in pdfs/)
                if not subdirectory.endswith('.pdf'):
                    collection_name = subdirectory.lower().replace(' ', '_').replace('-', '_')
                    print(f"Auto-determined collection name from subdirectory: {collection_name}")
                    return collection_name
        
        # Fallback: use "default" if not in expected pdfs/<subdirectory>/ structure
        print("PDF not in expected pdfs/<subdirectory>/ structure, using 'default' collection")
        return "default"
    
    @staticmethod
    def detect_and_handle_pdf_moves(pdf_path: str, target_collection: str, 
                                   vector_store_dir: str = "./chroma_db", 
                                   gpt_token_path: str = "gpt_token.txt") -> Dict[str, Any]:
        """
        Simple file-based PDF processing detection.
        This method is deprecated in favor of file-based tracking in batch processing.
        """
        pdf_filename = Path(pdf_path).name
        result = {
            "pdf_filename": pdf_filename,
            "target_collection": target_collection,
            "found_in_collections": [],
            "moved_from": None,
            "action_needed": "process"  # Always process for simplicity
        }
        
        # Note: This method is now simplified and always returns "process"
        # to avoid ChromaDB segmentation faults. The batch processing method
        # uses file-based tracking instead.
        
        return result
    
    @staticmethod
    def cleanup_deleted_pdfs(pdfs_dir: str = "pdfs", 
                           vector_store_dir: str = "./chroma_db",
                           gpt_token_path: str = "gpt_token.txt") -> Dict[str, Any]:
        """
        Clean up PDFs from collections that no longer exist in the file system.
        
        Args:
            pdfs_dir: Directory containing PDFs
            vector_store_dir: Directory for ChromaDB persistence
            gpt_token_path: Path to GPT token file
            
        Returns:
            Dictionary with cleanup results
        """
        from pathlib import Path
        
        result = {
            "collections_checked": [],
            "pdfs_removed": [],
            "logs_cleaned": 0,
            "errors": []
        }
        
        try:
            # Get all existing PDF files in the directory
            pdfs_path = Path(pdfs_dir)
            existing_pdfs = set()
            if pdfs_path.exists():
                for pdf_file in pdfs_path.glob("**/*.pdf"):
                    existing_pdfs.add(pdf_file.name)
            
            print(f"🧹 Cleanup: Found {len(existing_pdfs)} existing PDFs in {pdfs_dir}")
            
            # Get all collections and check for orphaned PDFs
            try:
                from vector_store import DocumentVectorStore
                
                # List of known collections (you might want to make this dynamic)
                known_collections = ["philosophy", "rag", "alliances", "china", "default"]
                
                for collection_name in known_collections:
                    try:
                        print(f"   🔍 Checking collection: {collection_name}")
                        vector_store = DocumentVectorStore(
                            collection_name=collection_name,
                            persist_directory=vector_store_dir,
                            gpt_token_path=gpt_token_path
                        )
                        
                        # Get PDFs in this collection
                        collection_pdfs = vector_store.get_available_pdfs()
                        result["collections_checked"].append(collection_name)
                        
                        # Find PDFs that no longer exist in file system
                        orphaned_pdfs = [pdf for pdf in collection_pdfs if pdf not in existing_pdfs]
                        
                        if orphaned_pdfs:
                            print(f"   🗑️  Found {len(orphaned_pdfs)} orphaned PDFs in {collection_name}")
                            
                            for orphaned_pdf in orphaned_pdfs:
                                try:
                                    # Remove from collection
                                    remove_result = PDFPipelineOrchestrator.remove_pdf_from_collections(
                                        orphaned_pdf, [collection_name], vector_store_dir, gpt_token_path
                                    )
                                    
                                    if remove_result["removed_from"]:
                                        result["pdfs_removed"].append({
                                            "pdf": orphaned_pdf,
                                            "collection": collection_name
                                        })
                                        print(f"      ✅ Removed {orphaned_pdf}")
                                    
                                    # Clean up processing log
                                    processing_log_dir = Path("processed_pdfs_log")
                                    log_file = processing_log_dir / f"{collection_name}_{orphaned_pdf}.processed"
                                    if log_file.exists():
                                        log_file.unlink()
                                        result["logs_cleaned"] += 1
                                        
                                except Exception as e:
                                    error_msg = f"Failed to remove {orphaned_pdf} from {collection_name}: {e}"
                                    result["errors"].append(error_msg)
                                    print(f"      ❌ {error_msg}")
                        else:
                            print(f"   ✅ No orphaned PDFs found in {collection_name}")
                            
                    except Exception as e:
                        if "does not exist" in str(e) or "not found" in str(e):
                            # Collection doesn't exist, skip silently
                            continue
                        else:
                            error_msg = f"Error checking collection {collection_name}: {e}"
                            result["errors"].append(error_msg)
                            print(f"   ⚠️  {error_msg}")
                            
            except ImportError as e:
                error_msg = f"Could not import vector_store: {e}"
                result["errors"].append(error_msg)
                print(f"⚠️  {error_msg}")
                
        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        return result
    
    @staticmethod
    def remove_pdf_from_collections(pdf_filename: str, collection_names: list, 
                                  vector_store_dir: str = "./chroma_db", 
                                  gpt_token_path: str = "gpt_token.txt") -> Dict[str, Any]:
        """
        Remove a PDF from specified collections.
        
        Args:
            pdf_filename: Name of the PDF file to remove
            collection_names: List of collection names to remove from
            vector_store_dir: Directory containing ChromaDB data
            gpt_token_path: Path to GPT token file
            
        Returns:
            Dictionary with removal results
        """
        result = {
            "pdf_filename": pdf_filename,
            "removed_from": [],
            "failed_removals": []
        }
        
        try:
            for collection_name in collection_names:
                try:
                    # Initialize vector store for this collection
                    vector_store = DocumentVectorStore(
                        collection_name=collection_name,
                        persist_directory=vector_store_dir,
                        gpt_token_path=gpt_token_path
                    )
                    
                    # Get all items with this PDF filename
                    all_items = vector_store.collection.get(include=["metadatas"])
                    ids_to_remove = []
                    
                    for i, metadata in enumerate(all_items["metadatas"]):
                        if metadata.get("pdf_filename") == pdf_filename:
                            ids_to_remove.append(all_items["ids"][i])
                    
                    if ids_to_remove:
                        vector_store.collection.delete(ids=ids_to_remove)
                        result["removed_from"].append(collection_name)
                        print(f"   🗑️  Removed {len(ids_to_remove)} chunks of {pdf_filename} from collection '{collection_name}'")
                    
                except Exception as e:
                    result["failed_removals"].append({"collection": collection_name, "error": str(e)})
                    print(f"   ⚠️  Failed to remove {pdf_filename} from collection '{collection_name}': {e}")
            
            return result
            
        except Exception as e:
            print(f"Error during PDF removal: {e}")
            result["error"] = str(e)
            return result
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete PDF processing pipeline.
        
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            print("=" * 60)
            print("Starting PDF Processing Pipeline")
            print("=" * 60)
            
            # Step 1: PDF Layout Parsing
            print("\n1. Running PDF Layout Parser...")
            self._run_layout_parser()
            
            # Step 2: Reading Order Processing
            print("\n2. Running Reading Order Processor...")
            self._run_reading_order_processor()
            
            # Step 3: Semantic Chunking
            print("\n3. Running Semantic Chunker...")
            self._run_semantic_chunker()
            
            # Step 4: Vector Store Creation
            print("\n4. Creating Vector Store...")
            self._run_vector_store()
            
            # Step 5: Generate Summary
            print("\n5. Generating Pipeline Summary...")
            summary = self._generate_pipeline_summary()
            
            # Cleanup if requested
            if not self.keep_intermediate_files:
                self._cleanup_intermediate_files()
            
            print("\n" + "=" * 60)
            print("PDF Processing Pipeline Complete!")
            print("=" * 60)
            
            return summary
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def _run_layout_parser(self) -> None:
        """Run the PDF layout parser."""
        try:
            print(f"   📄 Opening PDF: {self.pdf_path}")
            print(f"   🔧 Initializing layout parser...")
            
            parser = PDFLayoutParser(str(self.pdf_path), str(self.gpt_token_path))
            
            print(f"   📖 Starting layout analysis...")
            self.layout_data = parser.parse()
            
            # Print progress during parsing
            total_pages = self.layout_data["document_info"]["total_pages"]
            print(f"   ⚡ Processed {total_pages} pages")
            
            # Save layout data
            if self.keep_intermediate_files:
                print(f"   💾 Saving layout data to: {self.layout_output_path}")
            parser.save_json(str(self.layout_output_path))
            parser.close()
            
            # Print detailed summary
            total_text_spans = sum(len(page["text_spans"]) for page in self.layout_data["pages"])
            total_figures = sum(len(page["figures"]) for page in self.layout_data["pages"])
            total_tables = sum(len(page["tables"]) for page in self.layout_data["pages"])
            total_footnotes = sum(len(page["footnotes"]) for page in self.layout_data["pages"])
            
            print(f"   ✓ Layout parsing complete")
            print(f"   ✓ Pages processed: {total_pages}")
            print(f"   ✓ Text spans extracted: {total_text_spans:,}")
            print(f"   ✓ Figures found: {total_figures}")
            print(f"   ✓ Tables found: {total_tables}")
            print(f"   ✓ Footnotes found: {total_footnotes}")
            
            # TOC info
            if "table_of_contents" in self.layout_data:
                toc = self.layout_data["table_of_contents"]
                print(f"   ✓ TOC detected: {len(toc['entries'])} entries (confidence: {toc['confidence']:.2f})")
                if len(toc['entries']) > 0:
                    print(f"   📚 TOC sections: {', '.join([entry['title'][:30] + ('...' if len(entry['title']) > 30 else '') for entry in toc['entries'][:3]])}")
                    if len(toc['entries']) > 3:
                        print(f"       (and {len(toc['entries']) - 3} more sections)")
            else:
                print(f"   ℹ️  No table of contents detected")
                
        except Exception as e:
            raise RuntimeError(f"Layout parsing failed: {e}")
    
    def _run_reading_order_processor(self) -> None:
        """Run the reading order processor."""
        try:
            print(f"   📂 Loading layout data from: {self.layout_output_path.name}")
            processor = ReadingOrderProcessor(str(self.layout_output_path))
            
            total_pages = self.layout_data["document_info"]["total_pages"]
            print(f"   🔄 Processing reading order for {total_pages} pages...")
            print(f"   🔍 Analyzing block relationships...")
            
            self.reading_order_data = processor.process()
            
            # Save reading order data
            if self.keep_intermediate_files:
                print(f"   💾 Saving reading order data to: {self.reading_order_output_path}")
            processor.save_json(str(self.reading_order_output_path))
            
            # Print detailed summary
            processing_info = self.reading_order_data["processing_info"]
            print(f"   ✓ Reading order processing complete")
            print(f"   ✓ Content blocks organized: {processing_info['total_blocks']}")
            
            # Show block type breakdown
            block_types = processing_info['block_types']
            for block_type, count in block_types.items():
                print(f"   📝 {block_type.title()} blocks: {count}")
            
            print(f"   🧹 Headers removed: {processing_info['headers_removed']}")
            print(f"   🧹 Footers removed: {processing_info['footers_removed']}")
            print(f"   🔗 TOC associations created: {processing_info['toc_associations']}")
                
        except Exception as e:
            raise RuntimeError(f"Reading order processing failed: {e}")
    
    def _run_semantic_chunker(self) -> None:
        """Run the semantic chunker."""
        try:
            print(f"   📖 Loading reading order data from: {self.reading_order_output_path.name}")
            print(f"   🧠 Initializing semantic chunker with GPT analysis...")
            
            chunker = SemanticChunker(
                str(self.reading_order_output_path), 
                str(self.pdf_path), 
                str(self.gpt_token_path),
                figure_analysis_min_size=self.figure_analysis_min_size,
                figure_analysis_max_count=self.figure_analysis_max_count
            )
            
            # Get content block count before processing
            total_blocks = len(self.reading_order_data["content_blocks"])
            print(f"   🔄 Processing {total_blocks} content blocks...")
            print(f"   🤖 Running GPT analysis for semantic understanding...")
            
            self.semantic_chunks_data = chunker.process()
            
            # Save semantic chunks data
            if self.keep_intermediate_files:
                print(f"   💾 Saving semantic chunks to: {self.semantic_chunks_output_path}")
            chunker.save_json(str(self.semantic_chunks_output_path))
            chunker.close()
            
            # Print detailed summary
            chunking_info = self.semantic_chunks_data["chunking_info"]
            print(f"   ✓ Semantic chunking complete")
            print(f"   ✓ Semantic pages created: {chunking_info['total_semantic_pages']}")
            
            # Show page type breakdown
            page_types = chunking_info['page_types']
            for page_type, count in page_types.items():
                print(f"   📄 {page_type.title()} pages: {count}")
            
            print(f"   📦 Child chunks generated: {chunking_info['total_child_chunks']}")
            print(f"   📦 Parent chunks generated: {chunking_info['total_parent_chunks']}")
            print(f"   🔗 TOC-linked pages: {chunking_info['toc_linked_pages']}")
            
            # Show figure analysis info
            figures_with_ai = sum(1 for page in chunker.semantic_pages 
                                 if page.page_type == "figure" and 
                                 page.metadata and page.metadata.get("has_ai_summary"))
            total_figures = chunking_info['page_types'].get('figure', 0)
            if total_figures > 0:
                print(f"   🖼️  Figures with AI analysis: {figures_with_ai}/{total_figures}")
                
        except Exception as e:
            raise RuntimeError(f"Semantic chunking failed: {e}")
    
    def _run_vector_store(self) -> None:
        """Run the vector store creation."""
        try:
            print(f"   🗂️  Creating collection: {self.vector_store_collection}")
            print(f"   💾 Database location: {self.vector_store_dir}")
            
            # Initialize vector store
            self.vector_store = DocumentVectorStore(
                collection_name=self.vector_store_collection,
                persist_directory=str(self.vector_store_dir),
                gpt_token_path=str(self.gpt_token_path)
            )
            
            # Load semantic chunks
            print(f"   📂 Loading semantic chunks from: {self.semantic_chunks_output_path.name}")
            self.vector_store.load_semantic_chunks(
                str(self.semantic_chunks_output_path), 
                str(self.pdf_path)
            )
            
            # Get chunk counts before embedding
            chunking_info = self.semantic_chunks_data["chunking_info"]
            total_child = chunking_info['total_child_chunks']
            total_parent = chunking_info['total_parent_chunks']
            total_chunks = total_child + total_parent
            
            print(f"   🔢 Preparing to embed {total_chunks} chunks ({total_child} child + {total_parent} parent)")
            print(f"   🤖 Generating OpenAI embeddings...")
            
            self.vector_store.create_embeddings()
            
            # Print detailed summary
            stats = self.vector_store.get_collection_stats()
            print(f"   ✓ Vector store creation complete")
            print(f"   ✅ Successfully embedded {stats['total_chunks']:,} chunks")
            print(f"   📦 Child chunks: {stats['child_chunks']:,}")
            print(f"   📦 Parent chunks: {stats['parent_chunks']:,}")
            print(f"   🗂️  Collection name: {self.vector_store_collection}")
            print(f"   📍 Storage location: {self.vector_store_dir}")
            
            # Show PDF info
            pdf_name = self.pdf_path.name
            print(f"   📄 PDF processed: {pdf_name}")
            
        except Exception as e:
            raise RuntimeError(f"Vector store creation failed: {e}")
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the pipeline results."""
        summary = {
            "input_file": str(self.pdf_path),
            "output_directory": str(self.output_dir),
            "processing_timestamp": str(Path().cwd()),  # Placeholder
            "pipeline_stages": {
                "layout_parser": {
                    "status": "completed" if self.layout_data else "failed",
                    "output_file": str(self.layout_output_path) if self.keep_intermediate_files else None
                },
                "reading_order_processor": {
                    "status": "completed" if self.reading_order_data else "failed",
                    "output_file": str(self.reading_order_output_path) if self.keep_intermediate_files else None
                },
                "semantic_chunker": {
                    "status": "completed" if self.semantic_chunks_data else "failed",
                    "output_file": str(self.semantic_chunks_output_path) if self.keep_intermediate_files else None
                },
                "vector_store": {
                    "status": "completed" if self.vector_store else "failed",
                    "collection_name": self.vector_store_collection,
                    "storage_directory": str(self.vector_store_dir)
                }
            }
        }
        
        # Add detailed statistics if available
        if self.layout_data:
            layout_stats = {
                "total_pages": self.layout_data["document_info"]["total_pages"],
                "text_spans": sum(len(page["text_spans"]) for page in self.layout_data["pages"]),
                "figures": sum(len(page["figures"]) for page in self.layout_data["pages"]),
                "tables": sum(len(page["tables"]) for page in self.layout_data["pages"]),
                "footnotes": sum(len(page["footnotes"]) for page in self.layout_data["pages"])
            }
            summary["pipeline_stages"]["layout_parser"]["statistics"] = layout_stats
        
        if self.reading_order_data:
            summary["pipeline_stages"]["reading_order_processor"]["statistics"] = \
                self.reading_order_data["processing_info"]
        
        if self.semantic_chunks_data:
            summary["pipeline_stages"]["semantic_chunker"]["statistics"] = \
                self.semantic_chunks_data["chunking_info"]
        
        if self.vector_store:
            summary["pipeline_stages"]["vector_store"]["statistics"] = \
                self.vector_store.get_collection_stats()
        
        return summary
    
    def _cleanup_intermediate_files(self) -> None:
        """Remove intermediate JSON files if requested."""
        files_to_remove = [
            self.layout_output_path,
            self.reading_order_output_path,
            self.semantic_chunks_output_path
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                print(f"   ✓ Removed intermediate file: {file_path}")
    
    def query_processed_document(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the processed document using the vector store.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            
        Returns:
            Query results with chunk information
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Run the pipeline first.")
        
        return self.vector_store.query_similar_chunks(
            query=query,
            n_results=n_results,
            chunk_type="child",
            pdf_filenames=[self.pdf_path.name]
        )
    
    def query_collection(self, query: str, n_results: int = 5, pdf_filenames: Optional[list] = None) -> Dict[str, Any]:
        """
        Query the entire collection (potentially multiple PDFs).
        
        Args:
            query: Search query text
            n_results: Number of results to return
            pdf_filenames: Optional list of PDF filenames to filter by
            
        Returns:
            Query results with chunk information from across the collection
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Run the pipeline first.")
        
        return self.vector_store.query_similar_chunks(
            query=query,
            n_results=n_results,
            chunk_type="child",
            pdf_filenames=pdf_filenames
        )
    
    @staticmethod
    def get_collection_info(vector_store_dir: str, collection_name: str, gpt_token_path: str = "gpt_token.txt") -> Dict[str, Any]:
        """
        Get information about an existing vector store collection.
        
        Args:
            vector_store_dir: Directory containing the ChromaDB data
            collection_name: Name of the collection to inspect
            gpt_token_path: Path to GPT token file
            
        Returns:
            Collection statistics and available PDFs
        """
        vector_store = DocumentVectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_dir,
            gpt_token_path=gpt_token_path
        )
        
        stats = vector_store.get_collection_stats()
        available_pdfs = vector_store.get_available_pdfs()
        
        return {
            "collection_stats": stats,
            "available_pdfs": available_pdfs,
            "collection_name": collection_name,
            "storage_location": vector_store_dir
        }
    
    @staticmethod
    def clear_all_collections(vector_store_dir: str = "./chroma_db") -> Dict[str, Any]:
        """
        Clear all collections from ChromaDB (hard reset).
        
        Args:
            vector_store_dir: Directory containing ChromaDB data
            
        Returns:
            Dictionary with clearing results
        """
        import chromadb
        from chromadb.config import Settings
        
        result = {
            "collections_deleted": [],
            "failed_deletions": [],
            "total_collections": 0
        }
        
        try:
            persist_directory = Path(vector_store_dir)
            if not persist_directory.exists():
                print(f"ChromaDB directory not found: {vector_store_dir}")
                return result
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # List all collections
            collections = client.list_collections()
            result["total_collections"] = len(collections)
            
            if not collections:
                print("No collections found to delete.")
                return result
            
            print(f"🗑️  Deleting {len(collections)} collections...")
            
            for collection in collections:
                try:
                    print(f"   🗑️  Deleting collection: {collection.name}")
                    client.delete_collection(collection.name)
                    result["collections_deleted"].append(collection.name)
                except Exception as e:
                    print(f"   ⚠️  Failed to delete collection {collection.name}: {e}")
                    result["failed_deletions"].append({"collection": collection.name, "error": str(e)})
            
            print(f"✅ Successfully deleted {len(result['collections_deleted'])} collections")
            if result["failed_deletions"]:
                print(f"⚠️  Failed to delete {len(result['failed_deletions'])} collections")
            
            return result
            
        except Exception as e:
            print(f"Error during collection clearing: {e}")
            result["error"] = str(e)
            return result
    
    @staticmethod
    def batch_process_pdfs_directory(pdfs_dir: str = "pdfs", 
                                   output_dir: str = "processed_output",
                                   gpt_token_path: str = "gpt_token.txt",
                                   vector_store_dir: str = "./chroma_db",
                                   vector_store_only: bool = True,
                                   skip_existing: bool = True,
                                   hard_reset: bool = False,
                                   figure_analysis_min_size: float = 0.15,
                                   figure_analysis_max_count: int = 20) -> Dict[str, Any]:
        """
        Process all PDFs in the pdfs directory and its subdirectories.
        Simplified batch processing that avoids problematic ChromaDB detection.
        
        Args:
            pdfs_dir: Directory containing PDFs to process
            output_dir: Directory to save output files
            gpt_token_path: Path to GPT token file
            vector_store_dir: Directory for ChromaDB persistence
            vector_store_only: Skip saving intermediate JSON files
            skip_existing: Skip PDFs that already have processing logs (file-based tracking)
            hard_reset: Clear all collections before processing (ignores skip_existing)
            figure_analysis_min_size: Minimum fraction of page area for figure GPT analysis
            figure_analysis_max_count: Maximum number of figures to process with GPT per document
            
        Returns:
            Summary of batch processing results
        """
        from pathlib import Path
        import glob
        
        # Handle hard reset if requested
        if hard_reset:
            print("🔥 HARD RESET: Clearing all collections before processing...")
            print("⚠️  This will delete all existing PDF embeddings!")
            
            # Clear all collections
            clear_result = PDFPipelineOrchestrator.clear_all_collections(vector_store_dir)
            
            if clear_result["collections_deleted"]:
                print(f"✅ Cleared {len(clear_result['collections_deleted'])} collections")
            
            # Clear processing logs
            processing_log_dir = Path("processed_pdfs_log")
            if processing_log_dir.exists():
                for log_file in processing_log_dir.glob("*.processed"):
                    log_file.unlink()
                print("✅ Cleared processing logs")
            
            # Force skip_existing to False when doing hard reset
            skip_existing = False
            print("🔄 Will process all PDFs from scratch...")
            print("=" * 80)
        
        pdfs_path = Path(pdfs_dir)
        if not pdfs_path.exists():
            raise ValueError(f"PDFs directory not found: {pdfs_dir}")
        
        # Find all PDF files recursively
        pdf_files = list(pdfs_path.glob("**/*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdfs_dir}")
            return {"processed": [], "skipped": [], "failed": [], "total_found": 0}
        
        print(f"🚀 Starting batch processing of PDFs directory...")
        print(f"🔍 Found {len(pdf_files)} PDF files in {pdfs_dir}")
        print("=" * 80)
        
        # Run cleanup first to remove orphaned PDFs from collections
        if not hard_reset:  # Skip cleanup if we're doing hard reset (everything gets cleared anyway)
            print("🧹 CLEANUP: Checking for orphaned PDFs in collections...")
            cleanup_result = PDFPipelineOrchestrator.cleanup_deleted_pdfs(
                pdfs_dir, vector_store_dir, gpt_token_path
            )
            
            if cleanup_result["pdfs_removed"]:
                print(f"✅ Removed {len(cleanup_result['pdfs_removed'])} orphaned PDFs from collections")
                if cleanup_result["logs_cleaned"]:
                    print(f"✅ Cleaned {cleanup_result['logs_cleaned']} processing logs")
            else:
                print("✅ No orphaned PDFs found")
            
            if cleanup_result["errors"]:
                print(f"⚠️  {len(cleanup_result['errors'])} cleanup errors occurred")
            
            print("=" * 80)
        
        processed = []
        skipped = []
        failed = []
        moved_count = 0
        
        # Create processing log directory
        processing_log_dir = Path("processed_pdfs_log")
        processing_log_dir.mkdir(exist_ok=True)
        
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_name = pdf_file.name
            
            # Handle relative path display safely
            try:
                if pdf_file.is_absolute():
                    relative_path = str(pdf_file.relative_to(Path.cwd()))
                else:
                    relative_path = str(pdf_file)
            except ValueError:
                # If relative_to fails, just use the path as-is
                relative_path = str(pdf_file)
            
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_name}")
            print(f"   📁 Path: {relative_path}")
            
            # Determine target collection for this PDF
            target_collection = PDFPipelineOrchestrator._determine_collection_name_static(str(pdf_file))
            print(f"   🗂️  Target collection: {target_collection}")
            
            # Check if PDF already processed using file-based tracking
            if skip_existing:
                log_file = processing_log_dir / f"{target_collection}_{pdf_name}.processed"
                if log_file.exists():
                    print(f"   ✅ Already processed (found processing log) - skipping")
                    skipped.append({
                        "file": str(pdf_file), 
                        "reason": "already_processed",
                        "collection": target_collection
                    })
                    continue
            
            try:
                # Check if PDF exists in other collections using processing logs
                processing_logs = list(processing_log_dir.glob(f"*_{pdf_name}.processed"))
                other_collection_logs = [log for log in processing_logs 
                                       if not log.name.startswith(f"{target_collection}_")]
                
                if other_collection_logs:
                    print(f"   🔄 PDF found in other collections, handling move...")
                    moved_from = []
                    
                    for log_file in other_collection_logs:
                        # Extract collection name from log file name
                        log_name = log_file.name
                        old_collection = log_name.replace(f"_{pdf_name}.processed", "")
                        moved_from.append(old_collection)
                        
                        print(f"   📦 Removing from collection: {old_collection}")
                        
                        # Remove PDF from the old collection
                        removal_result = PDFPipelineOrchestrator.remove_pdf_from_collections(
                            pdf_name, [old_collection], vector_store_dir, gpt_token_path
                        )
                        
                        if removal_result["removed_from"]:
                            print(f"   ✅ Successfully removed from {old_collection}")
                            # Remove the old processing log
                            log_file.unlink()
                            print(f"   🗑️  Removed old processing log: {log_file.name}")
                        else:
                            print(f"   ⚠️  Failed to remove from {old_collection}")
                    
                    if moved_from:
                        print(f"   🚀 Moving PDF from {', '.join(moved_from)} → {target_collection}")
                        moved_count += 1
                
                # Create orchestrator for this PDF
                orchestrator = PDFPipelineOrchestrator(
                    pdf_path=str(pdf_file),
                    output_dir=output_dir,
                    gpt_token_path=gpt_token_path,
                    vector_store_collection=None,  # Auto-detect from subdirectory
                    vector_store_dir=vector_store_dir,
                    keep_intermediate_files=not vector_store_only,
                    figure_analysis_min_size=figure_analysis_min_size,
                    figure_analysis_max_count=figure_analysis_max_count
                )
                
                print(f"   🔧 Collection: {orchestrator.vector_store_collection}")
                
                # Run the pipeline
                summary = orchestrator.run_complete_pipeline()
                
                # Create processing log
                log_file = processing_log_dir / f"{orchestrator.vector_store_collection}_{pdf_name}.processed"
                log_file.write_text(f"Processed: {Path.cwd()}\nTimestamp: {log_file.stat().st_mtime if log_file.exists() else 'new'}\n")
                
                processed_entry = {
                    "file": str(pdf_file),
                    "collection": orchestrator.vector_store_collection,
                    "summary": summary
                }
                
                # Add move information if applicable
                if other_collection_logs:
                    processed_entry["moved_from"] = moved_from
                
                processed.append(processed_entry)
                
                print(f"   ✅ Successfully processed: {pdf_name}")
                
            except Exception as e:
                print(f"   ❌ Failed to process {pdf_name}: {str(e)[:100]}...")
                failed.append({
                    "file": str(pdf_file),
                    "error": str(e)
                })
                continue
        
        # Print final summary
        print("\n" + "=" * 80)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully processed: {len(processed)}")
        print(f"🔄 PDFs moved between collections: {moved_count}")
        print(f"⏭️  Skipped: {len(skipped)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"📄 Total PDFs found: {len(pdf_files)}")
        
        if processed:
            print(f"\n🗂️  Collections created/updated:")
            collections = {}
            for item in processed:
                collection = item["collection"]
                if collection not in collections:
                    collections[collection] = []
                collections[collection].append(Path(item["file"]).name)
            
            for collection, files in collections.items():
                print(f"   {collection}: {len(files)} PDFs")
                for file in files[:3]:  # Show first 3
                    print(f"     - {file}")
                if len(files) > 3:
                    print(f"     - ... and {len(files) - 3} more")
        
        # Show moved PDFs details
        if moved_count > 0:
            print(f"\n🔄 Moved PDFs details:")
            for item in processed:
                if "moved_from" in item:
                    pdf_name = Path(item["file"]).name
                    moved_from = item["moved_from"]
                    target_collection = item["collection"]
                    print(f"   📦 {pdf_name}: {', '.join(moved_from)} → {target_collection}")
        
        if skipped:
            print(f"\n⏭️  Skipped files:")
            skip_reasons = {}
            for item in skipped:
                reason = item["reason"]
                if reason not in skip_reasons:
                    skip_reasons[reason] = []
                skip_reasons[reason].append(Path(item["file"]).name)
            
            for reason, files in skip_reasons.items():
                reason_display = reason.replace("_", " ").title()
                print(f"   {reason_display}: {len(files)} files")
        
        if failed:
            print(f"\n❌ Failed files:")
            for item in failed:
                file_name = Path(item["file"]).name
                error = item["error"][:60] + "..." if len(item["error"]) > 60 else item["error"]
                print(f"   - {file_name}: {error}")
        
        
        return {
            "processed": processed,
            "moved_count": moved_count,
            "skipped": skipped,
            "failed": failed,
            "total_found": len(pdf_files),
            "collections_created": list(collections.keys()) if processed else []
        }


def main():
    """Main function for the PDF pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Run complete PDF processing pipeline")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to process (not needed for batch processing)")
    parser.add_argument("--output-dir", default="processed_output", 
                       help="Directory to save output files")
    parser.add_argument("--gpt-token", default="gpt_token.txt", 
                       help="Path to GPT token file for OpenAI API")
    parser.add_argument("--vector-store-collection", default=None,
                       help="ChromaDB collection name (auto-determined from subdirectory if not specified)")
    parser.add_argument("--vector-store-dir", default="./chroma_db",
                       help="Directory for ChromaDB persistence")
    parser.add_argument("--vector-store-only", action="store_true",
                       help="Skip saving intermediate JSON files")
    parser.add_argument("--query", help="Test query to run after processing")
    parser.add_argument("--n-results", type=int, default=5,
                       help="Number of query results to return")
    parser.add_argument("--collection-info", action="store_true",
                       help="Show information about the target collection before processing")
    parser.add_argument("--query-collection", 
                       help="Query the entire collection (all PDFs) instead of just the processed document")
    parser.add_argument("--batch-process", action="store_true",
                       help="Process all PDFs in the pdfs directory and subdirectories")
    parser.add_argument("--pdfs-dir", default="pdfs",
                       help="Directory containing PDFs for batch processing")
    parser.add_argument("--no-skip-existing", action="store_true",
                       help="Process all PDFs even if they might already exist in collections")
    parser.add_argument("--hard-reset", action="store_true",
                       help="Clear all collections before processing (WARNING: deletes all existing embeddings)")
    parser.add_argument("--figure-min-size", type=float, default=0.15,
                       help="Minimum fraction of page area for figure GPT analysis (0.0-1.0, default: 0.15)")
    parser.add_argument("--figure-max-count", type=int, default=20,
                       help="Maximum number of figures to process with GPT per document (default: 20)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch_process and not args.pdf_path:
        parser.error("Either provide a PDF file path or use --batch-process")
    
    try:
        # Handle batch processing
        if args.batch_process:
            if args.hard_reset:
                print("🔥 Starting HARD RESET batch processing...")
            else:
                print("🚀 Starting batch processing of PDFs directory...")
                
            batch_results = PDFPipelineOrchestrator.batch_process_pdfs_directory(
                pdfs_dir=args.pdfs_dir,
                output_dir=args.output_dir,
                gpt_token_path=args.gpt_token,
                vector_store_dir=args.vector_store_dir,
                vector_store_only=args.vector_store_only,
                skip_existing=not args.no_skip_existing,
                hard_reset=args.hard_reset,
                figure_analysis_min_size=args.figure_min_size,
                figure_analysis_max_count=args.figure_max_count
            )
            
            # Save batch results summary
            summary_path = Path(args.output_dir) / "batch_processing_summary.json"
            summary_path.parent.mkdir(exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 Batch processing summary saved to: {summary_path}")
            return 0
        
        # Single PDF processing (original functionality)
        # Initialize and run the pipeline
        orchestrator = PDFPipelineOrchestrator(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            gpt_token_path=args.gpt_token,
            vector_store_collection=args.vector_store_collection,
            vector_store_dir=args.vector_store_dir,
            keep_intermediate_files=not args.vector_store_only,
            figure_analysis_min_size=args.figure_min_size,
            figure_analysis_max_count=args.figure_max_count
        )
        
        # Show collection info if requested
        if args.collection_info:
            print("\nCollection Information (before processing):")
            print("-" * 50)
            try:
                info = PDFPipelineOrchestrator.get_collection_info(
                    args.vector_store_dir, 
                    orchestrator.vector_store_collection,  # Use the orchestrator's determined collection name
                    args.gpt_token
                )
                stats = info["collection_stats"]
                print(f"Collection: {info['collection_name']}")
                print(f"Total chunks: {stats['total_chunks']}")
                print(f"PDFs in collection: {stats['pdf_count']}")
                if info["available_pdfs"]:
                    print(f"Available PDFs: {', '.join(info['available_pdfs'])}")
                else:
                    print("Available PDFs: None (new collection)")
            except Exception as e:
                print(f"Could not access collection info: {e}")
        
        # Run the complete pipeline
        summary = orchestrator.run_complete_pipeline()
        
        # Save pipeline summary
        summary_path = Path(args.output_dir) / f"{Path(args.pdf_path).stem}_pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nPipeline summary saved to: {summary_path}")
        
        # Run test queries if provided
        if args.query:
            print(f"\nRunning test query on processed document: '{args.query}'")
            print("-" * 50)
            
            results = orchestrator.query_processed_document(args.query, args.n_results)
            
            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. {result['section_title']} (Page {result['page_number']}) - {result['pdf_filename']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
        
        if args.query_collection:
            print(f"\nRunning collection-wide query: '{args.query_collection}'")
            print("-" * 50)
            
            results = orchestrator.query_collection(args.query_collection, args.n_results)
            
            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. {result['section_title']} (Page {result['page_number']}) - {result['pdf_filename']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
        
        # Show final collection info
        final_info = PDFPipelineOrchestrator.get_collection_info(
            args.vector_store_dir, 
            orchestrator.vector_store_collection,  # Use the orchestrator's determined collection name
            args.gpt_token
        )
        final_stats = final_info["collection_stats"]
        print(f"\n📊 Final Collection Status:")
        print(f"   Collection: {orchestrator.vector_store_collection}")
        print(f"   Total chunks: {final_stats['total_chunks']}")
        print(f"   PDFs in collection: {final_stats['pdf_count']}")
        print(f"   Available PDFs: {', '.join(final_info['available_pdfs'])}")
        
        print(f"\n✅ Processing complete for: {args.pdf_path}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())