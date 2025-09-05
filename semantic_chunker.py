#!/usr/bin/env python3
"""
Semantic Chunker

Takes JSON output from reading_order_processor.py and creates semantic pages with
intelligent chunking. Each table/figure gets its own semantic page. Generates
summaries for figures using GPT-4 Vision. Creates hierarchical chunks with
child chunks (180-350 tokens) and parent chunks (800-1600 tokens).

Usage:
    python semantic_chunker.py reading_order.json -o semantic_chunks.json
"""

import json
import re
import base64
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from openai import OpenAI
import tiktoken
import fitz  # PyMuPDF
from PIL import Image
import traceback


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str  # "child" | "parent"
    content: str
    token_count: int
    section_title: Optional[str]
    page_number: int
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SemanticPage:
    page_id: str
    page_type: str  # "section" | "table" | "figure"
    title: str
    content: str
    child_chunks: List[Chunk]
    parent_chunks: List[Chunk]
    original_page_numbers: List[int]
    metadata: Optional[Dict[str, Any]] = None


class SemanticChunker:
    def __init__(self, reading_order_json_path: str, pdf_path: Optional[str] = None, gpt_token_path: str = "gpt_token.txt",
                 figure_analysis_min_size: float = 0.15, figure_analysis_max_count: int = 20):
        """
        Initialize SemanticChunker with smart figure processing limits.
        
        Args:
            reading_order_json_path: Path to reading order JSON file
            pdf_path: Optional path to PDF file for figure analysis
            gpt_token_path: Path to GPT token file
            figure_analysis_min_size: Minimum fraction of page area a figure must occupy (0.0-1.0)
            figure_analysis_max_count: Maximum number of figures to process with GPT per document
        """
        self.reading_order_json_path = Path(reading_order_json_path)
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.gpt_token_path = Path(gpt_token_path)
        self.content_blocks: List[Dict] = []
        self.semantic_pages: List[SemanticPage] = []
        self.table_of_contents: Optional[Dict[str, Any]] = None
        self.pdf_doc = None
        
        # Smart figure processing parameters
        self.figure_analysis_min_size = max(0.0, min(1.0, figure_analysis_min_size))  # Clamp to 0-1
        self.figure_analysis_max_count = max(0, figure_analysis_max_count)
        self.figures_processed_count = 0
        
        # Global counters for unique chunk IDs
        self.global_child_counter = 0
        self.global_parent_counter = 0
        
        # Create a unique prefix based on PDF filename for truly unique IDs across PDFs
        if self.pdf_path:
            # Use PDF stem (filename without extension) as prefix
            self.pdf_prefix = self.pdf_path.stem.replace(' ', '_').replace('-', '_')[:20]  # Limit length
        else:
            # Fallback to reading order filename
            self.pdf_prefix = self.reading_order_json_path.stem.replace('_reading_order', '')[:20]
        
        # Initialize OpenAI client
        try:
            with open(self.gpt_token_path, 'r') as f:
                api_key = f.readline().strip()
            self.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
            self.openai_client = None
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
        # Open PDF if path provided
        if self.pdf_path and self.pdf_path.exists():
            try:
                self.pdf_doc = fitz.open(str(self.pdf_path))
                print(f"Opened PDF: {self.pdf_path}")
            except Exception as e:
                print(f"Warning: Could not open PDF {self.pdf_path}: {e}")
                self.pdf_doc = None
    
    def _should_analyze_figure_with_gpt(self, figure_block: Dict) -> bool:
        """
        Determine if a figure should be analyzed with GPT based on size and count limits.
        
        Args:
            figure_block: Figure block data from reading order processor
            
        Returns:
            True if figure should be processed with GPT, False otherwise
        """
        # Check if we've reached the maximum count
        if self.figures_processed_count >= self.figure_analysis_max_count:
            return False
        
        # Check figure size if we have bbox information
        bbox_data = figure_block.get("bbox", {})
        if bbox_data and self.pdf_doc:
            try:
                # Get page dimensions
                page_number = figure_block.get("page_number", 1)
                pdf_page = self.pdf_doc[page_number - 1]  # 0-indexed
                page_width = pdf_page.rect.width
                page_height = pdf_page.rect.height
                page_area = page_width * page_height
                
                # Calculate figure area
                figure_width = bbox_data.get("x1", 0) - bbox_data.get("x0", 0)
                figure_height = bbox_data.get("y1", 0) - bbox_data.get("y0", 0)
                figure_area = abs(figure_width * figure_height)
                
                # Check if figure meets minimum size requirement
                if page_area > 0:
                    size_fraction = figure_area / page_area
                    if size_fraction < self.figure_analysis_min_size:
                        return False
                
            except Exception as e:
                # If we can't determine size, err on the side of processing
                print(f"Warning: Could not determine figure size for GPT analysis: {e}")
        
        return True
    
    def load_content_blocks(self) -> None:
        """Load content blocks from reading order processor output."""
        try:
            with open(self.reading_order_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.content_blocks = data.get("content_blocks", [])
            self.table_of_contents = data.get("table_of_contents")
            
            print(f"Loaded {len(self.content_blocks)} content blocks")
            if self.table_of_contents:
                toc_entries = len(self.table_of_contents.get("entries", []))
                print(f"Table of contents: {toc_entries} entries")
            
        except Exception as e:
            raise ValueError(f"Failed to load content blocks: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using GPT-4 tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def process(self) -> Dict[str, Any]:
        """Process content blocks into semantic pages with chunking."""
        self.load_content_blocks()
        
        # Group blocks into sections and standalone items
        sections = self._identify_sections()
        
        # Process each section/item into semantic pages
        for section in sections:
            if section["type"] == "section":
                semantic_page = self._create_section_page(section)
            elif section["type"] == "table":
                semantic_page = self._create_table_page(section)
            elif section["type"] == "figure":
                semantic_page = self._create_figure_page(section)
            else:
                continue  # Skip unknown types
            
            if semantic_page:
                self.semantic_pages.append(semantic_page)
        
        return self._to_dict()
    
    def _identify_sections(self) -> List[Dict[str, Any]]:
        """Identify sections, tables, and figures from content blocks with improved heading classification."""
        sections = []
        current_section = None
        
        for i, block in enumerate(self.content_blocks):
            block_type = block.get("block_type", "paragraph")
            
            if block_type == "heading":
                heading_content = block["content"].strip()
                
                # Check if this is a semantic heading that should start a new section
                if self._is_semantic_heading(block, i):
                    # Save current section if it exists
                    if current_section and current_section["blocks"]:
                        sections.append(current_section)
                    
                    # Get TOC information for this heading if available
                    toc_info = block.get("toc_association", {}) if block.get("toc_association", {}) is not None else {}
                    section_title = heading_content
                    
                    # Use TOC title if this is a TOC heading
                    if toc_info.get("is_toc_heading"):
                        section_title = toc_info.get("toc_title", section_title)
                    
                    # Start new section
                    current_section = {
                        "type": "section",
                        "title": section_title,
                        "blocks": [block],
                        "start_index": i,
                        "toc_info": toc_info if toc_info else None
                    }
                else:
                    # This is metadata/non-semantic heading - add to current section or create metadata section
                    if current_section is None:
                        # Create a metadata section to contain non-semantic headings
                        current_section = {
                            "type": "section",
                            "title": self._extract_document_title_from_metadata(i),
                            "blocks": [],
                            "start_index": i,
                            "toc_info": None
                        }
                    
                    # Add metadata heading to current section
                    current_section["blocks"].append(block)
                    
                    # If this block has TOC association and current section doesn't, update section TOC info
                    block_toc = block.get("toc_association")
                    if block_toc and not current_section.get("toc_info"):
                        current_section["toc_info"] = block_toc
                        # Also update the title if it's from TOC and more specific
                        if block_toc.get("toc_title") and (current_section.get("title") == "Untitled Section" or 
                                                           len(block_toc.get("toc_title", "")) > len(current_section.get("title", ""))):
                            current_section["title"] = block_toc["toc_title"]
                            print(f"Updated section title to: {block_toc['toc_title']}")
                    elif block_toc:
                        print(f"Block has TOC '{block_toc.get('toc_title')}' but section already has TOC info")
            
            elif block_type in ["table", "figure"]:
                # Save current section if it exists
                if current_section and current_section["blocks"]:
                    sections.append(current_section)
                    current_section = None
                
                # Create standalone page for table/figure
                sections.append({
                    "type": block_type,
                    "title": self._extract_title_from_block(block),
                    "blocks": [block],
                    "start_index": i,
                    "toc_info": block.get("toc_association")
                })
            
            else:
                # Add to current section or create section based on TOC
                if current_section is None:
                    # Try to determine section from TOC association
                    toc_info = block.get("toc_association", {})
                    section_title = "Untitled Section"
                    
                    if toc_info and toc_info.get("toc_title"):
                        section_title = toc_info["toc_title"]
                    
                    current_section = {
                        "type": "section",
                        "title": section_title,
                        "blocks": [],
                        "start_index": i,
                        "toc_info": toc_info if toc_info else None
                    }
                
                # STRICT TOC boundary enforcement: Never allow chunks to span multiple TOC sections
                block_toc = block.get("toc_association")
                current_toc = current_section.get("toc_info", {})
                
                # Check for TOC section boundary
                should_create_new_section = False
                
                if block_toc and current_toc:
                    # Both have TOC - check if different
                    if block_toc.get("toc_title") != current_toc.get("toc_title"):
                        should_create_new_section = True
                        reason = f"TOC change: '{current_toc.get('toc_title')}' → '{block_toc.get('toc_title')}'"
                elif block_toc and not current_toc:
                    # Block has TOC but current section doesn't - start new section with TOC
                    should_create_new_section = True
                    reason = f"First TOC block: '{block_toc.get('toc_title')}'"
                elif not block_toc and current_toc:
                    # Block doesn't have TOC but current section does - continue current section but warn
                    print(f"⚠️  Block without TOC in section '{current_toc.get('toc_title')}' - continuing current section")
                
                if should_create_new_section:
                    # Save current section
                    sections.append(current_section)
                    
                    # Create new section for this TOC boundary
                    current_section = {
                        "type": "section", 
                        "title": block_toc.get("toc_title", "Untitled Section"),
                        "blocks": [block],
                        "start_index": len(sections),
                        "toc_info": block_toc
                    }
                    print(f"✅ Created new section: {reason}")
                else:
                    current_section["blocks"].append(block)
                    
                    # If this block has TOC association and current section doesn't, update section TOC info
                    if block_toc and not current_section.get("toc_info"):
                        current_section["toc_info"] = block_toc
                        # Also update the title if it's from TOC and more specific
                        if block_toc.get("toc_title") and (current_section.get("title") == "Untitled Section" or 
                                                           len(block_toc.get("toc_title", "")) > len(current_section.get("title", ""))):
                            current_section["title"] = block_toc["toc_title"]
                            print(f"Updated section title to: {block_toc['toc_title']}")
        
        # Don't forget the last section
        if current_section and current_section["blocks"]:
            sections.append(current_section)
        
        return sections
    
    def _is_semantic_heading(self, block: Dict, block_index: int) -> bool:
        """Determine if a heading block represents a semantic section boundary."""
        content = block["content"].strip()
        metadata = block.get("metadata", {})
        
        # Check font size - semantic headings are usually larger
        avg_font_size = metadata.get("average_font_size", 0)
        
        # Get context of surrounding blocks for better classification
        surrounding_context = self._get_surrounding_context(block_index)
        
        # Patterns that indicate NON-semantic headings (metadata)
        metadata_patterns = [
            r'^[A-Z][a-z]+ Theory [A-Z][a-z]+ Prac$',  # Journal names like "Ethic Theory Moral Prac"
            r'^DOI\s+10\.',  # DOI references
            r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # Author names like "William MacAskill"
            r'^Accepted:.*\d{4}',  # Publication dates
            r'^.*Springer.*\d{4}$',  # Publisher info
            r'^Keywords?\s*$',  # Just "Keywords"
            r'^Abstract\s*$',  # Just "Abstract"
            r'^\w{1,3}\s*$',  # Very short fragments
            r'^\.\s*\w+$',  # Fragments starting with period
        ]
        
        # Check for metadata patterns
        for pattern in metadata_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Patterns that indicate SEMANTIC headings (real sections)
        semantic_patterns = [
            r'^\d+\s+',  # Numbered sections "1 Introduction", "2 Methods"
            r'^\d+\.\d+',  # Subsections "1.1", "2.3"
            r'^(Introduction|Conclusion|Discussion|Methods?|Results?|Analysis|Background)',  # Common section names
            r'^(Abstract|Summary)$',  # Document structure
            r'[A-Z][a-z]+.*[A-Z][a-z]+',  # Title case with multiple words (likely paper title)
        ]
        
        # Check for semantic patterns
        for pattern in semantic_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        # Font size heuristics
        if avg_font_size >= 12:  # Larger fonts are more likely to be section headings
            # But exclude very small fonts (likely metadata)
            if len(content) > 3 and not self._looks_like_metadata(content):
                return True
        
        # If we have substantial text content and it's not clearly metadata, treat as semantic
        if len(content) > 10 and not self._looks_like_metadata(content):
            # Check if this has paragraph content nearby (suggests it's a real section)
            if surrounding_context.get("has_nearby_paragraphs", False):
                return True
        
        # Default: if unsure, treat short/suspicious content as metadata
        return False
    
    def _looks_like_metadata(self, content: str) -> bool:
        """Check if content looks like document metadata rather than a section heading."""
        metadata_indicators = [
            len(content) < 4,  # Very short
            content.isupper(),  # All caps (often metadata)
            re.search(r'\d{4}', content),  # Contains year
            re.search(r'^\w+\s*$', content),  # Single word
            content.startswith('.'),  # Starts with period
            '@' in content or '#' in content,  # Contains special chars
            re.search(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', content),  # Name pattern
        ]
        
        return any(metadata_indicators)
    
    def _get_surrounding_context(self, block_index: int) -> Dict[str, Any]:
        """Get context about blocks surrounding the current one."""
        context = {
            "has_nearby_paragraphs": False,
            "next_block_type": None,
            "prev_block_type": None
        }
        
        # Check next few blocks for paragraphs
        for i in range(block_index + 1, min(block_index + 4, len(self.content_blocks))):
            if i < len(self.content_blocks):
                next_block = self.content_blocks[i]
                if next_block.get("block_type") == "paragraph":
                    context["has_nearby_paragraphs"] = True
                    if context["next_block_type"] is None:
                        context["next_block_type"] = "paragraph"
                    break
                elif next_block.get("block_type") != "heading":
                    if context["next_block_type"] is None:
                        context["next_block_type"] = next_block.get("block_type")
        
        # Check previous block
        if block_index > 0:
            prev_block = self.content_blocks[block_index - 1]
            context["prev_block_type"] = prev_block.get("block_type")
        
        return context
    
    def _extract_document_title_from_metadata(self, start_index: int) -> str:
        """Extract a meaningful document title from the initial metadata blocks."""
        # Look through the first several blocks for the paper title
        title_candidates = []
        
        for i in range(start_index, min(start_index + 10, len(self.content_blocks))):
            if i < len(self.content_blocks):
                block = self.content_blocks[i]
                if block.get("block_type") == "heading":
                    content = block["content"].strip()
                    
                    # Skip obvious metadata
                    if not self._looks_like_metadata(content) and len(content) > 10:
                        metadata = block.get("metadata", {})
                        font_size = metadata.get("average_font_size", 0)
                        
                        # Title likely has larger font and substantial content
                        title_candidates.append({
                            "content": content,
                            "font_size": font_size,
                            "length": len(content)
                        })
        
        # Pick the best title candidate
        if title_candidates:
            # Prefer longer titles with larger fonts
            best_candidate = max(title_candidates, 
                               key=lambda x: x["font_size"] * 0.5 + x["length"] * 0.1)
            return best_candidate["content"]
        
        return "Document Content"
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """Split a long paragraph into smaller logical segments."""
        if len(text) <= 800:
            return [text]
        
        # Look for sentence boundaries that could be paragraph breaks
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        paragraphs = []
        current_paragraph = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would make the paragraph too long, start a new one
            if current_length + len(sentence) > 600 and current_paragraph:
                # Look for natural paragraph break indicators in the sentence
                paragraph_starters = [
                    'However,', 'Moreover,', 'Furthermore,', 'Nevertheless,', 'Therefore,',
                    'In addition,', 'On the other hand,', 'For example,', 'Similarly,',
                    'In contrast,', 'As a result,', 'Consequently,', 'Meanwhile,',
                    'First,', 'Second,', 'Third,', 'Finally,', 'Another',
                    'The design argument', 'Design arguments', 'This argument'
                ]
                
                is_natural_break = any(sentence.startswith(starter) for starter in paragraph_starters)
                
                if is_natural_break or current_length > 400:
                    # End current paragraph
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [sentence]
                    current_length = len(sentence)
                else:
                    current_paragraph.append(sentence)
                    current_length += len(sentence)
            else:
                current_paragraph.append(sentence)
                current_length += len(sentence)
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Filter out very short segments
        filtered_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) > 50:  # Minimum meaningful length
                filtered_paragraphs.append(para.strip())
        
        return filtered_paragraphs if filtered_paragraphs else [text]
    
    def _extract_title_from_block(self, block: Dict) -> str:
        """Extract title from table/figure block."""
        metadata = block.get("metadata", {})
        
        # Try caption first
        caption = metadata.get("caption", "")
        if caption:
            # Extract title from caption (e.g., "Table 1: Sales Data" -> "Sales Data")
            title_match = re.search(r'^(?:Table|Figure|Fig\.?)\s*\d*:?\s*(.+)', caption, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
            return caption
        
        # Fallback titles
        if block.get("block_type") == "table":
            return f"Table (Page {block.get('page_number', '?')})"
        elif block.get("block_type") == "figure":
            return f"Figure (Page {block.get('page_number', '?')})"
        
        return "Untitled"
    
    def _create_section_page(self, section: Dict) -> SemanticPage:
        """Create a semantic page from a section with text blocks."""
        blocks = section["blocks"]
        
        # Combine all content with smart paragraph splitting and preserve bounding box mapping
        full_content = ""
        page_numbers = set()
        content_bbox_map = []  # List of (start_char, end_char, bbox_data) tuples
        
        for block in blocks:
            block_content = block["content"]
            content_start = len(full_content)
            
            # If this is a very long paragraph from PyPDF2, try to split it intelligently
            if (block.get("metadata", {}).get("source") == "pypdf2" and 
                len(block_content) > 800):
                # Split long PyPDF2 paragraphs into smaller segments
                split_content = self._split_long_paragraph(block_content)
                for segment in split_content:
                    segment_start = len(full_content)
                    if full_content:
                        full_content += "\n\n"
                        segment_start += 2  # Account for added newlines
                    full_content += segment
                    segment_end = len(full_content)
                    
                    # Store bounding box mapping for this segment
                    if "bbox" in block:
                        content_bbox_map.append({
                            "start_char": segment_start,
                            "end_char": segment_end,
                            "bbox": block["bbox"],
                            "page_number": block.get("page_number", 1)
                        })
            else:
                if full_content:
                    full_content += "\n\n"
                    content_start += 2  # Account for added newlines
                full_content += block_content
                content_end = len(full_content)
                
                # Store bounding box mapping for this block
                if "bbox" in block:
                    content_bbox_map.append({
                        "start_char": content_start,
                        "end_char": content_end,
                        "bbox": block["bbox"],
                        "page_number": block.get("page_number", 1)
                    })
            
            page_numbers.add(block.get("page_number", 1))
        
        # Create child chunks
        child_chunks = self._create_child_chunks(full_content, section["title"], list(page_numbers))
        
        # Create parent chunks from child chunks
        parent_chunks = self._create_parent_chunks(child_chunks, section["title"])
        
        page_id = f"section_{len(self.semantic_pages) + 1}"
        
        return SemanticPage(
            page_id=page_id,
            page_type="section",
            title=section["title"],
            content=full_content,
            child_chunks=child_chunks,
            parent_chunks=parent_chunks,
            original_page_numbers=sorted(page_numbers),
            metadata={
                "block_count": len(blocks),
                "block_types": [block.get("block_type") for block in blocks],
                "toc_info": section.get("toc_info"),
                "toc_level": section.get("toc_info", {}).get("toc_level", 0) if section.get("toc_info") else None
            }
        )
    
    def _create_table_page(self, section: Dict) -> SemanticPage:
        """Create a semantic page from a table block."""
        table_block = section["blocks"][0]
        
        # Get table content - prefer markdown format
        content = table_block.get("content", "")
        metadata = table_block.get("metadata", {})
        
        # Add caption if available
        caption = metadata.get("caption", "")
        if caption:
            content = f"{caption}\n\n{content}"
        
        page_numbers = [table_block.get("page_number", 1)]
        
        # Create child chunks (tables usually fit in one chunk)
        child_chunks = self._create_child_chunks(content, section["title"], page_numbers)
        
        # Parent chunk is the same as child for standalone tables
        parent_chunks = self._create_parent_chunks(child_chunks, section["title"])
        
        page_id = f"table_{len(self.semantic_pages) + 1}"
        
        return SemanticPage(
            page_id=page_id,
            page_type="table",
            title=section["title"],
            content=content,
            child_chunks=child_chunks,
            parent_chunks=parent_chunks,
            original_page_numbers=page_numbers,
            metadata={
                "row_count": metadata.get("row_count"),
                "column_count": metadata.get("column_count"),
                "caption": caption,
                "toc_info": section.get("toc_info")
            }
        )
    
    def _create_figure_page(self, section: Dict) -> SemanticPage:
        """Create a semantic page from a figure block with GPT-4 vision summary."""
        figure_block = section["blocks"][0]
        metadata = figure_block.get("metadata", {})
        
        # Start with caption if available
        content = metadata.get("caption", "")
        
        # Generate figure summary using GPT-4 Vision if available and meets criteria
        figure_summary = None
        should_analyze = self._should_analyze_figure_with_gpt(figure_block)
        
        if should_analyze:
            figure_summary = self._generate_figure_summary(figure_block)
            if figure_summary:
                self.figures_processed_count += 1
                if content:
                    content += "\n\nFigure Analysis:\n" + figure_summary
                else:
                    content = "Figure Analysis:\n" + figure_summary
        else:
            # Log why we're skipping this figure
            bbox_data = figure_block.get("bbox", {})
            if self.figures_processed_count >= self.figure_analysis_max_count:
                print(f"   ⏭️  Skipping figure GPT analysis (reached limit of {self.figure_analysis_max_count})")
            elif bbox_data and self.pdf_doc:
                try:
                    page_number = figure_block.get("page_number", 1)
                    pdf_page = self.pdf_doc[page_number - 1]
                    page_area = pdf_page.rect.width * pdf_page.rect.height
                    figure_width = bbox_data.get("x1", 0) - bbox_data.get("x0", 0)
                    figure_height = bbox_data.get("y1", 0) - bbox_data.get("y0", 0)
                    figure_area = abs(figure_width * figure_height)
                    if page_area > 0:
                        size_fraction = figure_area / page_area
                        print(f"   ⏭️  Skipping figure GPT analysis (size {size_fraction:.1%} < {self.figure_analysis_min_size:.1%} minimum)")
                except:
                    pass
        
        if not content:
            content = f"Figure on page {figure_block.get('page_number', '?')}"
        
        page_numbers = [figure_block.get("page_number", 1)]
        
        # Create child chunks
        child_chunks = self._create_child_chunks(content, section["title"], page_numbers)
        
        # Parent chunk is the same as child for standalone figures
        parent_chunks = self._create_parent_chunks(child_chunks, section["title"])
        
        page_id = f"figure_{len(self.semantic_pages) + 1}"
        
        return SemanticPage(
            page_id=page_id,
            page_type="figure",
            title=section["title"],
            content=content,
            child_chunks=child_chunks,
            parent_chunks=parent_chunks,
            original_page_numbers=page_numbers,
            metadata={
                "image_type": metadata.get("image_type"),
                "caption": metadata.get("caption"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "has_ai_summary": bool(figure_summary),
                "toc_info": section.get("toc_info")
            }
        )
    
    def _generate_figure_summary(self, figure_block: Dict) -> Optional[str]:
        """Generate a summary of a figure using GPT-4 Vision."""
        if not self.openai_client or not self.pdf_doc:
            return None
        
        try:
            # Extract image from PDF using bounding box
            image_base64 = self._extract_figure_image(figure_block)
            if not image_base64:
                return None
            
            # Get existing caption for context
            metadata = figure_block.get("metadata", {})
            caption = metadata.get("caption", "")
            
            # Prepare the prompt for GPT-4 Vision
            system_prompt = """You are analyzing a figure from a document. Provide a transcription of any text in the figure and a summary/description of any images within the figure."""
            
            user_prompt = "Analyze this figure and provide a detailed summary of what it shows."
            if caption:
                user_prompt += f" The figure caption is: '{caption}'"
            
            # Call GPT-4 Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Warning: Could not generate figure summary: {e}")
            return None
    
    def _extract_figure_image(self, figure_block: Dict) -> Optional[str]:
        """Extract figure image from PDF using bounding box coordinates."""
        if not self.pdf_doc:
            return None
        
        try:
            # Get page number and bounding box
            page_number = figure_block.get("page_number", 1)
            bbox_data = figure_block.get("bbox", {})
            
            if not bbox_data:
                return None
            
            # Convert to PyMuPDF page (0-indexed)
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(self.pdf_doc):
                return None
            
            page = self.pdf_doc[page_index]
            
            # Create clipping rectangle from bounding box
            clip_rect = fitz.Rect(
                bbox_data.get("x0", 0),
                bbox_data.get("y0", 0), 
                bbox_data.get("x1", 100),
                bbox_data.get("y1", 100)
            )
            
            # Add some padding around the figure
            padding = 10
            clip_rect = fitz.Rect(
                max(0, clip_rect.x0 - padding),
                max(0, clip_rect.y0 - padding),
                min(page.rect.width, clip_rect.x1 + padding),
                min(page.rect.height, clip_rect.y1 + padding)
            )
            
            # Render the clipped area as image
            # Use high DPI for better quality
            matrix = fitz.Matrix(2.0, 2.0)  # 2x scaling for better quality
            pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return img_base64
            
        except Exception as e:
            print(f"Warning: Could not extract figure image: {e}")
            return None
    
    def _create_child_chunks(self, content: str, section_title: str, page_numbers: List[int]) -> List[Chunk]:
        """Create child chunks of 180-350 tokens with 15-20% overlap."""
        child_chunks = []
        
        # Split content into sentences for better chunking boundaries
        sentences = self._split_into_sentences(content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        target_min_tokens = 180
        target_max_tokens = 350
        overlap_tokens = int(target_min_tokens * 0.175)  # 17.5% overlap (middle of 15-20%)
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if not sentence or not sentence.strip():
                i += 1
                continue
                
            sentence_tokens = self.count_tokens(sentence)
            if sentence_tokens is None:
                sentence_tokens = 0
            
            # Check if adding this sentence would exceed max tokens
            if current_tokens + sentence_tokens > target_max_tokens and current_chunk:
                # Create chunk if we have minimum content
                if current_tokens >= target_min_tokens:
                    self.global_child_counter += 1
                    chunk_id = f"child_{self.pdf_prefix}_{self.global_child_counter}"
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        chunk_type="child",
                        content=current_chunk.strip(),
                        token_count=current_tokens,
                        section_title=section_title,
                        page_number=page_numbers[0] if page_numbers else 1,
                        chunk_index=chunk_index,
                        metadata={"page_numbers": page_numbers}
                    )
                    child_chunks.append(chunk)
                    chunk_index += 1
                    
                    # Create overlap with previous chunk
                    overlap_content = self._get_overlap_content(current_chunk, overlap_tokens)
                    current_chunk = overlap_content
                    current_tokens = self.count_tokens(current_chunk) if current_chunk else 0
                else:
                    # Current chunk is too small, continue building
                    pass
            
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
            i += 1
        
        # Handle remaining content
        if current_chunk.strip() and current_tokens >= 50:  # Allow smaller final chunks
            self.global_child_counter += 1
            chunk_id = f"child_{self.pdf_prefix}_{self.global_child_counter}"
            chunk = Chunk(
                chunk_id=chunk_id,
                chunk_type="child",
                content=current_chunk.strip(),
                token_count=current_tokens,
                section_title=section_title,
                page_number=page_numbers[0] if page_numbers else 1,
                chunk_index=chunk_index,
                metadata={"page_numbers": page_numbers}
            )
            child_chunks.append(chunk)
        
        return child_chunks
    
    def _create_parent_chunks(self, child_chunks: List[Chunk], section_title: str) -> List[Chunk]:
        """Create parent chunks of 800-1600 tokens by grouping child chunks."""
        parent_chunks = []
        
        if not child_chunks:
            return parent_chunks
        
        target_min_tokens = 800
        target_max_tokens = 1600
        
        current_content = ""
        current_tokens = 0
        current_children = []
        chunk_index = 0
        
        for child_chunk in child_chunks:
            child_token_count = child_chunk.token_count if child_chunk.token_count is not None else 0
            
            # Check if adding this child would exceed max tokens
            if current_tokens + child_token_count > target_max_tokens and current_content:
                # Create parent chunk if we have minimum content
                if current_tokens >= target_min_tokens:
                    self.global_parent_counter += 1
                    chunk_id = f"parent_{self.pdf_prefix}_{self.global_parent_counter}"
                    
                    page_numbers = []
                    for child in current_children:
                        page_numbers.extend(child.metadata.get("page_numbers", [child.page_number]))
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        chunk_type="parent",
                        content=current_content.strip(),
                        token_count=current_tokens,
                        section_title=section_title,
                        page_number=min(page_numbers) if page_numbers else 1,
                        chunk_index=chunk_index,
                        metadata={
                            "child_chunks": [child.chunk_id for child in current_children],
                            "page_numbers": sorted(set(page_numbers))
                        }
                    )
                    parent_chunks.append(chunk)
                    chunk_index += 1
                    
                    # Reset for next parent chunk
                    current_content = ""
                    current_tokens = 0
                    current_children = []
            
            # Add child to current parent
            if current_content:
                current_content += "\n\n" + child_chunk.content
            else:
                current_content = child_chunk.content
            current_tokens += child_token_count
            current_children.append(child_chunk)
        
        # Handle remaining children
        if current_content.strip():
            self.global_parent_counter += 1
            chunk_id = f"parent_{self.pdf_prefix}_{self.global_parent_counter}"
            
            page_numbers = []
            for child in current_children:
                page_numbers.extend(child.metadata.get("page_numbers", [child.page_number]))
            
            chunk = Chunk(
                chunk_id=chunk_id,
                chunk_type="parent",
                content=current_content.strip(),
                token_count=current_tokens,
                section_title=section_title,
                page_number=min(page_numbers) if page_numbers else 1,
                chunk_index=chunk_index,
                metadata={
                    "child_chunks": [child.chunk_id for child in current_children],
                    "page_numbers": sorted(set(page_numbers))
                }
            )
            parent_chunks.append(chunk)
        
        return parent_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking boundaries."""
        # Simple sentence splitting - can be improved with more sophisticated methods
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_content(self, content: str, overlap_tokens: int) -> str:
        """Get the last portion of content for overlap with next chunk."""
        if overlap_tokens <= 0:
            return ""
        
        # Split into sentences and take from the end
        sentences = self._split_into_sentences(content)
        
        overlap_content = ""
        tokens_so_far = 0
        
        # Build overlap from the end
        for sentence in reversed(sentences):
            if not sentence or not sentence.strip():
                continue
                
            sentence_tokens = self.count_tokens(sentence)
            if sentence_tokens is None:
                sentence_tokens = 0
                
            if tokens_so_far + sentence_tokens <= overlap_tokens:
                if overlap_content:
                    overlap_content = sentence + " " + overlap_content
                else:
                    overlap_content = sentence
                tokens_so_far += sentence_tokens
            else:
                break
        
        return overlap_content
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert semantic pages to dictionary for JSON serialization."""
        result = {
            "chunking_info": {
                "total_semantic_pages": len(self.semantic_pages),
                "page_types": {
                    page_type: len([p for p in self.semantic_pages if p.page_type == page_type])
                    for page_type in ["section", "table", "figure"]
                },
                "total_child_chunks": sum(len(page.child_chunks) for page in self.semantic_pages),
                "total_parent_chunks": sum(len(page.parent_chunks) for page in self.semantic_pages),
                "toc_linked_pages": sum(1 for page in self.semantic_pages if page.metadata and page.metadata.get("toc_info"))
            },
            "semantic_pages": [asdict(page) for page in self.semantic_pages]
        }
        
        # Include TOC information if present
        if self.table_of_contents:
            result["table_of_contents"] = self.table_of_contents
        
        return result
    
    def save_json(self, output_path: str) -> None:
        """Save the semantic chunks to a JSON file."""
        semantic_data = self._to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_data, f, indent=2, ensure_ascii=False)
    
    def close(self):
        """Close the PDF document."""
        if self.pdf_doc:
            self.pdf_doc.close()


def main():
    """Example usage of the Semantic Chunker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create semantic chunks from reading order data")
    parser.add_argument("reading_order_json", help="Path to the reading order JSON file")
    parser.add_argument("--pdf", help="Path to the original PDF file (required for figure analysis)")
    parser.add_argument("-o", "--output", help="Output JSON file path", 
                       default="semantic_chunks.json")
    parser.add_argument("--gpt-token", help="Path to GPT token file", 
                       default="gpt_token.txt")
    
    args = parser.parse_args()
    
    try:
        # Process the reading order data
        chunker = SemanticChunker(args.reading_order_json, args.pdf, args.gpt_token)
        semantic_data = chunker.process()
        
        # Save to JSON
        chunker.save_json(args.output)
        chunker.close()
        
        print(f"Semantic chunking complete. Results saved to {args.output}")
        
        # Print summary
        chunking_info = semantic_data["chunking_info"]
        print(f"Summary:")
        print(f"  Semantic pages: {chunking_info['total_semantic_pages']}")
        print(f"  Page types: {chunking_info['page_types']}")
        print(f"  Child chunks: {chunking_info['total_child_chunks']}")
        print(f"  Parent chunks: {chunking_info['total_parent_chunks']}")
        
        # Show figure analysis info
        figures_with_ai = sum(1 for page in chunker.semantic_pages 
                             if page.page_type == "figure" and 
                             page.metadata and page.metadata.get("has_ai_summary"))
        total_figures = chunking_info['page_types'].get('figure', 0)
        print(f"  Figures with AI analysis: {figures_with_ai}/{total_figures}")
        
    except Exception as e:
        print(f"Error processing semantic chunks: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())