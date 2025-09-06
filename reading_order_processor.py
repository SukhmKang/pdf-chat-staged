#!/usr/bin/env python3
"""
Reading Order Processor

Takes JSON output from pdf_layout_parser.py and organizes content into reading order
with semantic block types. Merges text spans into paragraphs, removes headers/footers,
and sorts elements by their natural reading flow.

Usage:
    python reading_order_processor.py layout_output.json -o reading_order.json
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
from openai import OpenAI


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    def overlaps_vertically(self, other: 'BoundingBox', tolerance: float = 5.0) -> bool:
        """Check if two bounding boxes overlap vertically within tolerance."""
        return not (self.y1 + tolerance < other.y0 or other.y1 + tolerance < self.y0)


@dataclass
class ContentBlock:
    block_type: str  # "heading" | "paragraph" | "list" | "table" | "figure" | "footnote"
    content: str
    bbox: BoundingBox
    page_number: int
    metadata: Optional[Dict[str, Any]] = None
    toc_association: Optional[Dict[str, Any]] = None  # Link to TOC entry if applicable
    
    @property
    def reading_order_key(self) -> Tuple[int, float, float]:
        """Generate sorting key for reading order: (page, y-position, x-position)."""
        return (self.page_number, self.bbox.y0, self.bbox.x0)


class ReadingOrderProcessor:
    def __init__(self, layout_json_path: str):
        self.layout_json_path = Path(layout_json_path)
        self.layout_data: Dict[str, Any] = {}
        self.content_blocks: List[ContentBlock] = []
        self.table_of_contents: Optional[Dict[str, Any]] = None
        self.header_footer_patterns: Dict[str, List[Tuple[str, float]]] = {
            "headers": [],
            "footers": []
        }
    
    def load_layout_data(self) -> None:
        """Load and validate the layout JSON data."""
        try:
            with open(self.layout_json_path, 'r', encoding='utf-8') as f:
                self.layout_data = json.load(f)
            
            # Validate required structure
            if "pages" not in self.layout_data:
                raise ValueError("Invalid layout data: missing 'pages' field")
            
            # Extract table of contents if present
            self.table_of_contents = self.layout_data.get("table_of_contents")
            
            print(f"Loaded layout data for {len(self.layout_data['pages'])} pages")
            if self.table_of_contents:
                toc_entries = len(self.table_of_contents.get("entries", []))
                confidence = self.table_of_contents.get("confidence", 0)
                print(f"Table of contents detected: {toc_entries} entries (confidence: {confidence:.2f})")
            
        except Exception as e:
            raise ValueError(f"Failed to load layout data: {e}")
    
    def process(self) -> Dict[str, Any]:
        """Process the layout data and return organized content blocks."""
        self.load_layout_data()
        self._detect_headers_footers()
        
        # Process each page using hybrid approach
        for page_data in self.layout_data["pages"]:
            page_number = page_data["page_number"]
            
            # Get clean paragraphs from PyPDF2 extraction
            clean_paragraphs = page_data.get("clean_paragraphs", [])
            
            # Get layout elements (figures, tables, footnotes) from detailed parsing
            tables = self._process_tables(page_data.get("tables", []), page_number)
            figures = self._process_figures(page_data.get("figures", []), page_number)
            footnotes = self._process_footnotes(page_data.get("footnotes", []), page_number)
            
            # Merge clean paragraphs with layout elements using position-based insertion
            merged_blocks = self._merge_paragraphs_with_layout_elements(
                clean_paragraphs, tables + figures + footnotes, page_number
            )
            
            self.content_blocks.extend(merged_blocks)
        
        # Associate content blocks with TOC entries
        self._associate_blocks_with_toc()
        
        # Sort all blocks by reading order
        self.content_blocks.sort(key=lambda block: block.reading_order_key)
        
        return self._to_dict()
    
    def _detect_headers_footers(self) -> None:
        """Detect repeated text patterns that appear as headers/footers."""
        pages = self.layout_data["pages"]
        total_pages = len(pages)
        
        if total_pages < 2:
            return  # Need multiple pages to detect patterns
        
        # Collect text spans by y-position bands
        top_band_texts = defaultdict(list)  # Top 15% of page
        bottom_band_texts = defaultdict(list)  # Bottom 15% of page
        
        for page_data in pages:
            page_number = page_data["page_number"]
            text_spans = page_data.get("text_spans", [])
            
            if not text_spans:
                continue
            
            # Calculate page height from column geometry or estimate
            page_bbox = page_data.get("column_geometry", {}).get("page_bbox", {})
            page_height = page_bbox.get("y1", 792) - page_bbox.get("y0", 0)
            
            top_threshold = page_bbox.get("y0", 0) + page_height * 0.15
            bottom_threshold = page_bbox.get("y0", 0) + page_height * 0.85
            
            for span in text_spans:
                bbox = span["bbox"]
                text = span["text"].strip()
                
                if not text or len(text) < 3:  # Skip very short text
                    continue
                
                # Normalize text for pattern matching
                normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
                
                if bbox["y0"] < top_threshold:
                    # Top band (potential header)
                    y_band = round(bbox["y0"] / 5) * 5  # Group by 5-pixel bands
                    top_band_texts[y_band].append((normalized_text, page_number))
                
                elif bbox["y0"] > bottom_threshold:
                    # Bottom band (potential footer)
                    y_band = round(bbox["y0"] / 5) * 5
                    bottom_band_texts[y_band].append((normalized_text, page_number))
        
        # Find patterns that appear on >60% of pages
        min_appearances = max(2, int(total_pages * 0.6))
        
        # Process headers
        for y_band, texts in top_band_texts.items():
            text_counts = Counter(text for text, _ in texts)
            for text, count in text_counts.items():
                if count >= min_appearances:
                    self.header_footer_patterns["headers"].append((text, y_band))
        
        # Process footers
        for y_band, texts in bottom_band_texts.items():
            text_counts = Counter(text for text, _ in texts)
            for text, count in text_counts.items():
                if count >= min_appearances:
                    self.header_footer_patterns["footers"].append((text, y_band))
        
        print(f"Detected {len(self.header_footer_patterns['headers'])} header patterns")
        print(f"Detected {len(self.header_footer_patterns['footers'])} footer patterns")
    
    def _merge_paragraphs_with_layout_elements(self, clean_paragraphs: List[Dict], 
                                             layout_elements: List[ContentBlock], 
                                             page_number: int) -> List[ContentBlock]:
        """Merge clean paragraphs from PyPDF2 with layout elements using position-based insertion."""
        merged_blocks = []
        
        # Convert clean paragraphs to ContentBlocks
        paragraph_blocks = []
        for para_data in clean_paragraphs:
            # Create bounding box from estimated position
            estimated_y = para_data.get("estimated_y_position", 0)
            bbox = BoundingBox(
                x0=50,  # Estimated left margin
                y0=estimated_y,
                x1=550,  # Estimated right margin
                y1=estimated_y + 50  # Estimated height
            )
            
            # Determine block type automatically
            text = para_data.get("text", "")
            block_type = self._classify_paragraph_type(text)
            
            content_block = ContentBlock(
                block_type=block_type,
                content=text,
                bbox=bbox,
                page_number=page_number,
                metadata={
                    "paragraph_index": para_data.get("paragraph_index", 0),
                    "source": "pypdf2",
                    "estimated_position": True
                }
            )
            paragraph_blocks.append(content_block)
        
        # Combine and sort all blocks by Y position
        all_blocks = paragraph_blocks + layout_elements
        all_blocks.sort(key=lambda block: (block.page_number, block.bbox.y0))
        
        return all_blocks
    
    def _classify_paragraph_type(self, text: str) -> str:
        """Classify paragraph type based on content."""
        if not text or len(text.strip()) < 3:
            return "paragraph"
        
        # Check for heading patterns
        if self._is_heading(text, 12, False, 1):  # Use existing heading detection
            return "heading"
        
        # Check for list patterns
        if self._is_list_item(text):
            return "list"
        
        # Default to paragraph
        return "paragraph"
    
    def _filter_headers_footers(self, text_spans: List[Dict], page_number: int) -> List[Dict]:
        """Remove text spans that match detected header/footer patterns."""
        filtered_spans = []
        
        for span in text_spans:
            text = span["text"].strip()
            normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
            bbox = span["bbox"]
            y_band = round(bbox["y0"] / 5) * 5
            
            # Check if this span matches any header pattern
            is_header = any(
                pattern_text == normalized_text and abs(pattern_y - y_band) <= 10
                for pattern_text, pattern_y in self.header_footer_patterns["headers"]
            )
            
            # Check if this span matches any footer pattern
            is_footer = any(
                pattern_text == normalized_text and abs(pattern_y - y_band) <= 10
                for pattern_text, pattern_y in self.header_footer_patterns["footers"]
            )
            
            if not (is_header or is_footer):
                filtered_spans.append(span)
        
        return filtered_spans
    
    def _merge_text_spans_to_paragraphs(self, text_spans: List[Dict], page_number: int) -> List[ContentBlock]:
        """Merge text spans into coherent paragraphs based on positioning and formatting."""
        if not text_spans:
            return []
        
        # First, identify footnote text spans to handle them separately
        footnote_spans, regular_spans = self._separate_footnote_spans(text_spans, page_number)
        
        # Detect column structure for this page
        column_boundaries = self._detect_column_boundaries(regular_spans)
        
        # Group spans by columns first, then by lines within each column
        column_paragraphs = []
        for column_spans in self._group_spans_by_columns(regular_spans, column_boundaries):
            if column_spans:
                lines = self._group_spans_into_lines(column_spans)
                paragraphs = self._merge_lines_into_paragraphs(lines, page_number)
                column_paragraphs.extend(paragraphs)
        
        # Process footnotes separately
        if footnote_spans:
            footnote_lines = self._group_spans_into_lines(footnote_spans)
            footnote_blocks = self._merge_lines_into_paragraphs(footnote_lines, page_number, force_block_type="footnote")
            column_paragraphs.extend(footnote_blocks)
        
        return column_paragraphs
    
    def _separate_footnote_spans(self, text_spans: List[Dict], _page_number: int) -> Tuple[List[Dict], List[Dict]]:
        """Separate footnote spans from regular text spans."""
        footnote_spans = []
        regular_spans = []
        
        if not text_spans:
            return footnote_spans, regular_spans
        
        # Calculate page height and bottom threshold
        all_y_positions = [span["bbox"]["y1"] for span in text_spans]
        page_bottom = max(all_y_positions) if all_y_positions else 792
        page_top = min(span["bbox"]["y0"] for span in text_spans) if text_spans else 0
        page_height = page_bottom - page_top
        
        # Bottom 20% of page is potential footnote area
        footnote_threshold = page_bottom - (page_height * 0.2)
        
        for span in text_spans:
            span_y = span["bbox"]["y0"]
            span_text = span["text"].strip()
            font_size = span.get("font_size", 12)
            
            # Criteria for footnotes:
            # 1. In bottom 20% of page
            # 2. Small font size (< 10pt typically)
            # 3. Text pattern suggests footnote (starts with number/symbol)
            is_footnote = (
                span_y > footnote_threshold and
                font_size < 10 and
                (re.match(r'^\d+[\s\.]', span_text) or  # Numbered footnote
                 re.match(r'^[\*\†\‡]', span_text) or   # Symbol footnote
                 len([s for s in text_spans if s["bbox"]["y0"] > footnote_threshold]) > 1)  # Multiple spans in footnote area
            )
            
            if is_footnote:
                footnote_spans.append(span)
            else:
                regular_spans.append(span)
        
        return footnote_spans, regular_spans
    
    def _detect_column_boundaries(self, text_spans: List[Dict]) -> List[float]:
        """Detect column boundaries based on text span x-positions."""
        if not text_spans:
            return []
        
        # Collect all left edges of text spans
        left_edges = [span["bbox"]["x0"] for span in text_spans]
        left_edges.sort()
        
        # Find significant gaps that indicate column boundaries
        boundaries = []
        min_gap = 30  # Minimum gap between columns
        
        for i in range(1, len(left_edges)):
            gap = left_edges[i] - left_edges[i-1]
            if gap > min_gap:
                # Check if this gap appears consistently (multiple spans start after it)
                boundary_x = left_edges[i-1]
                spans_after_boundary = sum(1 for span in text_spans if span["bbox"]["x0"] > boundary_x + gap/2)
                
                if spans_after_boundary > 3:  # At least a few spans in the next column
                    boundaries.append(boundary_x + gap/2)
        
        return boundaries
    
    def _group_spans_by_columns(self, text_spans: List[Dict], column_boundaries: List[float]) -> List[List[Dict]]:
        """Group text spans by columns based on detected boundaries."""
        if not column_boundaries:
            return [text_spans]  # Single column
        
        columns = [[] for _ in range(len(column_boundaries) + 1)]
        
        for span in text_spans:
            span_x = span["bbox"]["x0"]
            
            # Find which column this span belongs to
            column_idx = 0
            for i, boundary in enumerate(column_boundaries):
                if span_x > boundary:
                    column_idx = i + 1
                else:
                    break
            
            columns[column_idx].append(span)
        
        # Remove empty columns
        return [col for col in columns if col]
    
    def _group_spans_into_lines(self, text_spans: List[Dict]) -> List[List[Dict]]:
        """Group text spans into lines based on y-coordinates."""
        if not text_spans:
            return []
        
        lines = []
        current_line = []
        current_y = None
        line_tolerance = 5  # pixels
        
        # Sort spans by y-coordinate, then x-coordinate
        sorted_spans = sorted(text_spans, key=lambda s: (s["bbox"]["y0"], s["bbox"]["x0"]))
        
        for span in sorted_spans:
            span_y = span["bbox"]["y0"]
            
            if current_y is None or abs(span_y - current_y) <= line_tolerance:
                # Same line
                current_line.append(span)
                current_y = span_y
            else:
                # New line
                if current_line:
                    lines.append(current_line)
                current_line = [span]
                current_y = span_y
        
        # Don't forget the last line
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _merge_lines_into_paragraphs(self, lines: List[List[Dict]], page_number: int, force_block_type: Optional[str] = None) -> List[ContentBlock]:
        """Merge lines into paragraphs with improved logic and fragment rejoining."""
        paragraphs = []
        current_paragraph_lines = []
        last_line_bottom = None
        last_line_text = ""
        
        for line in lines:
            # Calculate line properties
            line_top = min(span["bbox"]["y0"] for span in line)
            line_bottom = max(span["bbox"]["y1"] for span in line)
            line_text = " ".join(span["text"] for span in line).strip()
            
            if not line_text:
                continue
            
            # Check if this line should start a new paragraph
            should_start_new_paragraph = (
                last_line_bottom is None or  # First line
                line_top - last_line_bottom > 15 or  # Significant gap
                (self._is_likely_paragraph_start(line_text) and not self._is_sentence_continuation(last_line_text, line_text)) or  # Formatting suggests new paragraph but not mid-sentence
                self._is_footnote_line(line_text)  # Footnote pattern
            )
            
            # Special case: detect text fragments that are clearly sentence continuations
            if (not should_start_new_paragraph and 
                current_paragraph_lines and 
                self._is_sentence_continuation(last_line_text, line_text) and
                line_top - last_line_bottom <= 5):  # Very close vertically
                
                # This is likely a fragment that should be merged - don't start new paragraph
                should_start_new_paragraph = False
            
            if should_start_new_paragraph and current_paragraph_lines:
                # Create paragraph from accumulated lines
                paragraph = self._create_paragraph_block(current_paragraph_lines, page_number, force_block_type)
                if paragraph:
                    paragraphs.append(paragraph)
                current_paragraph_lines = []
            
            current_paragraph_lines.append(line)
            last_line_bottom = line_bottom
            last_line_text = line_text
        
        # Create the final paragraph
        if current_paragraph_lines:
            paragraph = self._create_paragraph_block(current_paragraph_lines, page_number, force_block_type)
            if paragraph:
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def _is_likely_paragraph_start(self, line_text: str) -> bool:
        """Determine if a line likely starts a new paragraph based on formatting."""
        # Common paragraph start indicators
        paragraph_indicators = [
            line_text[0].isupper() and len(line_text) > 20,  # Starts with capital and is substantial
            re.match(r'^\d+\.', line_text),  # Numbered list
            re.match(r'^[A-Z][a-z]+:', line_text),  # Section header
            re.match(r'^[•\-\*]', line_text),  # Bulleted list
        ]
        
        return any(paragraph_indicators)
    
    def _is_sentence_continuation(self, previous_text: str, current_text: str) -> bool:
        """Determine if current text is a continuation of the previous sentence."""
        if not previous_text or not current_text:
            return False
        
        # Check if previous text ends with incomplete sentence indicators
        incomplete_endings = [
            not previous_text.endswith(('.', '!', '?', ':', ';')),  # No sentence ending punctuation
            previous_text.endswith(','),  # Ends with comma
            previous_text.endswith(' and'),  # Ends with conjunction
            previous_text.endswith(' or'),
            previous_text.endswith(' but'),
            previous_text.endswith(' the'),  # Ends with article
            previous_text.endswith(' a'),
            previous_text.endswith(' an'),
            previous_text.endswith(' of'),  # Ends with preposition
            previous_text.endswith(' in'),
            previous_text.endswith(' to'),
            previous_text.endswith(' for'),
            previous_text.endswith(' with'),
        ]
        
        # Check if current text starts like a continuation
        continuation_starts = [
            current_text.startswith("'"),  # Starts with apostrophe (like "'s also common")
            current_text[0].islower(),  # Starts with lowercase letter
            current_text.startswith('and '),  # Starts with conjunction
            current_text.startswith('or '),
            current_text.startswith('but '),
            current_text.startswith('so '),
            current_text.startswith('however'),
            current_text.startswith('therefore'),
            current_text.startswith('thus'),
            not re.match(r'^[A-Z]', current_text),  # Doesn't start with capital letter
        ]
        
        # Also check for very short fragments that are likely continuations
        is_short_fragment = len(current_text.split()) <= 8 and len(current_text) < 50
        
        return (any(incomplete_endings) or any(continuation_starts) or is_short_fragment)
    
    def _is_footnote_line(self, line_text: str) -> bool:
        """Check if a line appears to be a footnote based on text patterns."""
        footnote_patterns = [
            re.match(r'^\d+[\s\.]', line_text),  # Numbered footnote
            re.match(r'^[\*\†\‡§¶]', line_text),  # Symbol footnote
            re.match(r'^[ivxlcdm]+[\s\.]', line_text, re.IGNORECASE),  # Roman numeral footnote
        ]
        return any(footnote_patterns)
    
    def _create_paragraph_block(self, lines: List[List[Dict]], page_number: int, force_block_type: Optional[str] = None) -> Optional[ContentBlock]:
        """Create a content block from multiple lines of text spans with automatic type detection."""
        if not lines:
            return None
        
        # Combine all text
        paragraph_text = ""
        all_spans = []
        
        for line in lines:
            line_text = " ".join(span["text"] for span in line).strip()
            if line_text:
                if paragraph_text:
                    paragraph_text += " "
                paragraph_text += line_text
            all_spans.extend(line)
        
        if not paragraph_text.strip() or len(paragraph_text.strip()) < 3:
            return None  # Skip very short text
        
        # Calculate combined bounding box
        min_x0 = min(span["bbox"]["x0"] for span in all_spans)
        min_y0 = min(span["bbox"]["y0"] for span in all_spans)
        max_x1 = max(span["bbox"]["x1"] for span in all_spans)
        max_y1 = max(span["bbox"]["y1"] for span in all_spans)
        
        bbox = BoundingBox(min_x0, min_y0, max_x1, max_y1)
        
        # Extract metadata about formatting
        font_sizes = [span["font_size"] for span in all_spans]
        font_names = [span["font_name"] for span in all_spans]
        font_weights = [span.get("font_weight", "normal") for span in all_spans]
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        dominant_font = Counter(font_names).most_common(1)[0][0] if font_names else None
        has_bold = any(weight == "bold" for weight in font_weights)
        
        # Detect block type if not forced
        if force_block_type:
            block_type = force_block_type
        else:
            block_type = self._detect_block_type(paragraph_text, all_spans, avg_font_size, has_bold)
        
        metadata = {
            "font_size_range": [min(font_sizes), max(font_sizes)] if font_sizes else [12, 12],
            "average_font_size": avg_font_size,
            "dominant_font": dominant_font,
            "has_bold": has_bold,
            "line_count": len(lines),
            "span_count": len(all_spans)
        }
        
        return ContentBlock(
            block_type=block_type,
            content=paragraph_text,
            bbox=bbox,
            page_number=page_number,
            metadata=metadata
        )
    
    def _detect_block_type(self, text: str, spans: List[Dict], avg_font_size: float, has_bold: bool) -> str:
        """Detect the semantic type of a text block."""
        text_stripped = text.strip()
        
        # Header detection
        if self._is_heading(text_stripped, avg_font_size, has_bold, len(spans)):
            return "heading"
        
        # List detection
        if self._is_list_item(text_stripped):
            return "list"
        
        # Default to paragraph
        return "paragraph"
    
    def _is_heading(self, text: str, avg_font_size: float, has_bold: bool, span_count: int) -> bool:
        """Detect if text is likely a heading with improved metadata filtering."""
        
        if len(text) > 100:
            return False
        
        # First, filter out obvious metadata patterns that should NOT be headings
        metadata_patterns = [
            r'^[A-Z][a-z]+ Theory [A-Z][a-z]+ Prac$',  # Journal names
            r'^DOI\s+10\.',  # DOI references
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Simple author names (2 words)
            r'^Accepted:.*\d{4}',  # Publication dates
            r'^.*Springer.*\d{4}$',  # Publisher info
            r'^Keywords?\s*$',  # Just "Keywords"
            r'^Abstract\s*$',  # Just "Abstract"  
            r'^\w{1,4}\s*$',  # Very short single words
            r'^\.\s*\w+',  # Starting with period
            r'^#\s*\w+',  # Starting with hash
            r'^\d{4}[\s\-]\d{4}$',  # Year ranges
            r'^[A-Z]{2,}[\s\-][A-Z]{2,}$',  # All caps abbreviations
        ]
        
        # If it matches metadata patterns, definitely not a heading
        for pattern in metadata_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Strong positive indicators for headings
        strong_heading_patterns = [
            re.match(r'^\d+\.?\s+[A-Z]', text),  # Numbered sections "1 Introduction" 
            re.match(r'^\d+\.\d+', text),  # Subsections "1.1", "2.3"
            re.match(r'^(Chapter|Section|Part|Appendix)\s+', text, re.IGNORECASE),  # Explicit sections
            re.match(r'^(Introduction|Conclusion|Discussion|Methods?|Results?|Analysis|Background)', text, re.IGNORECASE),  # Common academic sections
            re.match(r'^(Abstract|Summary|References|Bibliography|Acknowledgments)', text, re.IGNORECASE),  # Document structure
        ]
        
        # If it has strong heading patterns, likely a heading
        if any(strong_heading_patterns):
            return True
        
        # Font size criterion (much larger than typical body text)
        font_size_indicates_heading = avg_font_size >= 14
        
        # Formatting criterion (bold is a strong indicator)
        formatting_indicates_heading = has_bold
        
        # Text pattern criteria (more conservative)
        ends_without_period = not text.endswith('.')
        is_reasonable_length = 3 <= len(text.split()) <= 10  # Not too short, not too long
        has_multiple_words = len(text.split()) >= 2  # At least 2 words
        
        # Title case pattern for longer text (more likely to be actual titles)
        is_substantial_title = (text.istitle() or text.isupper()) and len(text) > 10
        
        # Combine criteria with higher threshold
        criteria_met = sum([
            font_size_indicates_heading * 2,  # Weight font size more heavily
            formatting_indicates_heading * 2,  # Weight bold more heavily  
            ends_without_period,
            is_reasonable_length,
            has_multiple_words,
            is_substantial_title,
            span_count <= 2  # Stricter span requirement
        ])
        
        # Require higher threshold (5 instead of 3) to be more conservative
        return criteria_met >= 5
    
    def _is_list_item(self, text: str) -> bool:
        """Detect if text is a list item."""
        list_patterns = [
            re.match(r'^\s*[•\-\*\+]\s+', text),  # Bullet points
            re.match(r'^\s*\d+[\.\)]\s+', text),  # Numbered lists
            re.match(r'^\s*[a-zA-Z][\.\)]\s+', text),  # Lettered lists
            re.match(r'^\s*[ivxlcdm]+[\.\)]\s+', text, re.IGNORECASE),  # Roman numeral lists
            re.match(r'^\s*\(\d+\)\s+', text),  # Parenthetical numbers
            re.match(r'^\s*\([a-zA-Z]\)\s+', text),  # Parenthetical letters
        ]
        
        return any(list_patterns)
    
    def _process_tables(self, tables: List[Dict], page_number: int) -> List[ContentBlock]:
        """Convert table data to content blocks."""
        content_blocks = []
        
        for table in tables:
            bbox = BoundingBox(**table["bbox"])
            
            # Use markdown representation for content
            content = table.get("markdown", "")
            if not content and table.get("grid_data"):
                # Fallback: create simple text representation
                grid = table["grid_data"]
                content = "\n".join(" | ".join(row) for row in grid)
            
            metadata = {
                "caption": table.get("caption"),
                "row_count": len(table.get("grid_data", [])),
                "column_count": len(table.get("grid_data", [[]])[0]) if table.get("grid_data") else 0,
                "grid_data": table.get("grid_data")
            }
            
            content_block = ContentBlock(
                block_type="table",
                content=content,
                bbox=bbox,
                page_number=page_number,
                metadata=metadata
            )
            content_blocks.append(content_block)
        
        return content_blocks
    
    def _process_figures(self, figures: List[Dict], page_number: int) -> List[ContentBlock]:
        """Convert figure data to content blocks."""
        content_blocks = []
        
        for figure in figures:
            bbox = BoundingBox(**figure["bbox"])
            
            # Use caption as content, or create descriptive text
            content = figure.get("caption", f"[Figure: {figure.get('image_type', 'unknown')} image]")
            
            metadata = {
                "image_type": figure.get("image_type"),
                "caption": figure.get("caption"),
                "width": bbox.width,
                "height": bbox.height
            }
            
            content_block = ContentBlock(
                block_type="figure",
                content=content,
                bbox=bbox,
                page_number=page_number,
                metadata=metadata
            )
            content_blocks.append(content_block)
        
        return content_blocks
    
    def _process_footnotes(self, footnotes: List[Dict], page_number: int) -> List[ContentBlock]:
        """Convert footnote data to content blocks."""
        content_blocks = []
        
        for footnote in footnotes:
            bbox = BoundingBox(**footnote["bbox"])
            
            content_block = ContentBlock(
                block_type="footnote",
                content=footnote["text"],
                bbox=bbox,
                page_number=page_number,
                metadata={
                    "reference_number": footnote.get("reference_number")
                }
            )
            content_blocks.append(content_block)
        
        return content_blocks
    
    def _associate_blocks_with_toc(self) -> None:
        """Associate content blocks with table of contents entries."""
        if not self.table_of_contents or not self.table_of_contents.get("entries"):
            return
        
        toc_entries = self.table_of_contents["entries"]
        
        # Calculate page number offset between TOC logical pages and physical PDF pages
        page_offset = self._calculate_page_offset(toc_entries)
        print("page_offset")
        print(page_offset)
        
        # Create a mapping of page numbers to TOC entries for quick lookup (using corrected pages)
        page_to_toc = {}
        for entry in toc_entries:
            logical_page = entry.get("page_number") or 0
            if logical_page > 0:
                # Convert logical page to physical page
                physical_page = logical_page + page_offset
                entry["_corrected_page"] = physical_page  # Store corrected page for reference
                
                if physical_page not in page_to_toc:
                    page_to_toc[physical_page] = []
                page_to_toc[physical_page].append(entry)
        
        print(f"Applied page offset of {page_offset} to TOC entries")
        
        # Associate content blocks with TOC entries
        for block in self.content_blocks:
            # Find the most relevant TOC entry for this block
            toc_entry = self._find_relevant_toc_entry(block, toc_entries, page_to_toc)
            
            if toc_entry:
                block.toc_association = {
                    "toc_title": toc_entry["title"],
                    "toc_level": toc_entry.get("level", 0),
                    "toc_page": toc_entry.get("page_number", 0),
                    "corrected_toc_page": toc_entry.get("_corrected_page", 0),
                    "is_toc_heading": self._is_toc_heading(block, toc_entry)
                }
    
    def _calculate_page_offset(self, toc_entries: List[Dict]) -> int:
        """Calculate offset between TOC logical page numbers and physical PDF pages.
        
        Uses GPT to identify the first page of the first TOC section in the document content,
        then calculates the offset between logical and physical page numbers.
        """

        if not toc_entries:
            print("No TOC entries available, using default offset: 0")
            return 0
            
        # Get the first TOC entry
        first_entry = toc_entries[0]
        first_title = first_entry["title"].strip()
        first_logical_page = first_entry.get("page_number", 0)
        
        if first_logical_page == 0:
            print("First TOC entry has no page number, using default offset: 0")
            return 0
            
        try:
            # Initialize OpenAI client
            with open("gpt_token.txt", "r") as f:
                api_key = f.readline().strip()
            client = OpenAI(api_key=api_key)
            
            # Collect content by page
            page_contents = {}
            for i, page_data in enumerate(self.layout_data["pages"]):
                page_number = page_data["page_number"]
                
                # Get clean paragraphs from PyPDF2 extraction
                clean_paragraphs = page_data.get("clean_paragraphs", [])
                page_num = i + 1

                page_contents[page_number] = list(map(lambda x: x['text'], clean_paragraphs))
            
            # Check each page individually with GPT (starting from page 1, up to page 20)
            for page_num in sorted(page_contents.keys())[:20]:
                page_text = "\n".join(page_contents[page_num])
                
                prompt = f"""
                I need to find where the section "{first_title}" appears in my PDF. I am providing you with one page of the PDF and want to determine if the section begins at this page.
                This section is listed as page {first_logical_page} in the Table of Contents.
                
                Here is the content from the page:
                {page_text}
                
                Does this page contain the start of the section "{first_title}"?
                Look for headings, section titles, or chapter beginnings that match this.

                Do not get confused with table of contents. The word(s) "${first_title}" might appear in the text if it is a table of contents, but that does not necessarily mean the section STARTS on the given page.
                Also do not get confused with the title page of the paper/book. A section name may appear on the title page, but that doesn't mean the section starts on that page.
                
                Respond with "YES" if this page contains the section, or "NO" if it doesn't.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                
                gpt_response = response.choices[0].message.content.strip().upper()
                
                if gpt_response == "YES":
                    calculated_offset = page_num - first_logical_page
                    print(f"GPT found '{first_title}' on page {page_num}: TOC page {first_logical_page} → actual page {page_num} (offset: {calculated_offset})")
                    return calculated_offset
            
            print(f"GPT could not find '{first_title}' in first 20 pages, using default offset: 0")
            return 0
                
        except Exception as e:
            print(f"Error using GPT for page offset calculation: {e}")
            print("Falling back to default offset: 0")
            return 0
    
    def _find_relevant_toc_entry(self, block: ContentBlock, toc_entries: List[Dict], 
                                page_to_toc: Dict[int, List[Dict]]) -> Optional[Dict]:
        """Find the most relevant TOC entry for a content block.
        
        Returns the TOC entry with the highest page number that's <= the block's page.
        If multiple TOC entries exist on the same page, uses GPT with full page context to determine which is most relevant.
        """
        block_page = block.page_number
        
        # Find all TOC entries with the highest page number that's <= block_page
        best_page = -1
        candidate_entries = []
        previous_page_entry = None
        
        # First pass: find the best page and track previous page entry
        for entry in toc_entries:
            entry_page = entry.get("_corrected_page") or entry.get("page_number") or 0
            
            if entry_page <= block_page:
                if entry_page > best_page:
                    best_page = entry_page
                    
        # Second pass: collect candidates and find previous page entry
        for entry in toc_entries:
            entry_page = entry.get("_corrected_page") or entry.get("page_number") or 0
            
            if entry_page == best_page:
                candidate_entries.append(entry)
            elif entry_page < block_page and entry_page != best_page:
                # Track the most recent entry before the current page (but not the entries starting on best_page)
                prev_entry_page = previous_page_entry.get("_corrected_page") or previous_page_entry.get("page_number") or 0 if previous_page_entry else -1
                
                if entry_page > prev_entry_page:
                    # New page that's more recent
                    previous_page_entry = entry
                elif entry_page == prev_entry_page:
                    # Same page - keep the later one in TOC order (since toc_entries is already sorted)
                    previous_page_entry = entry
        
        if not candidate_entries:
            return None
                
        # Only include previous entry if there are new sections starting on current page
        # AND the previous entry is not already in candidates
        if (previous_page_entry and 
            candidate_entries and  # Only if there are entries starting on current page
            best_page == block_page and  # Only if new sections start on this page
            previous_page_entry not in candidate_entries):
            candidate_entries.insert(0, previous_page_entry)  # Add at beginning as it's the continuing section
        
        # If only one candidate, return it
        if len(candidate_entries) == 1:
            return candidate_entries[0]
        
        # Multiple TOC entries possible - use GPT with full page context
        return self._select_toc_entry_with_page_context(block, candidate_entries)
    
    def _select_toc_entry_with_page_context(self, block: ContentBlock, 
                                          candidate_entries: List[Dict]) -> Optional[Dict]:
        """Use GPT to select the most relevant TOC entry using full page context."""
        try:
            # Initialize OpenAI client
            with open("gpt_token.txt", "r") as f:
                api_key = f.readline().strip()
            client = OpenAI(api_key=api_key)
            
            # Get the full page text from the stored layout data
            page_data = None
            for page in self.layout_data.get("pages", []):
                if page["page_number"] == block.page_number:
                    page_data = page
                    break
            
            if page_data and page_data.get("full_page_text"):
                full_page_text = page_data["full_page_text"]
                # Limit length for GPT processing
                if len(full_page_text) > 3000:
                    full_page_text = full_page_text[:3000] + "..."
            else:
                # Fallback: reconstruct from content blocks if full page text not available
                page_blocks = [b for b in self.content_blocks if b.page_number == block.page_number]
                page_blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))  # Sort by reading order
                full_page_text = "\n\n".join([
                    f"[{b.block_type.upper()}]: {b.content[:200]}" + ("..." if len(b.content) > 200 else "")
                    for b in page_blocks if b.content
                ])
            
            # Prepare TOC entries for this page using letters to avoid confusion with section numbers
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            toc_list = "\n".join([
                f"Option {option_letters[i]}: {entry['title']} (level {entry.get('level', 0)})"
                for i, entry in enumerate(candidate_entries[:len(option_letters)])
            ])
            
            # Get the target block content
            target_text = block.content[:300] if block.content else ""
            
            prompt = f"""I need to determine which section a specific text block belongs to on a page that contains multiple table of contents entries.

FULL PAGE CONTENT (in reading order):
{full_page_text}

TOC SECTION OPTIONS:
{toc_list}

TARGET TEXT BLOCK TO CLASSIFY:
[{block.block_type.upper()}]: {target_text}

Based on the full page context and the position of the target text block relative to the TOC section headings, which TOC section does this text block belong to? The text should belong to the section that appears before it in the reading order.

Respond with just the LETTER (A, B, C, etc.) of the most appropriate TOC section option."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            gpt_response = response.choices[0].message.content.strip().upper()
            
            # Parse the response letter
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            try:
                if gpt_response in option_letters:
                    selected_index = option_letters.index(gpt_response)
                    if selected_index < len(candidate_entries):
                        selected_entry = candidate_entries[selected_index]
                        print(f"GPT selected TOC entry '{selected_entry['title']}' (option {gpt_response}) for block on page {block.page_number}")
                        return selected_entry
                
                print(f"GPT returned invalid option '{gpt_response}', falling back to first candidate")
                return candidate_entries[0]
                
            except Exception:
                print(f"GPT returned unexpected response '{gpt_response}', falling back to first candidate")
                return candidate_entries[0]
                
        except Exception as e:
            print(f"Error using GPT for TOC selection: {e}")
            # Fallback: return the entry with the lowest level (most specific section)
            return min(candidate_entries, key=lambda e: e.get('level', 0))
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings."""
        # Normalize texts
        t1 = re.sub(r'[^\w\s]', '', text1.lower().strip())
        t2 = re.sub(r'[^\w\s]', '', text2.lower().strip())
        
        # Simple word overlap similarity
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_toc_heading(self, block: ContentBlock, toc_entry: Dict) -> bool:
        """Check if this block is the actual heading referenced in the TOC."""
        return (block.block_type == "heading" and 
                self._text_similarity(block.content, toc_entry["title"]) > 0.7)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert processed content to dictionary for JSON serialization."""
        result = {
            "document_info": self.layout_data.get("document_info", {}),
            "processing_info": {
                "total_blocks": len(self.content_blocks),
                "block_types": dict(Counter(block.block_type for block in self.content_blocks)),
                "headers_removed": len(self.header_footer_patterns["headers"]),
                "footers_removed": len(self.header_footer_patterns["footers"]),
                "toc_associations": sum(1 for block in self.content_blocks if block.toc_association)
            },
            "content_blocks": [asdict(block) for block in self.content_blocks]
        }
        
        # Include TOC information if present
        if self.table_of_contents:
            result["table_of_contents"] = self.table_of_contents
        
        return result
    
    def save_json(self, output_path: str) -> None:
        """Save the processed content to a JSON file."""
        processed_data = self._to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)


def main():
    """Example usage of the Reading Order Processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF layout data into reading order")
    parser.add_argument("layout_json", help="Path to the layout JSON file from pdf_layout_parser.py")
    parser.add_argument("-o", "--output", help="Output JSON file path", 
                       default="reading_order.json")
    
    args = parser.parse_args()
    
    try:
        # Process the layout data
        processor = ReadingOrderProcessor(args.layout_json)
        processed_data = processor.process()
        
        # Save to JSON
        processor.save_json(args.output)
        
        print(f"Reading order processing complete. Results saved to {args.output}")
        
        # Print summary
        processing_info = processed_data["processing_info"]
        print(f"Summary:")
        print(f"  Total blocks: {processing_info['total_blocks']}")
        print(f"  Block types: {processing_info['block_types']}")
        print(f"  Headers removed: {processing_info['headers_removed']}")
        print(f"  Footers removed: {processing_info['footers_removed']}")
        
    except Exception as e:
        print(f"Error processing layout data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())