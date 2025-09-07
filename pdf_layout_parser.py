#!/usr/bin/env python3
"""
PDF Layout Parser

Extracts layout blocks from PDF files including text spans, figures, tables, 
and footnotes with detailed positioning and formatting information.

Requirements:
- PyMuPDF (fitz): pip install PyMuPDF
- Optional: pdfplumber for table extraction: pip install pdfplumber
"""

import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
from openai import OpenAI
from tqdm import tqdm
try:
    import PyPDF2
    import logging
    # Suppress PyPDF2 warnings about unknown widths and other font parsing issues
    logging.getLogger("PyPDF2").setLevel(logging.ERROR)
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")


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


@dataclass
class TextSpan:
    text: str
    bbox: BoundingBox
    font_name: str
    font_size: float
    font_weight: str
    font_style: str
    line_order: int
    page_number: int


@dataclass
class Figure:
    bbox: BoundingBox
    caption: Optional[str]
    image_type: str
    page_number: int


@dataclass
class Table:
    bbox: BoundingBox
    grid_data: List[List[str]]
    markdown: str
    caption: Optional[str]
    page_number: int


@dataclass
class Footnote:
    text: str
    bbox: BoundingBox
    page_number: int
    reference_number: Optional[str]


@dataclass
class TOCEntry:
    title: str
    page_number: int
    level: int  # Indentation level (0 = main section, 1 = subsection, etc.)
    bbox: BoundingBox
    source_page: int  # Page where this TOC entry appears


@dataclass
class TableOfContents:
    entries: List[TOCEntry]
    detected_pages: List[int]  # Pages where TOC was detected
    confidence: float  # GPT confidence in TOC detection


@dataclass
class ColumnGeometry:
    column_count: int
    column_gaps: List[float]
    page_bbox: BoundingBox


@dataclass
class CleanParagraph:
    text: str
    estimated_y_position: float
    page_number: int
    paragraph_index: int


@dataclass
class PageLayout:
    page_number: int
    text_spans: List[TextSpan]
    figures: List[Figure]
    tables: List[Table]
    footnotes: List[Footnote]
    column_geometry: ColumnGeometry
    clean_paragraphs: List[CleanParagraph]  # From PyPDF2 extraction
    full_page_text: str  # Complete page text from PyPDF2 extract_text()


class PDFLayoutParser:
    def __init__(self, pdf_path: str, gpt_token_path: str = "gpt_token.txt"):
        self.pdf_path = Path(pdf_path)
        self.gpt_token_path = Path(gpt_token_path)
        self.doc = fitz.open(str(self.pdf_path))
        self.pages_layout: List[PageLayout] = []
        self.table_of_contents: Optional[TableOfContents] = None
        
        # Initialize PyPDF2 reader for clean text extraction
        self.pypdf_reader = None
        if PYPDF_AVAILABLE:
            try:
                with open(self.pdf_path, 'rb') as file:
                    self.pypdf_reader = PyPDF2.PdfReader(file)
                    # Store text for all pages
                    self.pypdf_pages_text = []
                    total_pages = len(self.pypdf_reader.pages)
                    print(f"📄 Extracting clean text from {total_pages} pages...")
                    
                    for page_num in tqdm(range(total_pages), desc="📝 Extracting text", unit="page", leave=False):
                        page_text = self.pypdf_reader.pages[page_num].extract_text()
                        self.pypdf_pages_text.append(page_text)
            except Exception as e:
                print(f"Warning: Could not initialize PyPDF2 reader: {e}")
                self.pypdf_reader = None
                self.pypdf_pages_text = []
        else:
            self.pypdf_pages_text = []
        
        # Initialize OpenAI client for TOC detection
        try:
            with open(self.gpt_token_path, 'r') as f:
                api_key = f.readline().strip()
            self.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client for TOC detection: {e}")
            self.openai_client = None
    
    def parse(self) -> Dict[str, Any]:
        """Parse the entire PDF and return structured layout data."""
        total_pages = len(self.doc)
        print(f"📄 Parsing {total_pages} pages...")
        
        # First pass: parse all pages with progress bar
        pbar = tqdm(range(total_pages), desc="📖 Parsing pages", unit="page")
        for page_num in pbar:
            page = self.doc[page_num]
            pbar.set_postfix({"page": f"{page_num + 1}/{total_pages}"})
            page_layout = self._parse_page(page, page_num + 1)
            self.pages_layout.append(page_layout)
        
        # Second pass: detect table of contents
        print("🔍 Detecting table of contents...")
        self.table_of_contents = self._detect_table_of_contents()
        
        # Print parsing summary
        total_text_spans = sum(len(page.text_spans) for page in self.pages_layout)
        total_figures = sum(len(page.figures) for page in self.pages_layout)
        total_tables = sum(len(page.tables) for page in self.pages_layout)
        total_footnotes = sum(len(page.footnotes) for page in self.pages_layout)
        
        print(f"✅ Parsing complete:")
        print(f"   📝 Text spans: {total_text_spans:,}")
        print(f"   🖼️  Figures: {total_figures}")
        print(f"   📊 Tables: {total_tables}")
        print(f"   📋 Footnotes: {total_footnotes}")
        
        if self.table_of_contents:
            print(f"   📚 TOC entries: {len(self.table_of_contents.entries)}")
        else:
            print("   📚 No TOC detected")
        
        return self._to_dict()
    
    def _parse_page(self, page: fitz.Page, page_number: int) -> PageLayout:
        """Parse a single page and extract all layout elements."""
        # Get page dimensions
        page_bbox = BoundingBox(*page.rect)
        
        # Extract text spans with detailed formatting
        text_spans = self._extract_text_spans(page, page_number)
        
        # Detect column geometry
        column_geometry = self._detect_column_geometry(page, text_spans, page_bbox)
        
        # Extract figures/images
        figures = self._extract_figures(page, page_number)
        
        # Extract tables
        tables = self._extract_tables(page, page_number)
        
        # Extract footnotes
        footnotes = self._extract_footnotes(page, page_number, text_spans)
        
        # Extract clean paragraphs from PyPDF2
        clean_paragraphs = self._extract_clean_paragraphs(page_number)
        
        # Get full page text from PyPDF2 (already extracted during initialization)
        full_page_text = ""
        if self.pypdf_pages_text and len(self.pypdf_pages_text) >= page_number:
            full_page_text = self.pypdf_pages_text[page_number - 1]  # Convert to 0-based index
        
        return PageLayout(
            page_number=page_number,
            text_spans=text_spans,
            figures=figures,
            tables=tables,
            footnotes=footnotes,
            column_geometry=column_geometry,
            clean_paragraphs=clean_paragraphs,
            full_page_text=full_page_text
        )
    
    def _extract_text_spans(self, page: fitz.Page, page_number: int) -> List[TextSpan]:
        """Extract text spans with detailed font and positioning information."""
        text_spans = []
        blocks = page.get_text("dict")
        
        line_order = 0
        for block in blocks["blocks"]:
            if block.get("type") == 0:  # Text block
                for line in block["lines"]:
                    line_order += 1
                    for span in line["spans"]:
                        # Skip spans with None or empty text
                        span_text = span.get("text", "")
                        if span_text is None:
                            span_text = ""
                        
                        bbox = BoundingBox(*span["bbox"])
                        
                        # Extract font information
                        font_name = span.get("font", "Unknown")
                        font_size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        
                        # Determine font weight and style from flags
                        font_weight = "bold" if flags & 2**4 else "normal"
                        font_style = "italic" if flags & 2**1 else "normal"
                        
                        text_span = TextSpan(
                            text=span_text,
                            bbox=bbox,
                            font_name=font_name,
                            font_size=font_size,
                            font_weight=font_weight,
                            font_style=font_style,
                            line_order=line_order,
                            page_number=page_number
                        )
                        text_spans.append(text_span)
        
        return text_spans
    
    def _extract_figures(self, page: fitz.Page, page_number: int) -> List[Figure]:
        """Extract figures/images with captions."""
        figures = []
        
        try:
            image_list = page.get_images(full=True)
            
            for img in image_list:
                try:
                    # Get image reference and bounding box
                    xref = img[0]
                    
                    # Get all instances of this image on the page
                    img_rects = page.get_image_rects(xref)
                    
                    if img_rects:
                        # Use the first rectangle if multiple instances
                        img_rect = img_rects[0]
                        bbox = BoundingBox(*img_rect)
                        
                        # Try to find caption near the image
                        caption = self._find_caption_near_bbox(page, bbox)
                        
                        # Get image type
                        try:
                            base_image = self.doc.extract_image(xref)
                            image_type = base_image["ext"]
                        except:
                            image_type = "unknown"
                        
                        figure = Figure(
                            bbox=bbox,
                            caption=caption,
                            image_type=image_type,
                            page_number=page_number
                        )
                        figures.append(figure)
                        
                except Exception as e:
                    # Skip problematic images but continue processing
                    print(f"Warning: Could not process image on page {page_number}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not extract images from page {page_number}: {e}")
        
        return figures
    
    def _extract_tables(self, page: fitz.Page, page_number: int) -> List[Table]:
        """Extract tables with grid data and markdown conversion."""
        tables = []
        
        # Try PyMuPDF's built-in table detection first
        try:
            detected_tables = page.find_tables()
            for table_obj in detected_tables:
                try:
                    # Extract table data using PyMuPDF's table extraction
                    table_data = table_obj.extract()
                    
                    if table_data and len(table_data) > 1:  # Must have header + data rows
                        # Clean the data (remove empty rows/columns)
                        cleaned_data = self._clean_table_data(table_data)
                        
                        if cleaned_data:
                            # Convert to markdown
                            markdown = self._grid_to_markdown(cleaned_data)
                            
                            # Find caption
                            bbox = BoundingBox(*table_obj.bbox)
                            caption = self._find_caption_near_bbox(page, bbox, search_above=True)
                            
                            table = Table(
                                bbox=bbox,
                                grid_data=cleaned_data,
                                markdown=markdown,
                                caption=caption,
                                page_number=page_number
                            )
                            tables.append(table)
                            
                except Exception as e:
                    print(f"Warning: Could not extract table data on page {page_number}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: PyMuPDF table detection not available or failed on page {page_number}: {e}")
            # Fallback to manual table detection
            tables.extend(self._extract_tables_fallback(page, page_number))
        
        return tables
    
    def _extract_tables_fallback(self, page: fitz.Page, page_number: int) -> List[Table]:
        """Fallback table extraction using text positioning analysis."""
        tables = []
        
        # Find table-like structures using text positioning
        table_regions = self._detect_table_regions(page)
        
        for region_bbox in table_regions:
            # Extract text within the table region with detailed positioning
            grid_data = self._extract_table_cells_by_position(page, region_bbox)
            
            if len(grid_data) > 1:  # Must have at least header + one row
                # Convert to markdown
                markdown = self._grid_to_markdown(grid_data)
                
                # Find caption
                bbox = BoundingBox(*region_bbox)
                caption = self._find_caption_near_bbox(page, bbox, search_above=True)
                
                table = Table(
                    bbox=bbox,
                    grid_data=grid_data,
                    markdown=markdown,
                    caption=caption,
                    page_number=page_number
                )
                tables.append(table)
        
        return tables
    
    def _extract_footnotes(self, page: fitz.Page, page_number: int, text_spans: List[TextSpan]) -> List[Footnote]:
        """Extract footnotes from the bottom of the page."""
        footnotes = []
        page_height = page.rect.height
        
        # Look for text in the bottom 20% of the page with small font size
        bottom_threshold = page_height * 0.8
        
        footnote_spans = [
            span for span in text_spans 
            if span.bbox.y0 > bottom_threshold and span.font_size < 10
        ]
        
        # Group consecutive footnote spans
        current_footnote_text = ""
        current_bbox = None
        reference_number = None
        
        for span in footnote_spans:
            # Skip spans with None text
            if not span.text:
                continue
                
            # Check if this looks like a footnote reference number
            if re.match(r'^\d+\s', span.text):
                # Save previous footnote if exists
                if current_footnote_text and current_bbox:
                    footnote = Footnote(
                        text=current_footnote_text.strip(),
                        bbox=current_bbox,
                        page_number=page_number,
                        reference_number=reference_number
                    )
                    footnotes.append(footnote)
                
                # Start new footnote
                current_footnote_text = span.text
                current_bbox = span.bbox
                reference_number = re.match(r'^(\d+)', span.text).group(1)
            else:
                # Continue current footnote
                if current_footnote_text:
                    current_footnote_text += " " + span.text
                    # Expand bounding box
                    if current_bbox:
                        current_bbox = BoundingBox(
                            min(current_bbox.x0, span.bbox.x0),
                            min(current_bbox.y0, span.bbox.y0),
                            max(current_bbox.x1, span.bbox.x1),
                            max(current_bbox.y1, span.bbox.y1)
                        )
        
        # Don't forget the last footnote
        if current_footnote_text and current_bbox:
            footnote = Footnote(
                text=current_footnote_text.strip(),
                bbox=current_bbox,
                page_number=page_number,
                reference_number=reference_number
            )
            footnotes.append(footnote)
        
        return footnotes
    
    def _extract_clean_paragraphs(self, page_number: int) -> List[CleanParagraph]:
        """Extract clean paragraphs from PyPDF2 with estimated positions."""
        clean_paragraphs = []
        
        if not self.pypdf_pages_text or page_number > len(self.pypdf_pages_text):
            return clean_paragraphs
        
        # Get clean text for this page (page_number is 1-indexed)
        page_text = self.pypdf_pages_text[page_number - 1]
        
        if not page_text.strip():
            return clean_paragraphs
        
        # Split into paragraphs - look for double newlines, or single newlines with indentation
        paragraphs = self._split_text_into_paragraphs(page_text)
        
        # Estimate Y positions for paragraphs by distributing them across the page
        page_height = 792  # Standard page height in points
        header_margin = 72   # Top margin
        footer_margin = 72   # Bottom margin
        usable_height = page_height - header_margin - footer_margin
        
        if len(paragraphs) > 1:
            y_step = usable_height / len(paragraphs)
        else:
            y_step = usable_height / 2
        
        for i, paragraph_text in enumerate(paragraphs):
            if paragraph_text.strip():
                # Estimate Y position (higher Y values are lower on page in PDF coordinates)
                estimated_y = header_margin + (i * y_step)
                
                clean_paragraph = CleanParagraph(
                    text=paragraph_text.strip(),
                    estimated_y_position=estimated_y,
                    page_number=page_number,
                    paragraph_index=i
                )
                clean_paragraphs.append(clean_paragraph)
        
        return clean_paragraphs
    
    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using improved heuristics."""
        if not text.strip():
            return []
        
        # Step 1: Try double newlines first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Step 2: If we get good segmentation, use it
        if len(paragraphs) >= 2:
            # Filter out very short segments that are likely headers/artifacts
            good_paragraphs = []
            for para in paragraphs:
                if len(para) > 50 or (len(para) > 20 and para.count('.') >= 1):
                    good_paragraphs.append(para)
            
            if good_paragraphs:
                return good_paragraphs
        
        # Step 3: More sophisticated line-by-line analysis
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            # Criteria for starting a new paragraph
            is_new_paragraph = False
            
            if current_paragraph:  # Not the first line
                # Check for paragraph break indicators
                prev_line = current_paragraph[-1] if current_paragraph else ""
                
                # Strong indicators of new paragraph
                if (
                    # Previous line ends with period and current starts with capital
                    (prev_line.endswith('.') and line[0].isupper()) or
                    # Current line starts with indentation spaces
                    line.startswith('    ') or
                    # Current line starts with paragraph markers
                    line.startswith('  ') and line[2].isupper() or
                    # Line starts with common paragraph starters
                    any(line.startswith(starter) for starter in [
                        'However,', 'Therefore,', 'Moreover,', 'Furthermore,', 'Nevertheless,',
                        'On the other hand,', 'In contrast,', 'Similarly,', 'For example,',
                        'In conclusion,', 'To summarize,', 'First,', 'Second,', 'Third,',
                        'Finally,', 'Another', 'The ', 'This ', 'That ', 'These ', 'Those '
                    ]) and prev_line.endswith('.') or
                    # Current paragraph is getting very long (likely multiple paragraphs)
                    len(' '.join(current_paragraph)) > 800
                ):
                    is_new_paragraph = True
            
            if is_new_paragraph and current_paragraph:
                # End current paragraph
                para_text = ' '.join(current_paragraph).strip()
                if para_text and len(para_text) > 20:  # Minimum paragraph length
                    paragraphs.append(para_text)
                current_paragraph = [line]
            else:
                current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph).strip()
            if para_text and len(para_text) > 20:
                paragraphs.append(para_text)
        
        # Step 4: Post-process to merge very short paragraphs with adjacent ones
        if len(paragraphs) > 1:
            merged_paragraphs = []
            for i, para in enumerate(paragraphs):
                if len(para) < 100 and i > 0 and len(merged_paragraphs[-1]) < 400:
                    # Merge short paragraph with previous one
                    merged_paragraphs[-1] += ' ' + para
                else:
                    merged_paragraphs.append(para)
            paragraphs = merged_paragraphs
        
        return paragraphs
    
    def _detect_column_geometry(self, _page: fitz.Page, text_spans: List[TextSpan], page_bbox: BoundingBox) -> ColumnGeometry:
        """Detect single vs multi-column layout."""
        if not text_spans:
            return ColumnGeometry(1, [], page_bbox)
        
        # Analyze x-coordinates of text spans to detect columns
        x_positions = [span.bbox.x0 for span in text_spans]
        x_positions.sort()
        
        # Find significant gaps that might indicate column boundaries
        gaps = []
        min_gap_size = 20  # Minimum gap size to consider as column separator
        
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > min_gap_size:
                gaps.append(gap)
        
        # Estimate column count based on major gaps
        if not gaps:
            column_count = 1
            column_gaps = []
        else:
            # Simple heuristic: if we have significant gaps, we likely have multiple columns
            major_gaps = [gap for gap in gaps if gap > 50]
            column_count = len(major_gaps) + 1 if major_gaps else 1
            column_gaps = major_gaps
        
        return ColumnGeometry(column_count, column_gaps, page_bbox)
    
    def _detect_table_regions(self, page: fitz.Page) -> List[fitz.Rect]:
        """Detect rectangular regions that likely contain tables."""
        table_regions = []
        
        # Use PyMuPDF's table detection if available
        try:
            tables = page.find_tables()
            for table in tables:
                table_regions.append(table.bbox)
        except:
            # Fallback: look for patterns of aligned text that suggest tables
            blocks = page.get_text("dict")
            
            # This is a simplified table detection - in practice, you might want
            # to use more sophisticated algorithms
            for block in blocks["blocks"]:
                if block.get("type") == 0:  # Text block
                    lines = block["lines"]
                    if len(lines) > 2:  # Potential table with multiple rows
                        # Check if lines are regularly spaced (suggesting table structure)
                        y_positions = [line["bbox"][1] for line in lines]
                        if self._has_regular_spacing(y_positions):
                            table_regions.append(fitz.Rect(block["bbox"]))
        
        return table_regions
    
    def _has_regular_spacing(self, positions: List[float], tolerance: float = 5.0) -> bool:
        """Check if positions have regular spacing (indicating table rows)."""
        if len(positions) < 3:
            return False
        
        differences = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_diff = sum(differences) / len(differences)
        
        # Check if all differences are close to the average
        return all(abs(diff - avg_diff) <= tolerance for diff in differences)
    
    def _text_to_grid(self, text_dict: Dict, _region_bbox: fitz.Rect) -> List[List[str]]:
        """Convert text within a region to a grid structure."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated table parsing
        lines = []
        
        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    if line_text.strip():
                        lines.append(line_text.strip())
        
        # Simple grid creation - split lines by whitespace
        grid = []
        for line in lines:
            # Split by multiple spaces (indicating column separation)
            columns = re.split(r'\s{2,}', line)
            grid.append(columns)
        
        return grid
    
    def _clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """Clean table data by removing empty rows and columns."""
        if not table_data:
            return []
        
        # First, convert None values to empty strings and ensure all cells are strings
        sanitized_data = []
        for row in table_data:
            if row is None:
                continue
            sanitized_row = []
            for cell in row:
                if cell is None:
                    sanitized_row.append("")
                elif isinstance(cell, str):
                    sanitized_row.append(cell)
                else:
                    sanitized_row.append(str(cell))
            sanitized_data.append(sanitized_row)
        
        if not sanitized_data:
            return []
        
        # Remove completely empty rows
        cleaned_rows = []
        for row in sanitized_data:
            if any(cell and cell.strip() for cell in row):
                cleaned_rows.append(row)
        
        if not cleaned_rows:
            return []
        
        # Remove completely empty columns
        max_cols = max(len(row) for row in cleaned_rows)
        
        # Pad all rows to same length
        for row in cleaned_rows:
            while len(row) < max_cols:
                row.append("")
        
        # Find non-empty columns
        non_empty_cols = []
        for col_idx in range(max_cols):
            if any(row[col_idx] and row[col_idx].strip() for row in cleaned_rows):
                non_empty_cols.append(col_idx)
        
        # Keep only non-empty columns
        if non_empty_cols:
            final_data = []
            for row in cleaned_rows:
                cleaned_row = [row[col_idx] for col_idx in non_empty_cols]
                final_data.append(cleaned_row)
            return final_data
        
        return cleaned_rows
    
    def _extract_table_cells_by_position(self, page: fitz.Page, region_bbox: fitz.Rect) -> List[List[str]]:
        """Extract table cells by analyzing text positioning within a region."""
        # Get all text spans within the table region
        text_dict = page.get_text("dict", clip=region_bbox)
        
        # Collect all text spans with their positions
        spans_with_pos = []
        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span.get("text", "")
                        if span_text and span_text.strip():
                            spans_with_pos.append({
                                "text": span_text.strip(),
                                "x0": span["bbox"][0],
                                "y0": span["bbox"][1],
                                "x1": span["bbox"][2],
                                "y1": span["bbox"][3]
                            })
        
        if not spans_with_pos:
            return []
        
        # Group spans by rows (similar y-coordinates)
        tolerance = 5  # pixels
        rows = []
        
        for span in spans_with_pos:
            span_y = span["y0"]
            
            # Find existing row with similar y-coordinate
            row_found = False
            for row in rows:
                if any(abs(existing_span["y0"] - span_y) <= tolerance for existing_span in row):
                    row.append(span)
                    row_found = True
                    break
            
            # Create new row if no matching row found
            if not row_found:
                rows.append([span])
        
        # Sort rows by y-coordinate (top to bottom)
        rows.sort(key=lambda row: min(span["y0"] for span in row))
        
        # For each row, sort spans by x-coordinate (left to right) and group into columns
        grid = []
        for row in rows:
            row.sort(key=lambda span: span["x0"])
            
            # Group spans into columns based on x-position gaps
            columns = []
            current_column_text = ""
            last_x1 = None
            
            for span in row:
                # Skip spans with None text
                span_text = span.get("text", "")
                if span_text is None:
                    span_text = ""
                
                # If there's a significant gap, start a new column
                if last_x1 is not None and span["x0"] - last_x1 > 20:  # 20 pixel gap threshold
                    if current_column_text:
                        columns.append(current_column_text.strip())
                    current_column_text = span_text
                else:
                    # Continue current column
                    if current_column_text:
                        current_column_text += " " + span_text
                    else:
                        current_column_text = span_text
                
                last_x1 = span["x1"]
            
            # Don't forget the last column
            if current_column_text:
                columns.append(current_column_text.strip())
            
            if columns:
                grid.append(columns)
        
        # Normalize grid (make all rows have the same number of columns)
        if grid:
            max_cols = max(len(row) for row in grid)
            for row in grid:
                while len(row) < max_cols:
                    row.append("")
        
        return grid
    
    def _grid_to_markdown(self, grid: List[List[str]]) -> str:
        """Convert grid data to markdown table format."""
        if not grid:
            return ""
        
        markdown_lines = []
        
        # Header row
        header = "| " + " | ".join(grid[0]) + " |"
        markdown_lines.append(header)
        
        # Separator row
        separator = "|" + "|".join([" --- " for _ in grid[0]]) + "|"
        markdown_lines.append(separator)
        
        # Data rows
        for row in grid[1:]:
            # Pad row to match header length
            padded_row = row + [""] * (len(grid[0]) - len(row))
            row_markdown = "| " + " | ".join(padded_row) + " |"
            markdown_lines.append(row_markdown)
        
        return "\n".join(markdown_lines)
    
    def _find_caption_near_bbox(self, page: fitz.Page, bbox: BoundingBox, 
                               search_above: bool = False, _search_below: bool = True) -> Optional[str]:
        """Find caption text near a given bounding box."""
        search_margin = 30  # pixels
        
        if search_above:
            search_rect = fitz.Rect(
                bbox.x0 - search_margin,
                bbox.y0 - 50,  # Look 50 pixels above
                bbox.x1 + search_margin,
                bbox.y0
            )
        else:  # search_below
            search_rect = fitz.Rect(
                bbox.x0 - search_margin,
                bbox.y1,
                bbox.x1 + search_margin,
                bbox.y1 + 50  # Look 50 pixels below
            )
        
        # Extract text in the search area
        caption_text = page.get_text("text", clip=search_rect).strip()
        
        # Filter for caption-like text (starts with Figure, Table, etc.)
        caption_patterns = [
            r'^(Figure|Fig\.)\s+\d+',
            r'^(Table|Tab\.)\s+\d+',
            r'^(Chart|Diagram)\s+\d+',
        ]
        
        for pattern in caption_patterns:
            if re.search(pattern, caption_text, re.IGNORECASE):
                return caption_text
        
        # If no pattern matches but there's text, return it anyway (might be a caption)
        if caption_text and len(caption_text) < 200:  # Reasonable caption length
            return caption_text
        
        return None
    
    def _detect_table_of_contents(self) -> Optional[TableOfContents]:
        """Detect table of contents using GPT analysis. For PDFs ≤30 pages, analyze entire PDF for major sections."""
        if not self.openai_client or len(self.pages_layout) == 0:
            return None
        
        # If PDF is 30 pages or less, analyze entire PDF for major sections
        if len(self.pages_layout) <= 30:
            return self._analyze_entire_pdf_for_sections()
        
        # Original logic for larger PDFs: analyze first 10 pages for TOC
        max_pages_to_check = min(10, len(self.pages_layout))
        potential_toc_pages = []
        
        # First pass: find pages that look like TOC
        for page_num in range(max_pages_to_check):
            page_layout = self.pages_layout[page_num]
            page_text = self.pypdf_pages_text[page_num]
            
            if self._is_potential_toc_page(page_text):
                print(page_num)
                potential_toc_pages.append((page_num + 1, page_text, page_layout))
        
        if not potential_toc_pages:
            return None
        
        # Second pass: check for TOC continuation pages
        # If we found TOC pages, check the immediate next pages too
        last_toc_page = max(page_num for page_num, _, _ in potential_toc_pages)
        
        # Check up to 3 pages after the last detected TOC page
        for page_num in range(last_toc_page, min(last_toc_page + 3, len(self.pages_layout))):
            if page_num + 1 not in [p[0] for p in potential_toc_pages]:  # Not already added
                page_layout = self.pages_layout[page_num]
                page_text = self.pypdf_pages_text[page_num]
                
                # More lenient criteria for continuation pages
                if self._is_potential_toc_continuation_page(page_text):
                    potential_toc_pages.append((page_num + 1, page_text, page_layout))
        
        # Sort pages by page number
        potential_toc_pages.sort(key=lambda x: x[0])
        
        # Use GPT to analyze and extract TOC entries
        return self._analyze_toc_with_gpt(potential_toc_pages)
    
    def _analyze_entire_pdf_for_sections(self) -> Optional[TableOfContents]:
        """Analyze entire PDF page by page to extract major sections and their page numbers."""
        try:
            all_sections = []
            
            # Process each page to find major sections
            for page_num in range(len(self.pages_layout)):
                page_text = self.pypdf_pages_text[page_num]
                
                if not page_text.strip():
                    continue
                
                # Limit text per page for GPT processing
                if len(page_text) > 4000:
                    page_text = page_text[:4000] + "..."
                
                # Get the page image for visual context
                page = self.doc[page_num]
                page_image = self._render_page_image(page)
                
                # Analyze this page for major sections
                sections_on_page = self._extract_sections_from_page_with_gpt(
                    page_text, page_num + 1, page_image
                )
                
                if sections_on_page:
                    all_sections.extend(sections_on_page)
            
            if not all_sections:
                return None
            
            # Create TableOfContents object from extracted sections
            toc_entries = []
            for section in all_sections:
                # Create a dummy bounding box since we don't have exact positioning
                dummy_bbox = BoundingBox(x0=0, y0=0, x1=100, y1=20)
                
                toc_entry = TOCEntry(
                    title=section['title'],
                    page_number=section['page_number'],
                    level=section['level'],
                    bbox=dummy_bbox,
                    source_page=section['page_number']  # Section appears on its own page
                )
                toc_entries.append(toc_entry)
            
            # Remove duplicates and sort by page number
            seen_titles = set()
            unique_entries = []
            for entry in sorted(toc_entries, key=lambda x: x.page_number):
                title_key = entry.title.lower().strip()
                if title_key not in seen_titles and title_key:
                    seen_titles.add(title_key)
                    unique_entries.append(entry)
            
            if not unique_entries:
                return None
            
            return TableOfContents(
                entries=unique_entries,
                detected_pages=list(range(1, len(self.pages_layout) + 1)),
                confidence=0.8  # High confidence for full PDF analysis
            )
            
        except Exception as e:
            print(f"Error in full PDF section analysis: {e}")
            return None
    
    def _render_page_image(self, page) -> str:
        """Render a page as an image and return base64 encoded string."""
        try:
            import base64
            import io
            
            # Render page as pixmap (image)
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            
            # Encode to base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return img_base64
            
        except Exception as e:
            print(f"Error rendering page image: {e}")
            return None
    
    def _extract_sections_from_page_with_gpt(self, page_text: str, page_number: int, page_image: str = None) -> List[dict]:
        """Use GPT to extract major sections from a single page."""
        system_prompt = """You are analyzing a single page from an academic paper or document. Your task is to identify major section headings on this page.

You will be provided with both the text content and a visual image of the page. Use the visual layout to help identify which text represents section headings by looking for:
- Larger/bold fonts
- Different formatting/styling
- Spacing around text
- Numbered sections
- Consistent formatting patterns

Extract any section headings that appear to be:
- Chapter titles
- Main section headings (Introduction, Methods, Results, Conclusion, etc.)
- Major subsection headings
- Appendix sections

For each section heading found, determine its hierarchical level:
- Level 0: Main chapters or major sections (e.g., "Introduction", "Chapter 1")  
- Level 1: Primary subsections (e.g., "1.1 Background", "Methods")
- Level 2: Secondary subsections (e.g., "1.1.1 Previous Work")

Do NOT include:
- Page headers/footers
- Figure/table captions
- Reference lists
- Very short phrases
- Author names or affiliations
- Regular paragraph text
- Table of Contents entries (if this page is a TOC page, return empty array)
- Section titles that appear in a list format with page numbers (these are TOC entries, not actual section headings)

IMPORTANT: Only extract actual section headings that appear in the document content itself, NOT section titles that appear in a table of contents listing.

Return a JSON object with a key "sections" mapping to an array of objects with: {"title": "section name", "level": 0-2}
If no major sections are found, have "sections" map to an empty array []."""

        # Prepare messages with image if available
        messages = [{"role": "system", "content": system_prompt}]
        
        if page_image:
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Page {page_number}:\n\nText content:\n{page_text}\n\nAnalyze the page image and text to extract major section headings. Return JSON object only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{page_image}"
                        }
                    }
                ]
            }
        else:
            user_message = {
                "role": "user",
                "content": f"Page {page_number}:\n\n{page_text}\n\nExtract major section headings from this page. Return JSON object only."
            }
        
        messages.append(user_message)

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",  # Use gpt-4o for vision capabilities
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        import json
        try:
            response_data = json.loads(content)
            
            # Extract sections from the "sections" key
            if not isinstance(response_data, dict) or 'sections' not in response_data:
                print("No 'sections' key found in response")
                return []
            
            sections_data = response_data['sections']
            if not isinstance(sections_data, list):
                print("'sections' value is not a list")
                return []
            
            # Add page number to each section
            sections = []
            for section in sections_data:
                if isinstance(section, dict) and 'title' in section and 'level' in section:
                    sections.append({
                        'title': section['title'],
                        'page_number': page_number,
                        'level': min(max(section['level'], 0), 2)  # Clamp level to 0-2
                    })
            
            return sections
            
        except json.JSONDecodeError:
            print("Failed to decode JSON")
            return []
                
    
    def _extract_page_text_for_toc_analysis(self, page_layout: PageLayout) -> str:
        """Extract and organize text from a page for TOC analysis."""
        # Prefer clean paragraphs if available (from PyPDF2), fallback to text spans
        if page_layout.clean_paragraphs:
            # Use clean paragraphs first - they have better text ordering
            lines = []
            for para in page_layout.clean_paragraphs:
                # Split paragraphs into lines for better TOC pattern detection
                para_lines = para.text.split('\n')
                for line in para_lines:
                    line = line.strip()
                    if line:
                        lines.append(line)
            
            # Also include text spans for detailed analysis of formatting patterns
            # Sort text spans by reading order (y-position, then x-position)
            sorted_spans = sorted(page_layout.text_spans, 
                                 key=lambda span: (span.bbox.y0, span.bbox.x0))
            
            # Group spans into lines and preserve formatting for TOC-specific patterns
            span_lines = []
            current_line = []
            current_y = None
            tolerance = 5
            
            for span in sorted_spans:
                if current_y is None or abs(span.bbox.y0 - current_y) <= tolerance:
                    current_line.append(span)
                    current_y = span.bbox.y0
                else:
                    if current_line:
                        line_text = " ".join(s.text for s in current_line if s.text)
                        span_lines.append(line_text.strip())
                    current_line = [span]
                    current_y = span.bbox.y0
            
            # Don't forget the last line
            if current_line:
                line_text = " ".join(s.text for s in current_line if s.text)
                span_lines.append(line_text.strip())
            
            # Combine both sources, giving priority to clean paragraphs
            all_lines = lines + span_lines
            # Remove duplicates while preserving order
            seen = set()
            unique_lines = []
            for line in all_lines:
                if line and line not in seen:
                    unique_lines.append(line)
                    seen.add(line)
            lines = unique_lines
        else:
            # Fallback to original text span method
            # Sort text spans by reading order (y-position, then x-position)
            sorted_spans = sorted(page_layout.text_spans, 
                                 key=lambda span: (span.bbox.y0, span.bbox.x0))
            
            # Group spans into lines and preserve formatting
            lines = []
            current_line = []
            current_y = None
            tolerance = 5
            
            for span in sorted_spans:
                if current_y is None or abs(span.bbox.y0 - current_y) <= tolerance:
                    current_line.append(span)
                    current_y = span.bbox.y0
                else:
                    if current_line:
                        line_text = " ".join(s.text for s in current_line if s.text)
                        lines.append(line_text.strip())
                    current_line = [span]
                    current_y = span.bbox.y0
            
            # Don't forget the last line
            if current_line:
                line_text = " ".join(s.text for s in current_line if s.text)
                lines.append(line_text.strip())
        
        return "\n".join(lines)
    
    def _is_potential_toc_page(self, page_text: str) -> bool:
        """Quick heuristic check if a page might contain a table of contents."""
        page_text = page_text.replace(" ", "").replace("\t", "")
        if "contents" in page_text.lower():
            return True
        toc_indicators = [
            "table of contents" in page_text.lower(),
            "contents" in page_text.lower(),
            # Look for patterns like "Chapter 1...5" or "Introduction...1"
            bool(re.search(r'\w+\.{2,}\d+', page_text)),
            # Look for multiple page number patterns
            len(re.findall(r'\b\d{1,3}\b', page_text)) > 5,
            # Look for hierarchical numbering
            bool(re.search(r'\d+\.\d+', page_text)),
            # Look for appendix patterns
            "appendix" in page_text.lower(),
            bool(re.search(r'\d+\.', page_text)),
            "acknowledgements" in page_text.lower(),
            "bibliography" in page_text.lower(),
            "references" in page_text.lower()
        ]
        
        return sum(toc_indicators) >= 2
    
    def _is_potential_toc_continuation_page(self, page_text: str) -> bool:
        """Check if a page might be a continuation of a table of contents."""
        page_text = page_text.replace(" ", "").replace("\t", "")
        continuation_indicators = [
            # Look for appendix entries (common in academic papers)
            "appendix" in page_text.lower(),
            # Look for patterns with page numbers
            bool(re.search(r'\w+\.{2,}\d+', page_text)),
            # Multiple page number patterns (but lower threshold than main TOC)
            len(re.findall(r'\b\d{1,3}\b', page_text)) > 3,
            # Look for bibliography, references, notes sections
            bool(re.search(r'(acknowledgements|bibliography|references|notes|abouttheauthor)', page_text.lower())),
            # Look for numbered sections continuing from previous page
            bool(re.search(r'\d+:', page_text)),
        ]
        
        return sum(continuation_indicators) >= 2
    
    def _analyze_toc_with_gpt(self, potential_toc_pages: List) -> Optional[TableOfContents]:
        """Use GPT with vision to analyze potential TOC pages and extract entries with multi-page support."""
        try:
            # Combine text and images from all potential TOC pages
            combined_text = ""
            detected_pages = []
            page_images = []
            
            for page_num, page_text, page_layout in potential_toc_pages:
                detected_pages.append(page_num)
                combined_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                
                # Render the page as an image for visual analysis
                try:
                    page = self.doc[page_num - 1]  # Convert to 0-indexed
                    page_image = self._render_page_image(page)
                    if page_image:
                        page_images.append((page_num, page_image))
                except Exception as e:
                    print(f"Warning: Could not render page {page_num} as image: {e}")
            
            # Limit text length for GPT
            if len(combined_text) > 8000:  # Reduced since we're also sending images
                combined_text = combined_text[:8000] + "..."
            
            system_prompt = """You are analyzing text and images that may contain a table of contents spanning multiple pages. Your task is to:
1. Determine if this content contains a table of contents (confidence: 0.0-1.0)
2. If it does, extract ALL TOC entries from ALL pages with:
   - Title (the section/chapter name)
   - Page number (if present)
   - Level (0 = main chapter, 1 = subsection, 2 = sub-subsection, etc.)

IMPORTANT: You have both text and visual images of the pages. Use the visual images to verify page numbers from the text as OCR can misread numbers.

The table of contents may span multiple pages. Look for:
- Main sections (Introduction, Summary, etc.) 
- Appendices (Appendix 1, Appendix 2, etc.)
- References, Notes, Bibliography
- About the Author sections

Skip any TOC sections with Roman numeral page numbers (i, ii, iii, iv, v, etc.). Start from the first section using Arabic numerals (1, 2, 3, etc.).

If the page(s) are not a table of contents, return an empty list for entries, even if there are section headings visible.

Return a JSON object with this structure:
{
  "is_toc": true/false,
  "confidence": 0.0-1.0,
  "entries": [
    {
      "title": "Summary",
      "page_number": 1,
      "level": 0
    },
    {
      "title": "Introduction",
      "page_number": 5,
      "level": 0
    },
    {
      "title": "Appendix 1: Title Here",
      "page_number": 77,
      "level": 0
    }
  ]
}"""

            # Build multimodal user message with text and images
            user_message_content = [
                {
                    "type": "text",
                    "text": f"Analyze this multi-page table of contents and extract ALL entries. Use the IMAGES to verify correct page numbers (OCR often misreads '1' as 'll'):\n\nText content:\n{combined_text}"
                }
            ]
            
            # Add images to the message
            for page_num, page_image in page_images:
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page_image}"
                    }
                })
            
            print(f"Sending TOC analysis with {len(page_images)} page images")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,  # Increased for more entries
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if not result.get("is_toc", False) or result.get("confidence", 0) < 0.6:  # Slightly lower threshold
                return None
            
            print(result.get("entries"))
            
            # Extract link text and destinations from all potential TOC pages
            all_link_text_destinations = {}
            for page_num, _, _ in potential_toc_pages:
                page_links = self._extract_link_text_and_destinations_from_page(page_num - 1)  # Convert to 0-indexed
                all_link_text_destinations.update(page_links)
            
            # Convert GPT results to TOCEntry objects
            toc_entries = []
            for entry_data in result.get("entries", []):
                # Try to find bounding box for this entry
                bbox, source_page = self._find_toc_entry_bbox_with_page(entry_data["title"], potential_toc_pages)
                
                # Try to find page number from clickable links using text matching
                page_number_from_link = None
                if all_link_text_destinations:
                    page_number_from_link = self._find_page_number_by_text_match(entry_data["title"], all_link_text_destinations)
                
                # Use link destination if found, otherwise use GPT extracted page number
                final_page_number = page_number_from_link or entry_data.get("page_number", 0)
                
                toc_entry = TOCEntry(
                    title=entry_data["title"],
                    page_number=final_page_number,
                    level=entry_data.get("level", 0),
                    bbox=bbox or BoundingBox(0, 0, 100, 20),
                    source_page=source_page or detected_pages[0] if detected_pages else 1
                )
                toc_entries.append(toc_entry)
            
            return TableOfContents(
                entries=toc_entries,
                detected_pages=detected_pages,
                confidence=result.get("confidence", 0.8)
            )
            
        except Exception as e:
            print(f"Warning: Could not analyze TOC with GPT: {e}")
            return None
    
    def _find_toc_entry_bbox(self, title: str, potential_toc_pages: List) -> Optional[BoundingBox]:
        """Find bounding box for a TOC entry by searching for the title text."""
        bbox, _ = self._find_toc_entry_bbox_with_page(title, potential_toc_pages)
        return bbox
    
    def _find_toc_entry_bbox_with_page(self, title: str, potential_toc_pages: List) -> tuple[Optional[BoundingBox], Optional[int]]:
        """Find bounding box and source page for a TOC entry by searching for the title text."""
        # Clean the title for better matching
        clean_title = title.lower().strip()
        
        # Try exact matches first
        for page_num, page_text, page_layout in potential_toc_pages:
            for span in page_layout.text_spans:
                if span.text and clean_title in span.text.lower():
                    return span.bbox, page_num
        
        # Try partial matches for longer titles
        words = clean_title.split()
        if len(words) > 1:
            for page_num, page_text, page_layout in potential_toc_pages:
                for span in page_layout.text_spans:
                    if span.text:
                        span_text = span.text.lower()
                        # Check if at least half the words match
                        matching_words = sum(1 for word in words if word in span_text)
                        if matching_words >= len(words) // 2:
                            return span.bbox, page_num
        
        # Try matching just the first few significant words
        if len(words) > 2:
            key_words = [w for w in words[:3] if len(w) > 3]  # First 3 words longer than 3 chars
            for page_num, page_text, page_layout in potential_toc_pages:
                for span in page_layout.text_spans:
                    if span.text:
                        span_text = span.text.lower()
                        if key_words and all(word in span_text for word in key_words):
                            return span.bbox, page_num
        
        return None, None
    
    def _extract_link_text_and_destinations_from_page(self, page_num: int) -> Dict[str, int]:
        """Extract clickable link text and destinations from a page."""
        if page_num < 0 or page_num >= len(self.doc):
            return {}
        
        page = self.doc[page_num]
        link_text_destinations = {}
        
        try:
            # Get all links from the page
            links = page.get_links()
            
            for link in links:
                try:
                    # Check if it's an internal link (goes to another page in the same document)
                    dest_page = None
                    if link.get("kind") == 1 and "page" in link:  # kind 1 is LINK_GOTO
                        dest_page = link["page"] + 1  # Convert to 1-indexed
                    elif link.get("kind") == fitz.LINK_GOTOR and link.get("to"):
                        # Handle other internal link types
                        try:
                            if isinstance(link["to"], dict) and "page" in link["to"]:
                                dest_page = link["to"]["page"] + 1
                        except:
                            continue
                    
                    if dest_page:
                        # Get the text content of the link by extracting text from the link's bounding box
                        link_rect = link["from"]
                        link_text = page.get_text("text", clip=link_rect).strip()
                        
                        if link_text:
                            # Clean up the text and use it as key
                            clean_text = link_text.lower().strip()
                            link_text_destinations[clean_text] = dest_page
                            
                except Exception as e:
                    # Skip problematic links but continue processing
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not extract links from page {page_num + 1}: {e}")
            
        return link_text_destinations
    
    def _find_page_number_by_text_match(self, toc_title: str, link_text_destinations: Dict[str, int]) -> Optional[int]:
        """Find page number by matching TOC title with link text using GPT."""
        if not link_text_destinations or not self.openai_client:
            return None
        
        # Try exact match first (case insensitive)
        clean_title = toc_title.lower().strip()
        for link_text, page_num in link_text_destinations.items():
            if clean_title == link_text.lower().strip():
                return page_num
        
        # Use GPT for intelligent matching
        try:
            links_list = []
            for link_text, page_num in link_text_destinations.items():
                links_list.append(f'"{link_text}" → page {page_num}')
            
            links_text = '\n'.join(links_list)
            
            system_prompt = """You are helping match a table of contents entry with clickable links. Given a TOC title and a list of clickable links with their destinations, find the best match.

Return only the page number of the best matching link, or "null" if no good match exists.

Rules:
- Look for exact or very close semantic matches
- Ignore minor differences in formatting, capitalization, or numbering
- Be conservative - only match if you're confident it's the right link"""

            user_prompt = f"""TOC Title: "{toc_title}"

Available clickable links:
{links_text}

Which link best matches this TOC title? Return only the page number or "null"."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for speed
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse the result as a page number
            try:
                if result.lower() == "null":
                    return None
                return int(result)
            except ValueError:
                print("GPT response in an unexpected format")
                return None
                
        except Exception as e:
            print(f"Warning: GPT matching failed for '{toc_title}': {e}")
            return None
    
    def _find_matching_link_destination(self, text_bbox: BoundingBox, link_destinations: Dict[BoundingBox, int]) -> Optional[int]:
        """Find the destination page number for a TOC entry by matching its bounding box with clickable links."""
        # Look for links that overlap or are very close to the text bounding box
        tolerance = 10  # pixels
        
        for link_bbox, dest_page in link_destinations.items():
            # Check if bounding boxes overlap or are very close
            if self._bboxes_overlap_or_close(text_bbox, link_bbox, tolerance):
                return dest_page
        
        return None
    
    def _bboxes_overlap_or_close(self, bbox1: BoundingBox, bbox2: BoundingBox, tolerance: float) -> bool:
        """Check if two bounding boxes overlap or are within tolerance distance."""
        # Check for overlap
        if (bbox1.x0 <= bbox2.x1 and bbox1.x1 >= bbox2.x0 and
            bbox1.y0 <= bbox2.y1 and bbox1.y1 >= bbox2.y0):
            return True
        
        # Check if they're close (within tolerance)
        x_distance = min(abs(bbox1.x0 - bbox2.x1), abs(bbox1.x1 - bbox2.x0), 
                        abs(bbox1.x0 - bbox2.x0), abs(bbox1.x1 - bbox2.x1))
        y_distance = min(abs(bbox1.y0 - bbox2.y1), abs(bbox1.y1 - bbox2.y0),
                        abs(bbox1.y0 - bbox2.y0), abs(bbox1.y1 - bbox2.y1))
        
        return x_distance <= tolerance and y_distance <= tolerance
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert the parsed layout to a dictionary for JSON serialization."""
        result = {
            "document_info": {
                "filename": self.pdf_path.name,
                "total_pages": len(self.pages_layout)
            },
            "pages": [asdict(page_layout) for page_layout in self.pages_layout]
        }
        
        # Add table of contents if detected
        if self.table_of_contents:
            result["table_of_contents"] = asdict(self.table_of_contents)
        
        return result
    
    def save_json(self, output_path: str) -> None:
        """Save the parsed layout to a JSON file."""
        layout_data = self._to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()


def main():
    """Example usage of the PDF Layout Parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse PDF layout and extract structured blocks")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file path", 
                       default="layout_output.json")
    parser.add_argument("--gpt-token", help="Path to GPT token file for TOC detection", 
                       default="gpt_token.txt")
    
    args = parser.parse_args()
    
    try:
        # Parse the PDF
        pdf_parser = PDFLayoutParser(args.pdf_path, args.gpt_token)
        layout_data = pdf_parser.parse()
        
        # Save to JSON
        pdf_parser.save_json(args.output)
        pdf_parser.close()
        
        print(f"Layout analysis complete. Results saved to {args.output}")
        
        # Print summary
        total_pages = layout_data["document_info"]["total_pages"]
        total_text_spans = sum(len(page["text_spans"]) for page in layout_data["pages"])
        total_figures = sum(len(page["figures"]) for page in layout_data["pages"])
        total_tables = sum(len(page["tables"]) for page in layout_data["pages"])
        total_footnotes = sum(len(page["footnotes"]) for page in layout_data["pages"])
        
        print(f"Summary:")
        print(f"  Pages: {total_pages}")
        print(f"  Text spans: {total_text_spans}")
        print(f"  Figures: {total_figures}")
        print(f"  Tables: {total_tables}")
        print(f"  Footnotes: {total_footnotes}")
        
        # Print TOC info if detected
        if "table_of_contents" in layout_data:
            toc = layout_data["table_of_contents"]
            print(f"  TOC detected: {len(toc['entries'])} entries (confidence: {toc['confidence']:.2f})")
            
            # Print details of TOC entries with page numbers from links
            entries_with_pages = [entry for entry in toc['entries'] if entry.get('page_number')]
            entries_without_pages = [entry for entry in toc['entries'] if not entry.get('page_number')]
            
            print(f"    Entries with page numbers from links: {len(entries_with_pages)}")
            print(f"    Entries without page numbers: {len(entries_without_pages)}")
            
            if entries_with_pages:
                print(f"    Sample entries with page numbers:")
                for entry in entries_with_pages[:3]:
                    print(f"      '{entry['title']}' -> page {entry['page_number']}")
        else:
            print(f"  TOC detected: None")
        
    except Exception as e:
        print(f"Error parsing PDF: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())