#!/usr/bin/env python3
"""
Comprehensive PDF Chat Interface

An interactive chat interface for conversing with PDF collections using RAG (Retrieval-Augmented Generation).
Supports multiple chat modes with persistent conversation history and advanced multimodal capabilities.

Features:
- Chat with single PDFs, TOC sections, page ranges, groups, or entire collections
- Persistent conversation history with context awareness
- Multimodal support: Tables formatted as markdown, figures as images
- Adaptive context utilization (up to 95% of GPT-4o's 128k window)
- Smart content type detection and processing
- Content-type aware search: Ask about "tables", "figures", or "charts" for targeted results
- Enhanced retrieval with HyDE and auto-TOC routing
- Real-time context statistics and optimization

Requirements:
- openai: pip install openai
- chromadb: pip install chromadb
- PyMuPDF: pip install PyMuPDF
- tiktoken: pip install tiktoken

Usage:
    # Start interactive chat mode
    python pdf_chat.py --collection demo --interactive
    
    # Chat with specific PDF
    python pdf_chat.py --collection demo --pdf document.pdf --interactive
    
    # Chat with TOC section
    python pdf_chat.py --collection demo --pdf document.pdf --toc-section "Methods" --interactive
    
    # List TOC sections for all PDFs
    python pdf_chat.py --collection demo --list-all-toc
    
    # List TOC sections for specific PDF
    python pdf_chat.py --collection demo --list-toc document.pdf
    
    # Non-interactive single question
    python pdf_chat.py --collection demo --question "What are the main findings?"
    
    # Enhanced mode with multimodal support
    python pdf_chat.py --collection demo --question "Analyze the data in Table 2" --enhanced
"""

import argparse
import json
import base64
import tiktoken
import random
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys

# Import our pipeline components
from query_collection import CollectionQueryTool
from openai import OpenAI
import anthropic
import fitz  # PyMuPDF for image extraction


class PDFChatSession:
    def __init__(self, collection_name: str, vector_store_dir: str = "./chroma_db",
                 gpt_token_path: str = "gpt_token.txt", history_file: str = "chat_history.json",
                 enhanced_mode: bool = False, anthropic_token_path : str = "anthropic_token.txt"):
        """
        Initialize comprehensive PDF chat session.
        
        Args:
            collection_name: Name of the ChromaDB collection
            vector_store_dir: Directory containing ChromaDB data
            gpt_token_path: Path to GPT token file
            history_file: Path to conversation history file
            enhanced_mode: Enable enhanced context utilization and multimodal features
        """
        self.collection_name = collection_name
        self.history_file = Path(history_file)
        self.vector_store_dir = Path(vector_store_dir)
        self.enhanced_mode = enhanced_mode
        
        # Initialize tokenizer for precise token counting (enhanced mode only)
        if enhanced_mode:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            # Enhanced context window management
            self.max_context_tokens = 120000  # 95% of GPT-4o's capacity
            self.min_response_tokens = 2000
            self.max_response_tokens = 30000
        else:
            self.tokenizer = None
            # Standard context management
            self.max_context_tokens = 8000
            self.min_response_tokens = 1000
            self.max_response_tokens = 30000
        
        # Initialize query tool with all features enabled
        self.query_tool = CollectionQueryTool(
            collection_name=collection_name,
            vector_store_dir=str(vector_store_dir),
            gpt_token_path=gpt_token_path,
            enable_hyde=False,  # Disable HyDE preprocessing
            enable_auto_toc=True,
            enable_hybrid_search=True  # Enable hybrid search by default
        )
        
        # Initialize OpenAI client
        try:
            with open(gpt_token_path, 'r') as f:
                api_key = f.readline().strip()
            self.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            sys.exit(1)

        try:
            with open(anthropic_token_path, 'r') as f:
                api_key = f.readline().strip()
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            sys.exit(1)
        
        # Load conversation history
        self.conversation_history = self._load_history()
        
        # Current chat context
        self.current_context = {
            "mode": "collection",
            "collection_name": collection_name,
            "pdf_filenames": None,
            "toc_sections": None,  # Changed from toc_section to toc_sections (array)
            "page_range": None,
            "description": f"Entire collection '{collection_name}'"
        }
        
        # Cache for opened PDF documents (enhanced mode only)
        if enhanced_mode:
            self.pdf_cache = {}
        
        print(f"📚 PDF Chat initialized for collection: {collection_name}")
        if enhanced_mode:
            print(f"🔥 Enhanced mode: ENABLED (multimodal support, 95% context utilization)")
            print(f"🧠 Context capacity: {self.max_context_tokens:,} tokens")
        else:
            print(f"📝 Standard mode: Basic chat functionality")
        print(f"💾 Chat history: {self.history_file}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using GPT-4o tokenizer (enhanced mode only)."""
        if self.enhanced_mode and self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation for standard mode
            return len(text.split()) * 1.3
    
    def _add_random_cute_message(self, response: str) -> str:
        """Easter egg: randomly append a cute message from Sukhm (5% chance)."""
        if random.random() < 0.05:  # 5% chance
            cute_messages = [
                "\n\nBy the way, you're super cute :) - Sukhm",
                "\n\nBy the way, you're really pretty - Sukhm", 
                "\n\nBy the way, you're the best Kristin - Sukhm",
                "\n\nBy the way, you're lovely :) - Sukhm",
                "\n\nBy the way, you're amazing - Sukhm",
                "\n\nBy the way, nice job on all your hard work... remember to take a break, okay? - Sukhm"
            ]
            selected_message = random.choice(cute_messages)
            return response + selected_message
        return response
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                print(f"📖 Loaded {len(history)} previous conversations")
                return history
            except Exception as e:
                print(f"Warning: Could not load chat history: {e}")
        return []
    
    def _save_history(self) -> None:
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save chat history: {e}")
    
    def _add_to_history(self, user_message: str, assistant_response: str, 
                       context: Dict[str, Any], retrieved_chunks: List[Dict] = None,
                       context_stats: Dict[str, int] = None) -> None:
        """Add conversation turn to history."""
        conversation_turn = {
            "timestamp": datetime.now().isoformat(),
            "context": context.copy(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "retrieved_chunks": len(retrieved_chunks) if retrieved_chunks else 0,
            "enhanced_mode": self.enhanced_mode
        }
        
        if context_stats:
            conversation_turn["context_stats"] = context_stats
            
        self.conversation_history.append(conversation_turn)
        self._save_history()
    
    # Enhanced mode methods
    def _get_pdf_document(self, pdf_filename: str) -> Optional[fitz.Document]:
        """Get or cache PDF document for image extraction (enhanced mode only)."""
        if not self.enhanced_mode:
            return None
            
        if pdf_filename in self.pdf_cache:
            return self.pdf_cache[pdf_filename]
        
        # Try to find the PDF file
        possible_paths = [
            Path(pdf_filename),
            Path.cwd() / pdf_filename,
            Path.cwd() / "pdfs" / pdf_filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    doc = fitz.open(str(path))
                    self.pdf_cache[pdf_filename] = doc
                    return doc
                except Exception as e:
                    print(f"Warning: Could not open PDF {pdf_filename}: {e}")
                break
        
        return None
    
    def _detect_content_type(self, chunk: Dict[str, Any]) -> str:
        """Detect content type: 'table', 'figure', or 'text' (enhanced mode only)."""
        if not self.enhanced_mode:
            return 'text'
            
        page_type = chunk.get('page_type', '').lower()
        section_title = chunk.get('section_title', '').lower()
        content = chunk.get('content', '').lower()
        
        # Check page type first
        if page_type == 'table':
            return 'table'
        elif page_type == 'figure':
            return 'figure'
        
        # Check section title patterns
        if any(keyword in section_title for keyword in ['table', 'fig', 'figure']):
            return 'table' if 'table' in section_title else 'figure'
        
        # Check content patterns for tables
        if any(indicator in content for indicator in ['|', '\t']):
            lines = content.split('\n')
            pipe_lines = [line for line in lines if '|' in line]
            if len(pipe_lines) >= 2:
                return 'table'
        
        # Check content patterns for figures
        figure_indicators = ['figure shows', 'image depicts', 'chart illustrates', 'graph displays']
        if any(indicator in content for indicator in figure_indicators):
            return 'figure'
        
        return 'text'
    
    def _format_table_as_markdown(self, chunk: Dict[str, Any]) -> str:
        """Format table content as clean markdown (enhanced mode only)."""
        content = chunk.get('content', '')
        
        # If already in markdown format, clean it up
        if '|' in content:
            lines = content.split('\n')
            table_lines = []
            
            for line in lines:
                if '|' in line:
                    cells = [cell.strip() for cell in line.split('|')]
                    while cells and not cells[0]:
                        cells.pop(0)
                    while cells and not cells[-1]:
                        cells.pop()
                    
                    if cells:
                        table_lines.append('| ' + ' | '.join(cells) + ' |')
                elif line.strip():
                    table_lines.append(line.strip())
            
            # Add separator after header if not present
            if len(table_lines) >= 2 and not table_lines[1].startswith('|--'):
                if '|' in table_lines[0] and '|' in table_lines[1]:
                    header_cols = len([cell for cell in table_lines[0].split('|') if cell.strip()])
                    separator = '| ' + ' | '.join(['---'] * (header_cols - 2)) + ' |'
                    table_lines.insert(1, separator)
            
            return '\n'.join(table_lines)
        
        return content
    
    def _extract_figure_image(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract figure image from PDF and return as base64 (enhanced mode only)."""
        if not self.enhanced_mode:
            return None
            
        pdf_filename = chunk.get('pdf_filename')
        page_number = chunk.get('page_number')
        
        if not pdf_filename or not page_number:
            return None
        
        pdf_doc = self._get_pdf_document(pdf_filename)
        if not pdf_doc:
            return None
        
        try:
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(pdf_doc):
                return None
            
            page = pdf_doc[page_index]
            matrix = fitz.Matrix(1.5, 1.5)  # Reasonable resolution
            pix = page.get_pixmap(matrix=matrix)
            
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode()
            
            return img_base64
            
        except Exception as e:
            print(f"Warning: Could not extract figure from {pdf_filename}, page {page_number}: {e}")
            return None
    
    def _detect_content_type_request(self, question: str) -> Optional[str]:
        """Detect if user is specifically asking about figures, tables, or specific content types."""
        question_lower = question.lower()
        
        # Table-specific keywords (more specific than general visual)
        table_keywords = ['table', 'tables', 'tabular', 'data table', 'spreadsheet', 'rows and columns']
        # Figure-specific keywords 
        figure_keywords = ['figure', 'figures', 'fig', 'image', 'images', 'chart', 'charts', 'graph', 'graphs', 
                          'diagram', 'diagrams', 'illustration', 'illustrations', 'plot', 'plots', 'visual', 'visuals']
        # Parent chunk indicators (larger context needed)
        parent_keywords = ['overview', 'summary', 'entire section', 'full context', 'complete', 'comprehensive']
        
        # Check for explicit table requests
        if any(keyword in question_lower for keyword in table_keywords):
            return "table"
            
        # Check for explicit figure requests
        if any(keyword in question_lower for keyword in figure_keywords):
            return "figure"
            
        # Check for parent chunk requests
        if any(keyword in question_lower for keyword in parent_keywords):
            return "parent"
            
        return None

    def _analyze_question_complexity(self, question: str) -> str:
        """Analyze question complexity (enhanced mode only)."""
        if not self.enhanced_mode:
            return "medium"
            
        question_lower = question.lower()
        
        # Visual content questions
        visual_keywords = ['figure', 'table', 'chart', 'graph', 'image', 'diagram', 'show me']
        # Complex analytical questions
        complex_keywords = ['compare', 'contrast', 'analyze', 'synthesize', 'evaluate', 'comprehensive']
        # Simple factual questions
        simple_keywords = ['what is', 'who is', 'when did', 'define', 'list', 'name']
        
        if any(keyword in question_lower for keyword in visual_keywords):
            return "visual"
        elif any(keyword in question_lower for keyword in complex_keywords):
            return "complex"
        elif any(keyword in question_lower for keyword in simple_keywords):
            return "simple"
        else:
            return "medium"
    
    def _fix_typos(self, question: str) -> str:
        """Fix typos in user query using GPT."""
        try:
            # Skip typo correction for very short queries or if they seem fine
            if len(question.split()) <= 2:
                return question
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for typo correction
                messages=[{
                    "role": "user",
                    "content": f"""Fix any typos in this question while preserving the original meaning and intent. Only fix obvious spelling/grammar errors. If there are no typos, return the exact original text.

Question: {question}

Fixed question:"""
                }],
                max_tokens=150,
                temperature=0.1
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Only use correction if it's meaningfully different
            if corrected.lower() != question.lower() and len(corrected) > 0:
                print(f"🔧 Typo correction: '{question}' → '{corrected}'")
                return corrected
            else:
                return question
                
        except Exception as e:
            print(f"Warning: Typo correction failed: {e}")
            return question

    def _retrieve_relevant_chunks(self, question: str, n_results: int = 8) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks based on current context and content type requests."""
        try:
            # Fix typos in the question first
            corrected_question = self._fix_typos(question)
            
            # Detect content type request
            content_type_request = self._detect_content_type_request(corrected_question)
            
            # Determine chunk type for search
            if content_type_request == "table":
                chunk_type = "child"  # Tables are usually in child chunks
                print(f"🔍 Detected table request - searching table content...")
            elif content_type_request == "figure":
                chunk_type = "child"  # Figures are usually in child chunks
                print(f"🔍 Detected figure request - searching figure content...")
            elif content_type_request == "parent":
                chunk_type = "parent"  # User wants broader context
                print(f"🔍 Detected parent context request - searching larger sections...")
            else:
                chunk_type = "child"  # Default to child chunks for better specificity
            
            # Adjust n_results if searching for specific content types
            if content_type_request in ["table", "figure"]:
                # Get more results initially to filter for specific content types
                search_results = n_results * 2
            else:
                search_results = n_results
            
            # Perform the search based on context mode using corrected question
            if self.current_context["mode"] == "collection":
                results = self.query_tool.query_entire_collection(
                    query=corrected_question,
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=True
                )
            elif self.current_context["mode"] == "pdf":
                results = self.query_tool.query_specific_pdfs(
                    query=corrected_question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=True
                )
            elif self.current_context["mode"] == "toc_section":
                results = self.query_tool.query_by_toc_section(
                    query=corrected_question,
                    toc_sections=self.current_context["toc_sections"],  # Now expects a list
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True
                )
            elif self.current_context["mode"] == "page_range":
                results = self.query_tool.query_specific_pdfs(
                    query=corrected_question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results * 2,
                    chunk_type=chunk_type,
                    use_hyde=True
                )
                # Filter results by page range
                if self.current_context["page_range"]:
                    start_page, end_page = self.current_context["page_range"]
                    filtered_results = []
                    for result in results.get("results", []):
                        page_num = result.get("page_number", 0)
                        if start_page <= page_num <= end_page:
                            filtered_results.append(result)
                    results["results"] = filtered_results[:search_results]
            elif self.current_context["mode"] == "group":
                results = self.query_tool.query_specific_pdfs(
                    query=corrected_question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=True
                )
            else:
                results = {"results": []}
            
            # Post-process results for content type filtering
            retrieved_chunks = results.get("results", [])
            
            if content_type_request in ["table", "figure"] and retrieved_chunks:
                # Filter chunks by detected content type
                filtered_chunks = []
                for chunk in retrieved_chunks:
                    detected_type = self._detect_content_type(chunk)
                    
                    # Match content types
                    if content_type_request == "table" and detected_type == "table":
                        filtered_chunks.append(chunk)
                    elif content_type_request == "figure" and detected_type == "figure":
                        filtered_chunks.append(chunk)
                    elif content_type_request not in ["table", "figure"]:
                        # Include all chunks if not specifically filtering
                        filtered_chunks.append(chunk)
                
                # If we found specific content, use it; otherwise fall back to all results
                if filtered_chunks:
                    print(f"📋 Found {len(filtered_chunks)} {content_type_request}(s) out of {len(retrieved_chunks)} total chunks")
                    return filtered_chunks[:n_results]
                else:
                    print(f"⚠️  No specific {content_type_request}s found, showing general results")
                    return retrieved_chunks[:n_results]
            
            return retrieved_chunks[:n_results]
            
        except Exception as e:
            print(f"Warning: Error retrieving chunks: {e}")
            return []
    
    def _build_standard_context_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Build context prompt for standard mode."""
        # Build conversation context from recent history (same collection only, not cleared)
        recent_history = []
        # Filter to same collection and mode first, then take last 5
        collection_history = [
            turn for turn in self.conversation_history
            if (turn["context"].get("collection_name") == self.current_context["collection_name"] and
                not turn.get("is_cleared", False))
        ]
        
        # Auto-compact conversation history if it's getting too long
        collection_history = self._get_compacted_history(collection_history)
        
        # Take the last 5 turns from the collection-specific history
        for turn in collection_history[-5:]:
            recent_history.append(f"Human: {turn['user_message']}")
            recent_history.append(f"Assistant: {turn['assistant_response']}")
        
        # Debug output
        total_turns = len(self.conversation_history)
        collection_turns = len(collection_history)
        included_turns = len(recent_history) // 2  # Divide by 2 since each turn has Human + Assistant
        print(f"🧠 Chat History: {included_turns} turns included from {collection_turns} collection turns ({total_turns} total) (collection: {self.current_context['collection_name']}, mode: {self.current_context['mode']})")
        if recent_history:
            # Show abbreviated version of most recent turn
            last_human = recent_history[-2] if len(recent_history) >= 2 else ""
            last_assistant = recent_history[-1] if recent_history else ""
            print(f"   Last turn - Human: \"{last_human[:50]}{'...' if len(last_human) > 50 else ''}\"")
            print(f"   Last turn - Assistant: \"{last_assistant[:50]}{'...' if len(last_assistant) > 50 else ''}\"")
        
        conversation_context = "\n".join(recent_history) if recent_history else "This is the start of our conversation."
        
        # Build retrieved content context
        if retrieved_chunks:
            content_context = "\n\n".join([
                f"[Source: PDF={chunk.get('metadata', {}).get('pdf_filename', 'Unknown')}, Page={chunk.get('metadata', {}).get('page_number', '?')}, ChunkID={chunk.get('metadata', {}).get('chunk_id', 'Unknown')}, Section={chunk.get('metadata', {}).get('section_title', 'Untitled')}]\n{chunk.get('content', '')}"
                for chunk in retrieved_chunks[:6]  # Top 6 most relevant
            ])
        else:
            content_context = "No specific relevant content was found in the documents."
        
        # Build system prompt
        system_prompt = f"""You are a helpful AI assistant in an app that answers questions about PDF documents using retrieved content. 

Current Context: {self.current_context['description']}

Instructions:
1. Answer based on the retrieved content below and previous conversation context
2. Include citations for all source references using this exact format: <citation pdf_name="file.pdf" page_number="123" chunk_id="id" cited_text="first 5-7 words...">
3. Citation tags are INVISIBLE to users - put all visible text OUTSIDE the tags
4. Say clearly if the content doesn't address the question
5. Your answer MUST include citations

Citation Examples:
✓ GOOD: The author argues that life is absurd <citation pdf_name="nagel.pdf" page_number="4" chunk_id="abc123" cited_text="Nothing we do now...">.
✗ WRONG: The author argues that <citation cited_text="life is absurd because nothing matters in the long run">.

The wrong example would display as "The author argues that." to users since citation content is hidden.

Current Context: {self.current_context['description']}

Previous Conversation:
{conversation_context}

Retrieved Content:
{content_context}

REQUIRED JSON RESPONSE FORMAT:
{{
  "found_answer": true/false,
  "confidence": "high/medium/low", 
  "response": "your detailed response with proper citations"
}}

Set "found_answer" based on whether the retrieved content can substantively address the question. Set "confidence" based on how well the content matches the question.

User Question: {question}"""

        print("system prompt")
        print(system_prompt)
        
        return system_prompt
    
    def _build_enhanced_multimodal_content(self, chunks: List[Dict], token_budget: int) -> Tuple[str, List[Dict], Dict[str, int]]:
        """Build enhanced multimodal content with tables and figures (enhanced mode only)."""
        if not chunks:
            return "No specific relevant content was found in the documents.", [], {}
        
        sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        text_parts = []
        image_contents = []
        current_tokens = 0
        content_stats = {
            "total_chunks": len(chunks),
            "text_chunks": 0,
            "table_chunks": 0,
            "figure_chunks": 0,
            "images_included": 0
        }
        
        for i, chunk in enumerate(sorted_chunks):
            content_type = self._detect_content_type(chunk)
            chunk_header = f"[Source {i+1}: PDF={chunk.get('metadata', {}).get('pdf_filename', 'Unknown')}, Page={chunk.get('metadata', {}).get('page_number', '?')}, ChunkID={chunk.get('metadata', {}).get('chunk_id', 'Unknown')}, Section={chunk.get('metadata', {}).get('section_title', 'Untitled')}]"
            
            if content_type == "table":
                formatted_table = self._format_table_as_markdown(chunk)
                chunk_text = f"{chunk_header}\n{formatted_table}\n"
                content_stats["table_chunks"] += 1
                
            elif content_type == "figure":
                image_base64 = self._extract_figure_image(chunk)
                
                if image_base64:
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                    image_contents.append(image_content)
                    content_stats["images_included"] += 1
                    
                    figure_description = chunk.get('content', 'Figure content not available')
                    chunk_text = f"{chunk_header}\n[FIGURE - See image above]\n{figure_description}\n"
                else:
                    chunk_text = f"{chunk_header}\n[FIGURE - Image not available]\n{chunk.get('content', '')}\n"
                
                content_stats["figure_chunks"] += 1
                
            else:
                chunk_text = f"{chunk_header}\n{chunk.get('content', '')}\n"
                content_stats["text_chunks"] += 1
            
            # Check token budget
            chunk_tokens = self._count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= token_budget:
                text_parts.append(chunk_text)
                current_tokens += chunk_tokens
            else:
                # Try to fit truncated version
                if content_type == "text" and current_tokens < token_budget * 0.9:
                    available_tokens = token_budget - current_tokens - 50
                    content = chunk.get('content', '')
                    words = content.split()
                    for j in range(len(words), 0, -10):
                        test_content = ' '.join(words[:j]) + "... [truncated]"
                        test_chunk = f"{chunk_header}\n{test_content}\n"
                        
                        if self._count_tokens(test_chunk) <= available_tokens:
                            text_parts.append(test_chunk)
                            current_tokens += self._count_tokens(test_chunk)
                            content_stats["text_chunks"] += 1
                            break
                break
        
        text_content = "\n".join(text_parts)
        return text_content, image_contents, content_stats
    
    def _generate_dual_answers_parallel(self, question: str, model: str = "gpt-5") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate two answers in parallel with mode-specific logic:
        - TOC mode: parent chunks vs child chunks within TOC section
        - Page range mode: parent chunks vs child chunks within page range  
        - Other modes: auto-TOC routing vs no auto-TOC routing
        
        Args:
            question: User's question
            model: Model to use for generation
            
        Returns:
            Tuple of (answer1_data, answer2_data) dictionaries
        """
        import concurrent.futures
        
        corrected_question = self._fix_typos(question)
        n_results = 8  # Standard mode default
        current_mode = self.current_context["mode"]
        
        if current_mode == "toc_section":
            print(f"\n🔄 TOC Mode: Generating dual answers with parent vs child chunks...")
            
            def generate_parent_answer():
                print(f"📊 Thread 1: Using parent chunks within TOC section...")
                parent_chunks = self._retrieve_chunks_by_type(
                    corrected_question, n_results, chunk_type="parent"
                )
                return self._generate_single_answer_standard(
                    corrected_question, parent_chunks, model, "PARENT-CHUNKS"
                )
            
            def generate_child_answer():
                print(f"📄 Thread 2: Using child chunks within TOC section...")
                child_chunks = self._retrieve_chunks_by_type(
                    corrected_question, n_results, chunk_type="child"
                )
                return self._generate_single_answer_standard(
                    corrected_question, child_chunks, model, "CHILD-CHUNKS"
                )
            
            # Execute both answer generations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                parent_future = executor.submit(generate_parent_answer)
                child_future = executor.submit(generate_child_answer)
                
                parent_answer_data = parent_future.result()
                child_answer_data = child_future.result()
            
            print(f"✅ TOC mode parallel generation complete")
            return parent_answer_data, child_answer_data
            
        elif current_mode == "page_range":
            print(f"\n🔄 Page Range Mode: Generating dual answers with parent vs child chunks...")
            
            def generate_parent_answer():
                print(f"📊 Thread 1: Using parent chunks within page range...")
                parent_chunks = self._retrieve_chunks_by_type(
                    corrected_question, n_results, chunk_type="parent"
                )
                return self._generate_single_answer_standard(
                    corrected_question, parent_chunks, model, "PARENT-CHUNKS"
                )
            
            def generate_child_answer():
                print(f"📄 Thread 2: Using child chunks within page range...")
                child_chunks = self._retrieve_chunks_by_type(
                    corrected_question, n_results, chunk_type="child"
                )
                return self._generate_single_answer_standard(
                    corrected_question, child_chunks, model, "CHILD-CHUNKS"
                )
            
            # Execute both answer generations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                parent_future = executor.submit(generate_parent_answer)
                child_future = executor.submit(generate_child_answer)
                
                parent_answer_data = parent_future.result()
                child_answer_data = child_future.result()
            
            print(f"✅ Page range mode parallel generation complete")
            return parent_answer_data, child_answer_data
            
        else:
            # Original logic for other modes (collection, pdf, group)
            print(f"\n🔄 {current_mode.upper()} Mode: Generating dual answers with AUTO-TOC vs NO-TOC...")
            
            def generate_toc_answer():
                print(f"📊 Thread 1: Auto-TOC routing enabled...")
                toc_chunks = self._retrieve_relevant_chunks_with_setting(
                    corrected_question, n_results, use_auto_toc=True
                )
                return self._generate_single_answer_standard(
                    corrected_question, toc_chunks, model, "AUTO-TOC"
                )
            
            def generate_non_toc_answer():
                print(f"📄 Thread 2: Auto-TOC routing disabled...")
                non_toc_chunks = self._retrieve_relevant_chunks_with_setting(
                    corrected_question, n_results, use_auto_toc=False
                )
                return self._generate_single_answer_standard(
                    corrected_question, non_toc_chunks, model, "NO-TOC"
                )
            
            # Execute both answer generations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                toc_future = executor.submit(generate_toc_answer)
                non_toc_future = executor.submit(generate_non_toc_answer)
                
                toc_answer_data = toc_future.result()
                non_toc_answer_data = non_toc_future.result()
            
            print(f"✅ {current_mode.upper()} mode parallel generation complete")
            return toc_answer_data, non_toc_answer_data
    
    def _retrieve_chunks_by_type(self, question: str, n_results: int, chunk_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks of a specific type within the current context mode.
        
        Args:
            question: User's question
            n_results: Number of results to return
            chunk_type: "parent" or "child"
            
        Returns:
            List of chunk dictionaries
        """
        try:
            current_mode = self.current_context["mode"]
            
            if current_mode == "toc_section":
                # Query within TOC section with specific chunk type
                results = self.query_tool.query_by_toc_section(
                    query=question,
                    toc_sections=self.current_context["toc_sections"],
                    pdf_filenames=self.current_context.get("pdf_filenames"),
                    n_results=n_results,
                    chunk_type=chunk_type,
                    use_hyde=True
                )
                
            elif current_mode == "page_range":
                # Query within page range with specific chunk type
                pdf_filename = self.current_context["pdf_filenames"][0] if self.current_context.get("pdf_filenames") else None
                page_range = self.current_context.get("page_range", (1, 10))  # Default fallback
                
                results = self.query_tool.query_by_page_range(
                    query=question,
                    pdf_filename=pdf_filename,
                    start_page=page_range[0],
                    end_page=page_range[1],
                    n_results=n_results,
                    chunk_type=chunk_type,
                    use_hyde=False  # Keep consistent with existing page range logic
                )
                
            else:
                # Fallback to regular retrieval for other modes
                print(f"Warning: _retrieve_chunks_by_type called for unsupported mode: {current_mode}")
                return self._retrieve_relevant_chunks_with_setting(question, n_results, use_auto_toc=True)
            
            # Extract and return the results
            chunks = results.get("results", [])
            print(f"📊 Retrieved {len(chunks)} {chunk_type} chunks for {current_mode} mode")
            return chunks
            
        except Exception as e:
            print(f"❌ Error retrieving {chunk_type} chunks: {e}")
            return []
    
    def _retrieve_relevant_chunks_with_setting(self, question: str, n_results: int, use_auto_toc: bool) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with specific auto-TOC setting."""
        try:
            content_type_request = self._detect_content_type_request(question)
            
            # Determine chunk type for search
            if content_type_request == "parent":
                chunk_type = "parent"
            else:
                chunk_type = "child"
            
            # Adjust n_results if searching for specific content types
            if content_type_request in ["table", "figure"]:
                search_results = n_results * 2
            else:
                search_results = n_results
            
            # Perform the search based on context mode
            if self.current_context["mode"] == "collection":
                results = self.query_tool.query_entire_collection(
                    query=question,
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=use_auto_toc
                )
            elif self.current_context["mode"] == "pdf":
                results = self.query_tool.query_specific_pdfs(
                    query=question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=use_auto_toc
                )
            elif self.current_context["mode"] == "toc_section":
                results = self.query_tool.query_by_toc_section(
                    query=question,
                    toc_sections=self.current_context["toc_sections"],  # Now expects a list
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True
                )
            elif self.current_context["mode"] == "page_range":
                results = self.query_tool.query_specific_pdfs(
                    query=question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results * 2,
                    chunk_type=chunk_type,
                    use_hyde=True
                )
                # Filter results by page range
                if self.current_context["page_range"]:
                    start_page, end_page = self.current_context["page_range"]
                    filtered_results = []
                    for result in results.get("results", []):
                        page_num = result.get("page_number", 0)
                        if start_page <= page_num <= end_page:
                            filtered_results.append(result)
                    results["results"] = filtered_results[:search_results]
            elif self.current_context["mode"] == "group":
                results = self.query_tool.query_specific_pdfs(
                    query=question,
                    pdf_filenames=self.current_context["pdf_filenames"],
                    n_results=search_results,
                    chunk_type=chunk_type,
                    use_hyde=True,
                    use_auto_toc=use_auto_toc
                )
            else:
                results = {"results": []}
            
            # Post-process results for content type filtering
            retrieved_chunks = results.get("results", [])
            
            if content_type_request in ["table", "figure"] and retrieved_chunks:
                # Filter chunks by detected content type (simplified for standard mode)
                filtered_chunks = []
                for chunk in retrieved_chunks:
                    content = chunk.get('content', '').lower()
                    
                    if content_type_request == "table" and ('|' in content or '\t' in content):
                        filtered_chunks.append(chunk)
                    elif content_type_request == "figure" and any(word in content for word in ['figure', 'image', 'chart']):
                        filtered_chunks.append(chunk)
                    elif content_type_request not in ["table", "figure"]:
                        filtered_chunks.append(chunk)
                
                if filtered_chunks:
                    return filtered_chunks[:n_results]
            
            return retrieved_chunks[:n_results]
            
        except Exception as e:
            print(f"Warning: Error retrieving chunks: {e}")
            return []
    
    def _generate_single_answer_standard(self, question: str, retrieved_chunks: List[Dict[str, Any]], 
                                       model: str, approach_name: str) -> Dict[str, Any]:
        """Generate a single answer using standard mode processing."""
        try:
            prompt = self._build_standard_context_prompt(question, retrieved_chunks)
            messages = [{"role": "user", "content": prompt}]
            
            context_stats = {
                "total_tokens": len(prompt.split()) * 1.3,
                "chunks_included": len(retrieved_chunks),
                "mode": "standard",
                "approach": approach_name
            }
            
            # Generate response
            response = self._make_chat_completion(model, messages, self.max_response_tokens)
            assistant_response_raw = response
            
            # Parse JSON response
            try:
                response_json = json.loads(assistant_response_raw)
                found_answer = response_json.get("found_answer", True)
                confidence = response_json.get("confidence", "medium")
                assistant_response = response_json.get("response", assistant_response_raw)
                context_stats["response_confidence"] = confidence
            except json.JSONDecodeError:
                found_answer = True
                confidence = "medium"
                assistant_response = assistant_response_raw
            
            return {
                "response": assistant_response,
                "chunks": retrieved_chunks,
                "context_stats": context_stats,
                "found_answer": found_answer,
                "confidence": confidence,
                "approach": approach_name
            }
            
        except Exception as e:
            print("Failed to generate answer", e)
            return {
                "response": f"Error generating {approach_name} response: {e}",
                "chunks": retrieved_chunks,
                "context_stats": {"approach": approach_name, "error": str(e)},
                "found_answer": False,
                "confidence": "low",
                "approach": approach_name
            }
    
    def _synthesize_answers(self, toc_answer_data: Dict[str, Any], non_toc_answer_data: Dict[str, Any], 
                           question: str, model: str = "gpt-4o", verbosity: str = "medium") -> Tuple[str, Dict[str, Any]]:
        """
        Use GPT-4o to synthesize the best aspects of both answers.
        """
        print(f"🧠 Synthesizing best aspects from both approaches...")
        
        synthesis_prompt = f"""You have been given the same question answered using two different document retrieval approaches:

APPROACH 1:
Confidence: {toc_answer_data['confidence']}
Chunks found: {len(toc_answer_data['chunks'])}

APPROACH 2:
Confidence: {non_toc_answer_data['confidence']}
Chunks found: {len(non_toc_answer_data['chunks'])}

VERBOSITY LEVEL: {verbosity.upper()}
- High: Provide comprehensive, detailed explanations with extensive background context
- Medium: Provide balanced explanations with moderate detail and context  
- Low: Provide concise, direct answers focused on essential information only

YOUR TASK:
Create a synthesized answer that combines the best aspects of both approaches.
1. Use proper markdown formatting (headers, lists, bold, italic, code blocks, etc.) to structure your answer clearly.
2. Maintain all citation tags in the exact format: <citation pdf_name="filename.pdf" page_number="123" chunk_id="chunk_identifier" cited_text="first few words only...">. Fix any citation formatting errors that you notice.
3. Adjust the length and detail level according to the {verbosity} verbosity setting above

CRITICAL CITATION FORMATTING RULES:
- Citation tags are INVISIBLE to users - anything you want the user to see must be OUTSIDE the citation tags
- The cited_text attribute should contain ONLY the first 5-7 words of the referenced text
- NEVER put full quotes, sentences, or explanations inside citation tags
- You can PARAPHRASE or directly quote - both are fine, just cite properly
- Place citation tags at the end of sentences

Examples:
PARAPHRASING: Nagel argues that human existence lacks ultimate meaning <citation pdf_name="nagel.pdf" page_number="4" chunk_id="abc123" cited_text="nothing we do now...">.
DIRECT QUOTE: "Nothing we do now will matter in a million years" <citation pdf_name="nagel.pdf" page_number="4" chunk_id="abc123" cited_text="Nothing we do now...">.

WRONG: <citation cited_text="Nothing we do now will matter in a million years, but if that is true...">

ORIGINAL QUESTION: {question}

APPROACH 1:
{toc_answer_data['response']}

APPROACH 2:
{non_toc_answer_data['response']}

REMINDER: Citation tags must be properly closed and contain only short reference text. Correct format:
<citation pdf_name="file.pdf" page_number="1" chunk_id="abc123" cited_text="first few words">

Provide your synthesized response in JSON format:
{{
  "synthesis_rationale": "Brief explanation of how you combined the responses",
  "synthesized_response": "Your combined response with all citation tags preserved and properly closed with >",
  "confidence": "high/medium/low",
  "primary_source": "approach_1/approach_2/balanced" 
}}"""

        try:
            # Use the specified model for synthesis
            response = self._make_chat_completion(model, [{"role": "user", "content": synthesis_prompt}], 10000)
            synthesis_result = response
            
            # Parse the synthesis response
            try:
                synthesis_json = json.loads(synthesis_result)
                synthesized_response = synthesis_json.get("synthesized_response", synthesis_result)
                synthesis_rationale = synthesis_json.get("synthesis_rationale", "")
                confidence = synthesis_json.get("confidence", "medium")
                primary_source = synthesis_json.get("primary_source", "balanced")
                
                print(f"🎯 Synthesis complete - Primary source: {primary_source.upper()}")
                
            except json.JSONDecodeError:
                print("⚠️  Failed to parse synthesis JSON, using raw response")
                synthesized_response = synthesis_result
                confidence = "medium"
                primary_source = "balanced"
                synthesis_rationale = ""
            
            # Combine context statistics
            combined_stats = {
                "synthesis_approach": True,
                "toc_chunks": len(toc_answer_data['chunks']),
                "non_toc_chunks": len(non_toc_answer_data['chunks']),
                "total_unique_chunks": len(set(
                    chunk.get('metadata', {}).get('chunk_id', f"chunk_{i}") 
                    for i, chunk in enumerate(toc_answer_data['chunks'] + non_toc_answer_data['chunks'])
                )),
                "toc_confidence": toc_answer_data['confidence'],
                "non_toc_confidence": non_toc_answer_data['confidence'],
                "synthesis_confidence": confidence,
                "primary_source": primary_source,
                "synthesis_rationale": synthesis_rationale,
                "total_tokens": toc_answer_data['context_stats'].get('total_tokens', 0) + 
                               non_toc_answer_data['context_stats'].get('total_tokens', 0),
                "synthesis_model": model
            }
            
            return synthesized_response, combined_stats
            
        except Exception as e:
            print(f"⚠️  Synthesis failed: {e}")
            # Fallback: return the response with higher confidence
            if toc_answer_data['confidence'] == 'high' or (
                toc_answer_data['confidence'] == 'medium' and non_toc_answer_data['confidence'] == 'low'
            ):
                fallback_response = toc_answer_data['response']
                fallback_stats = toc_answer_data['context_stats']
                fallback_stats['synthesis_fallback'] = 'auto_toc_selected'
            else:
                fallback_response = non_toc_answer_data['response']  
                fallback_stats = non_toc_answer_data['context_stats']
                fallback_stats['synthesis_fallback'] = 'broad_search_selected'
            
            return fallback_response, fallback_stats

    def _make_chat_completion(self, model: str, messages: list, max_tokens: int = 4000) -> object:
        """
        Make a chat completion request with model-specific parameters.
        
        Args:
            model: Model name ("gpt-5", "gpt-4o", "claude", or "claude-3-5-sonnet-20241022")
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            
        Returns:
            OpenAI API response object
        """

        if model == "gpt-5":
            # GPT-5 uses max_completion_tokens and doesn't support temperature
            max_tokens = 15000
            return self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"}
            ).choices[0].message.content.strip()
        elif "gpt-4" in model:
            # GPT-4 and other models use max_tokens and support temperature
            max_tokens = 10000
            return self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            ).choices[0].message.content.strip()
        else:
            max_tokens = 10000
            # Claude doesn't support response_format, so we need to handle JSON parsing manually
            
            # Determine which Claude model to use
            claude_model = "claude-3-7-sonnet-latest" if model == "claude-3-5-sonnet-20241022" else "claude-sonnet-4-20250514"
            
            for attempt in range(3):  # Try up to 3 times
                try:
                    response_text = self.anthropic_client.messages.create(
                        model=claude_model,
                        max_tokens=max_tokens,
                        messages=messages
                    ).content[0].text
                    
                    # Clean up Claude's response (remove backticks, backslashes, and trim whitespace)
                    response_text = response_text.strip()
                    
                    if response_text.startswith('```json') and response_text.endswith('```'):
                        response_text = response_text[7:-3].strip()
                    elif response_text.startswith('```') and response_text.endswith('```'):
                        response_text = response_text[3:-3].strip()
                    elif response_text.startswith('`') and response_text.endswith('`'):
                        response_text = response_text[1:-1].strip()
                    elif response_text.startswith("json"):
                        response_text = response_text[4:].strip()
                    
                    # Remove trailing backslashes and whitespace that Claude sometimes adds
                    # Handle patterns like "}\\\\" or "}\\\n\\" 
                    import re
                    response_text = re.sub(r'[\\\s]*$', '', response_text).strip()
                    
                    # Try to parse as JSON to validate format
                    json.loads(response_text)
                    return response_text
                    
                except json.JSONDecodeError:
                    if attempt == 2:  # Last attempt
                        print(f"⚠️  Claude response parsing failed after 3 attempts")
                        # Return the raw response on final failure
                        return response_text
                    else:
                        print(f"⚠️  Claude JSON parsing failed, retrying (attempt {attempt + 1}/3)")
                        # Add JSON format reminder to the last message
                        if messages and messages[-1].get("role") == "user":
                            messages[-1]["content"] += "\n\nIMPORTANT: Please ensure your response is valid JSON format."
                        continue
                except Exception as e:
                    if attempt == 2:
                        print(f"⚠️  Claude API error after 3 attempts: {e}")
                        raise e
                    else:
                        print(f"⚠️  Claude API error, retrying (attempt {attempt + 1}/3): {e}")
                        continue

    def _make_text_completion(self, model: str, messages: list, max_tokens: int = 4000) -> str:
        """
        Make a text completion request without JSON formatting.
        
        Args:
            model: Model name ("gpt-5", "gpt-4o", "claude", or "claude-3-5-sonnet-20241022")
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            
        Returns:
            Plain text response
        """
        if model == "gpt-5":
            max_tokens = 15000
            return self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=max_tokens
            ).choices[0].message.content.strip()
        elif "gpt-4" in model:
            max_tokens = 10000
            return self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            ).choices[0].message.content.strip()
        else:
            # Claude - simple text completion
            max_tokens = 10000
            # Determine which Claude model to use
            claude_model = "claude-3-5-sonnet-20241022" if model == "claude-3-5-sonnet-20241022" else "claude-sonnet-4-20250514"
            
            response_text = self.anthropic_client.messages.create(
                model=claude_model,
                max_tokens=max_tokens,
                messages=messages
            ).content[0].text
            return response_text.strip()

    
    def ask_question(self, question: str, model: str = "gpt-5", verbosity: str = "medium") -> str:
        """
        Ask a question and get an AI response using dual-answer synthesis approach.
        Generates two answers (with and without auto-TOC) in parallel, then synthesizes them.
        
        Args:
            question: User's question
            model: Model to use for answer generation
            
        Returns:
            AI assistant's synthesized response
        """
        try:
            # Generate dual answers in parallel
            toc_answer_data, non_toc_answer_data = self._generate_dual_answers_parallel(question, model)
            
            print(f"📊 Answer comparison:")
            print(f"   AUTO-TOC: {toc_answer_data['confidence']} confidence, {len(toc_answer_data['chunks'])} chunks")
            print(f"   NO-TOC: {non_toc_answer_data['confidence']} confidence, {len(non_toc_answer_data['chunks'])} chunks")
            
            # Synthesize the best aspects of both answers
            synthesized_response, combined_stats = self._synthesize_answers(
                toc_answer_data, non_toc_answer_data, question, model, verbosity
            )
            
            # Use all unique chunks from both approaches for history
            all_chunks = toc_answer_data['chunks'] + non_toc_answer_data['chunks']
            unique_chunks = []
            seen_chunk_ids = set()
            for chunk in all_chunks:
                chunk_id = chunk.get('metadata', {}).get('chunk_id', f"chunk_{len(unique_chunks)}")
                if chunk_id not in seen_chunk_ids:
                    unique_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
            
            # Add to conversation history
            self._add_to_history(question, synthesized_response, self.current_context, 
                               unique_chunks, combined_stats)
            
            # Show completion status
            confidence = combined_stats.get("synthesis_confidence", "medium")
            confidence_emoji = {"high": "🎯", "medium": "📊", "low": "⚠️"}.get(confidence, "📊")
            
            primary_source = combined_stats.get("primary_source", "balanced")
            if combined_stats.get("synthesis_fallback"):
                print(f"🔄 Synthesis failed, used fallback: {combined_stats['synthesis_fallback']} {confidence_emoji}")
            else:
                print(f"✨ Synthesis complete - {primary_source.upper()} approach {confidence_emoji}")
            
            # Easter egg: randomly append a cute message from Sukhm
            synthesized_response = self._add_random_cute_message(synthesized_response)
                
            return synthesized_response
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error generating a response: {e}"
            self._add_to_history(question, error_response, self.current_context)
            return error_response
    
    def _build_conversation_context(self, token_budget: int) -> str:
        """Build conversation context within token budget (enhanced mode only)."""
        if not self.enhanced_mode or not self.conversation_history:
            return "This is the start of our conversation."
    
        print(self.conversation_history)
        
        relevant_history = [
            turn for turn in self.conversation_history
            if (turn["context"]["mode"] == self.current_context["mode"] and
                turn["context"].get("collection_name") == self.current_context["collection_name"] and
                not turn.get("is_cleared", False))
        ]
        
        # Debug output for enhanced mode
        total_turns = len(self.conversation_history)
        relevant_turns = len(relevant_history)
        print(f"🧠 Enhanced Chat History: {relevant_turns} collection turns from {total_turns} total (collection: {self.current_context['collection_name']}, mode: {self.current_context['mode']})")
        if relevant_history:
            last_turn = relevant_history[-1]
            print(f"   Last relevant - Human: \"{last_turn['user_message'][:50]}{'...' if len(last_turn['user_message']) > 50 else ''}\"")
            print(f"   Last relevant - Assistant: \"{last_turn['assistant_response'][:50]}{'...' if len(last_turn['assistant_response']) > 50 else ''}\"")
        
        conversation_parts = []
        current_tokens = 0
        
        for turn in reversed(relevant_history[-15:]):
            user_msg = f"Human: {turn['user_message']}"
            assistant_msg = f"Assistant: {turn['assistant_response']}"
            
            turn_tokens = self._count_tokens(user_msg + assistant_msg)
            
            if current_tokens + turn_tokens <= token_budget:
                conversation_parts.insert(0, assistant_msg)
                conversation_parts.insert(0, user_msg)
                current_tokens += turn_tokens
            else:
                break
        
        return "\n".join(conversation_parts) if conversation_parts else "This is the start of our conversation."
    
    def _perform_fallback_search(self, question: str, max_results: int) -> list:
        """Perform a broader search when initial search yields limited results."""
        try:
            # Use collection-wide search without TOC restrictions
            if self.current_context["mode"] in ["toc_section", "pdf", "page_range"]:
                # Expand to full PDF search
                pdf_filenames = self.current_context.get("pdf_filenames")
                results = self.query_tool.query_specific_pdfs(
                    query=question,
                    pdf_filenames=pdf_filenames,
                    n_results=max_results,
                    chunk_type="child",
                    use_hyde=False,
                    use_auto_toc=False  # Disable TOC routing for broader search
                )
            else:
                # Full collection search
                results = self.query_tool.query_entire_collection(
                    query=question,
                    n_results=max_results,
                    chunk_type="child",
                    use_hyde=False,
                    use_auto_toc=False  # Disable TOC routing for broader search
                )
            
            return results.get("results", [])
            
        except Exception as e:
            print(f"Warning: Fallback search failed: {e}")
            return []
    
    def set_chat_mode(self, mode: str, **kwargs) -> None:
        """Set the chat mode and context."""
        self.current_context["mode"] = mode
        
        if mode == "pdf":
            pdf_filename = kwargs.get("pdf_filename")
            self.current_context["pdf_filenames"] = [pdf_filename] if pdf_filename else None
            self.current_context["description"] = f"PDF: {pdf_filename}"
            
        elif mode == "toc_section":
            pdf_filename = kwargs.get("pdf_filename")
            toc_sections = kwargs.get("toc_sections", [])
            # For backward compatibility, also accept single toc_section
            if not toc_sections and "toc_section" in kwargs:
                toc_sections = [kwargs["toc_section"]]
            self.current_context["pdf_filenames"] = [pdf_filename] if pdf_filename else None
            self.current_context["toc_sections"] = toc_sections
            if len(toc_sections) == 1:
                self.current_context["description"] = f"TOC Section '{toc_sections[0]}' in {pdf_filename}"
            else:
                self.current_context["description"] = f"TOC Sections {toc_sections} in {pdf_filename}"
            
        elif mode == "page_range":
            pdf_filename = kwargs.get("pdf_filename")
            page_range = kwargs.get("page_range")
            self.current_context["pdf_filenames"] = [pdf_filename] if pdf_filename else None
            self.current_context["page_range"] = page_range
            self.current_context["description"] = f"Pages {page_range[0]}-{page_range[1]} in {pdf_filename}"
            
        elif mode == "group":
            pdf_filenames = kwargs.get("pdf_filenames", [])
            self.current_context["pdf_filenames"] = pdf_filenames
            self.current_context["description"] = f"PDF Group: {', '.join(pdf_filenames[:3])}" + ("..." if len(pdf_filenames) > 3 else "")
            
        elif mode == "collection":
            self.current_context["pdf_filenames"] = None
            self.current_context["toc_sections"] = None  # Changed from toc_section to toc_sections
            self.current_context["page_range"] = None
            self.current_context["description"] = f"Entire collection '{self.collection_name}'"
        
        print(f"🎯 Chat mode: {self.current_context['description']}")
    
    def get_chat_history_for_collection(self) -> List[Dict[str, Any]]:
        """Get chat history for the current collection that isn't cleared."""
        collection_messages = []
        for message_pair in self.conversation_history:
            if (message_pair.get("context", {}).get("collection_name") == self.collection_name and 
                not message_pair.get("is_cleared", False) and not message_pair.get("is_summary", False)):
                
                # Add user message
                if message_pair.get("user_message"):
                    collection_messages.append({
                        "type": "user",
                        "content": message_pair.get("user_message"),
                        "timestamp": message_pair.get("timestamp")
                    })
                
                # Add assistant response
                if message_pair.get("assistant_response"):
                    collection_messages.append({
                        "type": "assistant", 
                        "content": message_pair.get("assistant_response"),
                        "timestamp": message_pair.get("timestamp"),
                        "sources": message_pair.get("retrieved_chunks", [])
                    })
        
        return collection_messages
    
    def clear_chat_history_for_collection(self) -> int:
        """Mark chat history as cleared for the current collection. Returns number of messages cleared."""
        cleared_count = 0
        for message_pair in self.conversation_history:
            if message_pair.get("context", {}).get("collection_name") == self.collection_name:
                message_pair["is_cleared"] = True
                cleared_count += 1
        
        # Save updated history
        self._save_history()
        
        print(f"🧹 Marked {cleared_count} message pairs as cleared for collection: {self.collection_name}")
        return cleared_count
    
    def _get_compacted_history(self, collection_history: List[Dict]) -> List[Dict]:
        """Smart compaction that only summarizes when needed and persists summaries."""
        # Calculate total tokens in conversation history
        total_tokens = self._calculate_history_tokens(collection_history)
        print("total tokens of conversation history", total_tokens)
        
        # Token threshold for triggering compaction (roughly 8000 tokens)
        TOKEN_THRESHOLD = 5000
        
        if total_tokens <= TOKEN_THRESHOLD:
            return collection_history
        
        # Check if we already have a recent summary by looking at recent portion
        recent_tokens = 0
        recent_count = 0
        for turn in reversed(collection_history):
            turn_tokens = self._calculate_turn_tokens(turn)
            recent_tokens += turn_tokens
            recent_count += 1
            if recent_tokens >= 2000:  # Check last ~2000 tokens for recent summary
                break
        
        has_recent_summary = any(turn.get("is_summary", False) for turn in collection_history[-recent_count:])
        
        if has_recent_summary:
            # We already have a summary in recent history, no need to re-summarize
            return collection_history
        
        # Check if there are enough new tokens since last summary to warrant new summarization  
        last_summary_idx = None
        for i, turn in enumerate(collection_history):
            if turn.get("is_summary", False):
                last_summary_idx = i
        
        if last_summary_idx is not None:
            # Count new tokens since last summary
            new_turns = collection_history[last_summary_idx + 1:]
            new_tokens = self._calculate_history_tokens(new_turns)
            if new_tokens < 3000:  # Not enough new content to summarize again
                return collection_history
        
        # Perform compaction
        return self._create_conversation_summary(collection_history)
    
    def _calculate_history_tokens(self, history: List[Dict]) -> int:
        """Calculate total tokens for a list of conversation turns."""
        total_tokens = 0
        for turn in history:
            total_tokens += self._calculate_turn_tokens(turn)
        return total_tokens
    
    def _calculate_turn_tokens(self, turn: Dict) -> int:
        """Calculate tokens for a single conversation turn."""
        user_msg = turn.get("user_message", "")
        assistant_msg = turn.get("assistant_response", "")
        return self._count_tokens(user_msg + assistant_msg)
    
    def _create_conversation_summary(self, conversation_history: List[Dict]) -> List[Dict]:
        """Create a summary of older conversation turns."""
        try:
            # Keep recent turns that total around 2000 tokens, summarize older ones
            recent_turns = []
            recent_tokens = 0
            target_recent_tokens = 2000
            
            for turn in reversed(conversation_history):
                turn_tokens = self._calculate_turn_tokens(turn)
                if recent_tokens + turn_tokens <= target_recent_tokens or len(recent_turns) == 0:
                    recent_turns.insert(0, turn)
                    recent_tokens += turn_tokens
                else:
                    break
            
            # Everything else goes to older_turns
            recent_count = len(recent_turns)
            older_turns = conversation_history[:-recent_count] if recent_count > 0 else conversation_history
            
            # Skip turns that are already summaries
            turns_to_summarize = [turn for turn in older_turns if not turn.get("is_summary", False)]
            
            if len(turns_to_summarize) < 2:  # Not enough to summarize (reduced from 3)
                return conversation_history
            
            # Build conversation text for summarization
            conversation_text = ""
            topic_keywords = set()
            for turn in turns_to_summarize:
                conversation_text += f"Human: {turn['user_message']}\n"
                conversation_text += f"Assistant: {turn['assistant_response']}\n\n"
                # Extract potential topic words for better summary
                words = turn['user_message'].lower().split()
                topic_keywords.update(word for word in words if len(word) > 4)
            
            # Create summarization prompt
            summary_prompt = f"""Summarize this conversation history, focusing on:
1. Main topics discussed: {', '.join(list(topic_keywords)[:10])}
2. Key arguments, conclusions, and insights
3. Important context for future questions

Keep the summary concise (5-6 sentences) but informative enough to maintain conversation continuity.

Conversation:
{conversation_text}"""

            # Use LLM to create summary (plain text, not JSON)
            messages = [{"role": "user", "content": summary_prompt}]
            summary = self._make_text_completion("gpt-4o", messages, 300)
            
            # Create summary entry with timestamp of the oldest summarized message
            summary_entry = {
                "timestamp": turns_to_summarize[0]["timestamp"], 
                "context": self.current_context.copy(),
                "user_message": f"[Summary of {len(turns_to_summarize)} earlier messages]",
                "assistant_response": summary,
                "retrieved_chunks": 0,
                "enhanced_mode": False,
                "is_summary": True,
                "summarized_count": len(turns_to_summarize)
            }
            
            # Keep any existing summaries + new summary + recent turns
            existing_summaries = [turn for turn in older_turns if turn.get("is_summary", False)]
            compacted = existing_summaries + [summary_entry] + recent_turns
            
            # Update the persistent conversation history
            self._update_summarized_history(turns_to_summarize, summary_entry)
            
            print(f"🗜️ Summarized {len(turns_to_summarize)} turns. History now: {len(compacted)} entries")
            return compacted
            
        except Exception as e:
            print(f"⚠️ Failed to create conversation summary: {e}")
            # Fallback: keep recent turns up to ~2000 tokens
            fallback_turns = []
            fallback_tokens = 0
            for turn in reversed(conversation_history):
                turn_tokens = self._calculate_turn_tokens(turn)
                if fallback_tokens + turn_tokens <= 2000 or len(fallback_turns) == 0:
                    fallback_turns.insert(0, turn)
                    fallback_tokens += turn_tokens
                else:
                    break
            return fallback_turns
    
    def _update_summarized_history(self, summarized_turns: List[Dict], summary_entry: Dict):
        """Mark the summarized turns in the persistent history and add summary."""
        # Mark original turns as summarized (don't delete them, just mark them)
        for turn in self.conversation_history:
            for summarized_turn in summarized_turns:
                if (turn.get("timestamp") == summarized_turn.get("timestamp") and
                    turn.get("user_message") == summarized_turn.get("user_message")):
                    turn["is_summarized"] = True
        
        # Add the summary entry to persistent history
        self.conversation_history.append(summary_entry)
        self._save_history()
    
    def show_context_info(self) -> None:
        """Show current chat context information."""
        print(f"\n📋 Current Chat Context:")
        print(f"   Mode: {self.current_context['mode']}")
        print(f"   Scope: {self.current_context['description']}")
        print(f"   Enhanced: {'Yes' if self.enhanced_mode else 'No'}")
        
        if self.current_context["pdf_filenames"]:
            print(f"   PDFs: {', '.join(self.current_context['pdf_filenames'])}")
        
        if self.current_context["toc_sections"]:
            if len(self.current_context["toc_sections"]) == 1:
                print(f"   TOC Section: {self.current_context['toc_sections'][0]}")
            else:
                print(f"   TOC Sections: {', '.join(self.current_context['toc_sections'])}")
            
        if self.current_context["page_range"]:
            print(f"   Page Range: {self.current_context['page_range'][0]}-{self.current_context['page_range'][1]}")
    
    def show_context_stats(self) -> None:
        """Show detailed context utilization statistics."""
        if self.conversation_history:
            recent_turn = self.conversation_history[-1]
            stats = recent_turn.get("context_stats", {})
            
            print(f"\n📊 Last Query Context Statistics:")
            if self.enhanced_mode:
                print(f"   Total tokens used: {stats.get('total_tokens', 0):,}")
                print(f"   Context utilization: {stats.get('context_utilization_pct', 0)}%")
                print(f"   Chunks included: {stats.get('chunks_included', 0)}")
                print(f"   Images included: {stats.get('images_included', 0)}")
                print(f"   Tables included: {stats.get('tables_included', 0)}")
                print(f"   Question type: {stats.get('complexity', 'unknown').upper()}")
            else:
                print(f"   Approximate tokens: {int(stats.get('total_tokens', 0)):,}")
                print(f"   Chunks included: {stats.get('chunks_included', 0)}")
                print(f"   Mode: {stats.get('mode', 'standard')}")
        else:
            print("📭 No context statistics available yet.")
    
    def list_toc_sections(self, pdf_filename: Optional[str] = None) -> None:
        """
        List available TOC sections for PDFs in the collection.
        
        Args:
            pdf_filename: Optional PDF filename to get sections for. If None, shows all PDFs.
        """
        try:
            toc_sections = self.query_tool.get_toc_sections()
            
            if not toc_sections:
                print("📋 No TOC sections found in the collection.")
                return
            
            print(f"\n📋 Available TOC Sections:")
            print("=" * 60)
            
            if pdf_filename:
                # Show sections for specific PDF
                if pdf_filename in toc_sections:
                    sections = toc_sections[pdf_filename]
                    print(f"\n📄 {pdf_filename}:")
                    if sections:
                        for i, section in enumerate(sections, 1):
                            if isinstance(section, dict):
                                page_info = f" (page {section['page']})" if section['page'] > 0 else ""
                                print(f"  {i:2d}. {section['title']}{page_info}")
                            else:
                                # Backward compatibility for old format
                                print(f"  {i:2d}. {section}")
                    else:
                        print("  No TOC sections found.")
                else:
                    print(f"❌ PDF '{pdf_filename}' not found in collection.")
                    available_pdfs = list(toc_sections.keys())
                    if available_pdfs:
                        print(f"Available PDFs: {', '.join(available_pdfs[:5])}" + 
                              ("..." if len(available_pdfs) > 5 else ""))
            else:
                # Show sections for all PDFs
                for pdf_name, sections in sorted(toc_sections.items()):
                    print(f"\n📄 {pdf_name}:")
                    if sections:
                        for i, section in enumerate(sections, 1):
                            if isinstance(section, dict):
                                page_info = f" (page {section['page']})" if section['page'] > 0 else ""
                                print(f"  {i:2d}. {section['title']}{page_info}")
                            else:
                                # Backward compatibility for old format
                                print(f"  {i:2d}. {section}")
                    else:
                        print("  No TOC sections found.")
            
            print("\n💡 Usage examples:")
            if pdf_filename and pdf_filename in toc_sections and toc_sections[pdf_filename]:
                example_section = toc_sections[pdf_filename][0]
                section_title = example_section["title"] if isinstance(example_section, dict) else example_section
                print(f"   mode toc {pdf_filename} \"{section_title}\"")
            elif toc_sections:
                first_pdf = next(iter(toc_sections))
                if toc_sections[first_pdf]:
                    example_section = toc_sections[first_pdf][0]
                    section_title = example_section["title"] if isinstance(example_section, dict) else example_section
                    print(f"   mode toc {first_pdf} \"{section_title}\"")
            
        except Exception as e:
            print(f"❌ Error listing TOC sections: {e}")
    
    def interactive_chat(self) -> None:
        """Start interactive chat session."""
        mode_desc = "Enhanced Multimodal" if self.enhanced_mode else "Standard"
        print(f"\n💬 {mode_desc} PDF Chat Mode")
        print("=" * 60)
        
        # Show collection info
        try:
            info = self.query_tool.get_collection_info()
            if info:
                stats = info["statistics"]
                print(f"Collection: {info['collection_name']}")
                print(f"PDFs: {stats['pdf_count']} | Chunks: {stats['total_chunks']}")
                if info["available_pdfs"]:
                    print(f"Available PDFs: {', '.join(info['available_pdfs'][:5])}" + 
                          ("..." if len(info["available_pdfs"]) > 5 else ""))
        except:
            pass
        
        self.show_context_info()
        
        print(f"\nCommands:")
        print(f"  ask <question>              - Ask a question")
        print(f"  toc [pdf_filename]          - List TOC sections (all PDFs or specific PDF)")
        if self.enhanced_mode:
            print(f"  stats                       - Show context utilization stats")
        print(f"  mode collection             - Chat with entire collection")
        print(f"  mode pdf <filename>         - Chat with specific PDF")
        print(f"  mode toc <pdf> <section>    - Chat with TOC section")
        print(f"  mode pages <pdf> <start> <end> - Chat with page range")
        print(f"  mode group <pdf1> <pdf2>... - Chat with group of PDFs")
        print(f"  context                     - Show current context")
        print(f"  history                     - Show recent conversation")
        print(f"  clear                       - Clear conversation history")
        print(f"  help                        - Show this help")
        print(f"  quit                        - Exit chat")
        print("-" * 60)
        
        if self.enhanced_mode:
            print(f"🖼️  Enhanced features: Tables as markdown, figures as images")
        
        print(f"🎯 Smart search: Ask about 'tables', 'figures', or 'charts' for targeted results")
        
        while True:
            try:
                user_input = input(f"\n💬 [{self.current_context['mode']}] Ask> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ["quit", "exit", "bye"]:
                    print("👋 Thanks for chatting! Conversation saved.")
                    if self.enhanced_mode and hasattr(self, 'pdf_cache'):
                        for doc in self.pdf_cache.values():
                            doc.close()
                    break
                
                elif command == "help":
                    print(f"\nCommands:")
                    print(f"  ask <question>              - Ask a question")
                    print(f"  toc [pdf_filename]          - List TOC sections (all PDFs or specific PDF)")
                    if self.enhanced_mode:
                        print(f"  stats                       - Show context stats")
                    print(f"  mode collection             - Chat with entire collection")
                    print(f"  mode pdf <filename>         - Chat with specific PDF")
                    print(f"  mode toc <pdf> <section>    - Chat with TOC section")
                    print(f"  mode pages <pdf> <start> <end> - Chat with page range")
                    print(f"  mode group <pdf1> <pdf2>... - Chat with group of PDFs")
                    print(f"  context                     - Show current context")
                    print(f"  history                     - Show recent conversation")
                    print(f"  clear                       - Clear conversation history")
                    print(f"  help                        - Show this help")
                    print(f"  quit                        - Exit chat")
                    print(f"\n🎯 Smart search: Ask about 'tables', 'figures', or 'charts' for targeted results")
                
                elif command == "context":
                    self.show_context_info()
                
                elif command == "toc":
                    # List TOC sections
                    if len(parts) >= 2:
                        # Show TOC for specific PDF
                        pdf_filename = parts[1]
                        self.list_toc_sections(pdf_filename)
                    else:
                        # Show TOC for all PDFs
                        self.list_toc_sections()
                
                elif command == "stats" and self.enhanced_mode:
                    self.show_context_stats()
                
                elif command == "history":
                    recent = self.conversation_history[-5:]
                    if recent:
                        print(f"\n📚 Recent Conversation:")
                        for turn in recent:
                            print(f"  👤 {turn['user_message'][:60]}...")
                            print(f"  🤖 {turn['assistant_response'][:60]}...")
                            print()
                    else:
                        print("📭 No conversation history yet.")
                
                elif command == "clear":
                    self.conversation_history = []
                    self._save_history()
                    print("🗑️  Conversation history cleared.")
                
                elif command == "mode" and len(parts) >= 2:
                    mode = parts[1].lower()
                    
                    if mode == "collection":
                        self.set_chat_mode("collection")
                    
                    elif mode == "pdf" and len(parts) >= 3:
                        pdf_filename = parts[2]
                        self.set_chat_mode("pdf", pdf_filename=pdf_filename)
                    
                    elif mode == "toc" and len(parts) >= 4:
                        pdf_filename = parts[2]
                        toc_section = " ".join(parts[3:])
                        self.set_chat_mode("toc_section", pdf_filename=pdf_filename, toc_section=toc_section)
                    
                    elif mode == "pages" and len(parts) >= 5:
                        pdf_filename = parts[2]
                        try:
                            start_page = int(parts[3])
                            end_page = int(parts[4])
                            self.set_chat_mode("page_range", pdf_filename=pdf_filename, 
                                             page_range=(start_page, end_page))
                        except ValueError:
                            print("❌ Invalid page range. Use: mode pages <pdf> <start_page> <end_page>")
                    
                    elif mode == "group" and len(parts) >= 3:
                        pdf_filenames = parts[2:]
                        self.set_chat_mode("group", pdf_filenames=pdf_filenames)
                    
                    else:
                        print("❌ Invalid mode command. Type 'help' for usage.")
                
                elif command == "ask" and len(parts) > 1:
                    question = " ".join(parts[1:])
                    response = self.ask_question(question)
                    print(f"\n🤖 {response}")
                
                else:
                    # Treat input as a direct question
                    response = self.ask_question(user_input)
                    print(f"\n🤖 {response}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Conversation saved.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    def generate_nice_message(self, user_question: str, model: str = "gpt-5") -> str:
        """
        Generate a nice, friendly message using the AI model.
        
        Args:
            user_question: The user's question or context
            model: Model to use for generation
            
        Returns:
            A nice, friendly response message
        """
        # TODO: Add your prompt here
        prompt = f"""You are an AI assistant in a PDF chat application that provides encouraging feedback to users.

        CONTEXT:
        - Current user: Kris
        - Application: PDF document Q&A chat interface
        - Your role: Generate thoughtful, genuine compliments about the user's questions

        TASK:
        Analyze the question below and provide a brief, authentic compliment that recognizes one of these aspects:
        - Intellectual curiosity or depth of inquiry
        - Critical thinking or analytical approach
        - Specificity and clarity of the question
        - Creative or unique perspective
        - Practical application of the content

        Keep your response:
        - Genuine and specific (avoid generic praise)
        - 1-2 sentences maximum
        - Focused on the question's quality, not just effort
        - Professional yet warm in tone

        USER'S QUESTION:
        "{user_question}"

        COMPLIMENT:"""

        try:
            # Use the chat completion method 
            messages = [{"role": "user", "content": prompt}]
            response = self._make_chat_completion(model, messages, 1000)
            
            # For non-JSON responses, return directly
            if model not in ["gpt-5", "gpt-4o"]:
                return response
            else:
                # Try to parse JSON response, fallback to raw if needed
                try:
                    import json
                    response_json = json.loads(response)
                    return response_json.get("message", response)
                except json.JSONDecodeError:
                    return response
                    
        except Exception as e:
            print(f"❌ Error generating nice message: {e}")
            return "Thanks for using the PDF chat system!"


def main():
    """Main function for PDF chat interface."""
    parser = argparse.ArgumentParser(description="Comprehensive PDF Chat with RAG")
    parser.add_argument("--collection", "-c", required=True,
                       help="ChromaDB collection name")
    parser.add_argument("--vector-store-dir", default="./chroma_db",
                       help="Directory containing ChromaDB data")
    parser.add_argument("--gpt-token", default="gpt_token.txt",
                       help="Path to GPT token file")
    parser.add_argument("--history-file", default="chat_history.json",
                       help="Path to conversation history file")
    
    # Chat mode options
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--question", "-q",
                       help="Ask a single question (non-interactive)")
    
    # Enhanced mode
    parser.add_argument("--enhanced", action="store_true",
                       help="Enable enhanced mode with multimodal support and optimized context")
    
    # Context options
    parser.add_argument("--pdf",
                       help="Focus chat on specific PDF")
    parser.add_argument("--toc-section",
                       help="Focus chat on specific TOC section (requires --pdf)")
    parser.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"),
                       help="Focus chat on page range (requires --pdf)")
    parser.add_argument("--pdfs", nargs="+",
                       help="Focus chat on group of PDFs")
    
    # TOC listing option
    parser.add_argument("--list-toc", metavar="PDF_FILENAME",
                       help="List TOC sections for specific PDF (or all PDFs if no filename provided)")
    parser.add_argument("--list-all-toc", action="store_true",
                       help="List TOC sections for all PDFs in collection")
    
    args = parser.parse_args()
    
    try:
        # Initialize chat session
        chat_session = PDFChatSession(
            collection_name=args.collection,
            vector_store_dir=args.vector_store_dir,
            gpt_token_path=args.gpt_token,
            history_file=args.history_file,
            enhanced_mode=args.enhanced
        )
        
        # Set context based on arguments
        if args.pdfs:
            chat_session.set_chat_mode("group", pdf_filenames=args.pdfs)
        elif args.pages and args.pdf:
            chat_session.set_chat_mode("page_range", pdf_filename=args.pdf, 
                                     page_range=tuple(args.pages))
        elif args.toc_section and args.pdf:
            chat_session.set_chat_mode("toc_section", pdf_filename=args.pdf, 
                                     toc_section=args.toc_section)
        elif args.pdf:
            chat_session.set_chat_mode("pdf", pdf_filename=args.pdf)
        else:
            chat_session.set_chat_mode("collection")
        
        # Handle TOC listing
        if args.list_all_toc:
            chat_session.list_toc_sections()
            return 0
        elif args.list_toc:
            chat_session.list_toc_sections(args.list_toc)
            return 0
        
        # Handle single question or interactive mode
        elif args.question:
            print(f"❓ Question: {args.question}")
            response = chat_session.ask_question(args.question)
            print(f"\n🤖 Response:\n{response}")
            if args.enhanced:
                chat_session.show_context_stats()
        
        elif args.interactive:
            chat_session.interactive_chat()
        
        else:
            print("Please specify --interactive for chat mode or --question for single query.")
            print("Use --help for more options.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())