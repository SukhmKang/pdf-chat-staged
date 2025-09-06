// PDF Chat Web Application
class PDFChatApp {
    constructor() {
        this.currentCollection = '';
        this.currentPDF = '';
        this.currentTOCSections = [];  // Changed from single string to array
        this.currentPageRange = null;  // For page range chat mode
        this.pdfDoc = null;
        this.currentPage = 1;
        this.scale = 1.2;
        this.canvas = null;
        this.context = null;
        this.pendingScrollPage = null;
        this.pendingCitedText = null;
        
        this.initializeElements();
        this.bindEvents();
        this.loadCollections();
        this.setupPDFCanvas();
        this.setTimeBasedGreeting();
        this.checkBirthday();
        
        // Load chat history for initially selected collection (if any)
        setTimeout(() => this.loadChatHistory(), 100);
        
        console.log('PDFChatApp initialized');
    }
    
    initializeElements() {
        // Sidebar elements
        this.collectionSelect = document.getElementById('collectionSelect');
        this.modelSelect = document.getElementById('modelSelect');
        this.selectedModel = document.getElementById('selectedModel');
        this.pdfList = document.getElementById('pdfList');
        this.tocList = document.getElementById('tocList');
        this.chatModeIndicator = document.getElementById('chatModeIndicator');
        this.backToCollectionBtn = document.getElementById('backToCollection');
        
        // Page range elements
        this.tocModeBtn = document.getElementById('tocModeBtn');
        this.pageRangeModeBtn = document.getElementById('pageRangeModeBtn');
        this.modeToggleContainer = document.getElementById('modeToggleContainer');
        this.noPdfMessage = document.getElementById('noPdfMessage');
        this.tocView = document.getElementById('tocView');
        this.pageRangeView = document.getElementById('pageRangeView');
        this.pageRangeInput = document.getElementById('pageRangeInput');
        this.applyPageRangeBtn = document.getElementById('applyPageRangeBtn');
        this.clearPageRangeBtn = document.getElementById('clearPageRangeBtn');
        this.currentPageRangeDiv = document.getElementById('currentPageRange');
        this.pageRangeDisplay = document.getElementById('pageRangeDisplay');
        this.modeTooltip = document.getElementById('modeTooltip');
        
        // Track current mode
        this.currentMode = 'pdf';  // 'pdf', 'toc', 'page_range'
        
        // Track verbosity setting
        this.currentVerbosity = 'medium';  // 'low', 'medium', 'high'
        
        // PDF viewer elements
        this.pdfTitle = document.getElementById('pdfTitle');
        this.pdfContainer = document.getElementById('pdfContainer');
        this.pdfControls = document.getElementById('pdfControls');
        this.pageInfo = document.getElementById('pageInfo');
        this.zoomInBtn = document.getElementById('zoomIn');
        this.zoomOutBtn = document.getElementById('zoomOut');
        this.fitWidthBtn = document.getElementById('fitWidth');
        this.scrollToTopBtn = document.getElementById('scrollToTop');
        this.zoomLevel = document.getElementById('zoomLevel');
        
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.clearChatBtn = document.getElementById('clearChat');
        this.verbositySelector = document.getElementById('verbositySelector');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Check for missing elements
        const requiredElements = {
            collectionSelect: this.collectionSelect,
            pdfList: this.pdfList,
            tocList: this.tocList,
            chatModeIndicator: this.chatModeIndicator,
            pdfTitle: this.pdfTitle,
            pdfContainer: this.pdfContainer,
            pdfControls: this.pdfControls,
            chatMessages: this.chatMessages,
            chatInput: this.chatInput,
            sendButton: this.sendButton,
            tocModeBtn: this.tocModeBtn,
            pageRangeModeBtn: this.pageRangeModeBtn,
            applyPageRangeBtn: this.applyPageRangeBtn
        };
        
        for (const [name, element] of Object.entries(requiredElements)) {
            if (!element) {
                console.error(`Required element missing: ${name}`);
            }
        }
    }
    
    bindEvents() {
        console.log('Starting bindEvents...');
        console.log('Elements check:', {
            collectionSelect: !!this.collectionSelect,
            chatInput: !!this.chatInput,
            sendButton: !!this.sendButton,
            clearChatBtn: !!this.clearChatBtn,
            zoomInBtn: !!this.zoomInBtn,
            zoomOutBtn: !!this.zoomOutBtn,
            fitWidthBtn: !!this.fitWidthBtn,
            scrollToTopBtn: !!this.scrollToTopBtn
        });
        
        // Collection selection
        if (this.collectionSelect) {
            this.collectionSelect.addEventListener('change', () => {
                this.currentCollection = this.collectionSelect.value;
                // Clear any selected PDF when collection changes
                this.clearPDFSelection();
                this.loadPDFs();
                this.loadChatHistory(); // Load chat history for new collection
                this.updateChatMode();
            });
        }

        // Model selection
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => {
                this.updateChatMode();
            });
        }

        // Back to collection button
        if (this.backToCollectionBtn) {
            this.backToCollectionBtn.addEventListener('click', () => {
                this.clearPDFSelection();
            });
        }
        
        // Chat mode toggle buttons
        if (this.tocModeBtn) {
            this.tocModeBtn.addEventListener('click', () => {
                this.switchToTOCMode();
            });
        }
        
        if (this.pageRangeModeBtn) {
            this.pageRangeModeBtn.addEventListener('click', () => {
                this.switchToPageRangeMode();
            });
        }
        
        // Page range controls
        if (this.applyPageRangeBtn) {
            this.applyPageRangeBtn.addEventListener('click', () => {
                this.applyPageRange();
            });
        }
        
        if (this.pageRangeInput) {
            this.pageRangeInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.applyPageRange();
                }
            });
        }
        
        if (this.clearPageRangeBtn) {
            this.clearPageRangeBtn.addEventListener('click', () => {
                this.clearPageRange();
            });
        }
        
        // Chat input
        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
        }
        
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => {
                // Add hearts animation if it's a birthday
                if (this.isBirthday || this.isAnniversary) {
                    this.createHeartsFromButton();
                }
                this.sendMessage();
            });
        }
        
        // Chat controls
        
        if (this.clearChatBtn) {
            this.clearChatBtn.addEventListener('click', () => {
                this.clearChat();
            });
        }
        
        // Verbosity selector
        if (this.verbositySelector) {
            this.verbositySelector.addEventListener('change', () => {
                this.currentVerbosity = this.verbositySelector.value;
                console.log('Verbosity changed to:', this.currentVerbosity);
            });
        }
        
        // PDF controls (with safety checks)
        if (this.zoomInBtn) {
            this.zoomInBtn.addEventListener('click', () => {
                this.scale = Math.min(3.0, this.scale * 1.2);
                this.handleZoomChange();
            });
        }
        
        if (this.zoomOutBtn) {
            this.zoomOutBtn.addEventListener('click', () => {
                this.scale = Math.max(0.5, this.scale / 1.2);
                this.handleZoomChange();
            });
        }
        
        if (this.fitWidthBtn) {
            this.fitWidthBtn.addEventListener('click', () => {
                this.fitToWidth();
            });
        }
        
        if (this.scrollToTopBtn) {
            this.scrollToTopBtn.addEventListener('click', () => {
                this.pdfContainer.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            });
        }
    }
    
    setTimeBasedGreeting() {
        const greetingElement = document.getElementById('timeBasedGreeting');
        if (!greetingElement) return;
        
        const now = new Date();
        const hour = now.getHours();
        let greeting;
        
        if (hour >= 5 && hour < 12) {
            greeting = "Good morning, Kris!";
        } else if (hour >= 12 && hour < 17) {
            greeting = "Good afternoon, Kris!";
        } else if (hour < 20) {
            greeting = "Good evening, Kris!";
        } else {
            greeting = "Don't work too late, please take care of yourself ok? :) "
        }
        
        greetingElement.textContent = greeting;
    }
    
    showInitialGreeting() {
        // Get the current time-based greeting
        const now = new Date();
        const hour = now.getHours();
        let greeting;
        
        if (hour >= 5 && hour < 12) {
            greeting = "Good morning, Kris!";
        } else if (hour >= 12 && hour < 17) {
            greeting = "Good afternoon, Kris!";
        } else if (hour < 20) {
            greeting = "Good evening, Kris!";
        } else {
            greeting = "Don't work too late, please take care of yourself ok? :) "
        }
        
        // Display the greeting in the chat area
        this.chatMessages.innerHTML = `
            <div class="chat-message bg-blue-50 p-3 rounded-lg mb-6">
                <div class="text-sm text-blue-800 font-medium">${greeting}</div>
                <div class="text-sm text-blue-600 mt-1">Select a collection and start asking questions about your PDFs.</div>
            </div>
        `;
    }
    
    async loadCollections() {
        try {
            console.log('Loading collections...');
            const response = await fetch('/api/collections');
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const collections = await response.json();
            console.log('Loaded collections:', collections);
            
            this.collectionSelect.innerHTML = '<option value="">Select a collection...</option>';
            collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection.name;
                option.textContent = `${collection.name} (${collection.pdf_count} PDFs)`;
                this.collectionSelect.appendChild(option);
                console.log('Added collection option:', collection.name);
            });
            
            console.log('Collections loaded successfully');
        } catch (error) {
            console.error('Error loading collections:', error);
            this.addMessage('Failed to load collections: ' + error.message, 'error');
        }
    }
    
    async loadPDFs() {
        if (!this.currentCollection) {
            this.pdfList.innerHTML = '<div class="text-gray-500 text-sm">Select a collection first</div>';
            return;
        }
        
        try {
            const response = await fetch(`/api/collections/${this.currentCollection}/pdfs`);
            const pdfs = await response.json();
            
            // Store PDFs data for citation handling
            this.currentPDFs = pdfs;
            
            this.pdfList.innerHTML = '';
            if (pdfs.length === 0) {
                this.pdfList.innerHTML = '<div class="text-gray-500 text-sm">No PDFs in this collection</div>';
                return;
            }
            
            pdfs.forEach(pdf => {
                const pdfItem = document.createElement('div');
                pdfItem.className = 'cursor-pointer p-2 rounded hover:bg-gray-100 border border-gray-200 max-w-full';
                pdfItem.innerHTML = `
                    <div class="font-medium text-sm truncate max-w-full" title="${pdf.title || pdf.filename}">${pdf.title || pdf.filename}</div>
                    <div class="text-xs text-gray-500">${this.formatFileSize(pdf.file_size)}</div>
                `;
                
                pdfItem.setAttribute('data-filename', pdf.filename);
                pdfItem.addEventListener('click', () => {
                    this.selectPDF(pdf);
                });
                
                this.pdfList.appendChild(pdfItem);
            });
        } catch (error) {
            console.error('Error loading PDFs:', error);
            this.showError('Failed to load PDFs');
        }
    }

    getLoadedPDFs() {
        return this.currentPDFs || [];
    }
    
    selectPDF(pdf) {
        this.currentPDF = pdf.filename;
        this.currentTOCSections = [];  // Clear selected sections
        this.currentPageRange = null;  // Clear page range
        this.currentMode = 'pdf';      // Reset to PDF mode
        
        // Update UI
        this.pdfTitle.textContent = pdf.title || pdf.filename;
        this.pdfControls.style.display = 'flex';
        
        // Enable mode selector now that PDF is selected
        this.enableModeSelector();
        
        // Show back to collection button
        if (this.backToCollectionBtn) {
            this.backToCollectionBtn.style.display = 'inline-block';
        }
        
        // Load PDF
        this.loadPDF(pdf.filename);
        
        // Load TOC sections
        this.loadTOCSections(pdf.toc_sections);
        
        // Reset to PDF mode (show TOC view by default)
        this.switchToPDFMode();
        
        // Update chat mode
        this.updateChatMode();
        
        // Highlight selected PDF
        document.querySelectorAll('#pdfList > div').forEach(item => {
            item.classList.remove('bg-blue-100', 'border-blue-300');
        });
        // Find the clicked element through the event delegation
        const clickedElement = document.querySelector(`[data-filename="${pdf.filename}"]`);
        if (clickedElement) {
            clickedElement.classList.add('bg-blue-100', 'border-blue-300');
        }
    }

    clearPDFSelection() {
        // Clear PDF-related state
        this.currentPDF = '';
        this.currentTOCSections = [];  // Clear selected sections array
        this.currentPageRange = null;  // Clear page range
        this.pendingScrollPage = null;
        this.pendingChunkHighlight = null;
        
        // Reset PDF viewer UI
        this.pdfTitle.textContent = 'No PDF Selected';
        this.pdfControls.style.display = 'none';
        
        // Hide back to collection button
        if (this.backToCollectionBtn) {
            this.backToCollectionBtn.style.display = 'none';
        }
        
        // Clear PDF container
        this.pdfContainer.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center text-gray-500">
                    <i class="fas fa-file-pdf text-6xl mb-4 opacity-50"></i>
                    <p>Select a PDF to view</p>
                </div>
            </div>
        `;
        
        // Clear TOC sections
        this.tocList.innerHTML = '<div class="text-gray-500 text-sm">Select a PDF first</div>';
        
        // Clear page range
        if (this.pageRangeInput) this.pageRangeInput.value = '';
        if (this.currentPageRangeDiv) this.currentPageRangeDiv.classList.add('hidden');
        
        // Disable mode selector
        this.disableModeSelector();
        
        // Reset mode
        this.currentMode = 'pdf';
        
        // Remove highlight from all PDFs
        document.querySelectorAll('#pdfList > div').forEach(item => {
            item.classList.remove('bg-blue-100', 'border-blue-300');
        });
        
        // Update chat mode
        this.updateChatMode();
    }
    
    loadTOCSections(sections) {
        this.tocList.innerHTML = '';
        
        if (!sections || sections.length === 0) {
            this.tocList.innerHTML = '<div class="text-gray-500 text-sm">No TOC sections available</div>';
            return;
        }
        
        // Store sections data for later use
        this.tocSections = sections;
        
        sections.forEach(section => {
            const tocItem = document.createElement('div');
            tocItem.className = 'cursor-pointer p-2 text-sm rounded hover:bg-gray-100 border border-gray-200';
            
            // Handle both old string format and new object format
            const title = typeof section === 'object' ? section.title : section;
            const page = typeof section === 'object' && section.page > 0 ? ` (p.${section.page})` : '';
            
            tocItem.textContent = `${title}${page}`;
            
            // Store the section data on the element for easy access
            tocItem.dataset.sectionTitle = title;
            tocItem.dataset.sectionPage = typeof section === 'object' ? section.page : '';
            
            tocItem.addEventListener('click', (event) => {
                // Support Ctrl/Cmd+click for multiple selections
                if (event.ctrlKey || event.metaKey) {
                    this.toggleTOCSection(title, tocItem);
                } else {
                    this.selectSingleTOCSection(title, tocItem);
                }
            });
            
            this.tocList.appendChild(tocItem);
        });
    }
    
    selectSingleTOCSection(section, tocItem) {
        // Check if this section is already selected (to allow deselection)
        if (this.currentTOCSections.includes(section)) {
            // Deselect - remove from array and clear highlighting
            this.currentTOCSections = this.currentTOCSections.filter(s => s !== section);
            tocItem.classList.remove('bg-green-100', 'border-green-300');
            
            // If no sections selected, return to PDF mode
            if (this.currentTOCSections.length === 0) {
                this.currentMode = 'pdf';
            }
        } else {
            // Select - clear other selections and select this one
            this.currentTOCSections = [section];
            this.currentMode = 'toc';
            
            // Update UI - clear all highlights and highlight only selected
            document.querySelectorAll('#tocList > div').forEach(item => {
                item.classList.remove('bg-green-100', 'border-green-300');
            });
            tocItem.classList.add('bg-green-100', 'border-green-300');
            
            // Scroll to the TOC section's start page
            const sectionPage = tocItem.dataset.sectionPage;
            if (sectionPage && sectionPage !== '' && this.pdfDoc) {
                const pageNumber = parseInt(sectionPage);
                if (pageNumber > 0) {
                    console.log(`Scrolling to TOC section "${section}" page ${pageNumber}`);
                    this.scrollToPage(pageNumber);
                }
            }
        }
        
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    toggleTOCSection(section, tocItem) {
        const currentIndex = this.currentTOCSections.indexOf(section);
        
        if (currentIndex === -1) {
            // Add section
            this.currentTOCSections.push(section);
            tocItem.classList.add('bg-green-100', 'border-green-300');
            this.currentMode = 'toc';
            
            // Scroll to the TOC section's start page if this is the first section being added
            if (this.currentTOCSections.length === 1) {
                const sectionPage = tocItem.dataset.sectionPage;
                if (sectionPage && sectionPage !== '' && this.pdfDoc) {
                    const pageNumber = parseInt(sectionPage);
                    if (pageNumber > 0) {
                        console.log(`Scrolling to TOC section "${section}" page ${pageNumber}`);
                        this.scrollToPage(pageNumber);
                    }
                }
            }
        } else {
            // Remove section
            this.currentTOCSections.splice(currentIndex, 1);
            tocItem.classList.remove('bg-green-100', 'border-green-300');
            
            // If no sections left, return to PDF mode
            if (this.currentTOCSections.length === 0) {
                this.currentMode = 'pdf';
            }
        }
        
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    async loadPDF(filename) {
        try {
            console.log(`Loading PDF: ${filename}`);
            
            const response = await fetch(`/api/pdfs/${filename}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            console.log(`PDF file size: ${arrayBuffer.byteLength} bytes`);
            
            // Load PDF with PDF.js
            const loadingTask = pdfjsLib.getDocument({
                data: arrayBuffer,
                cMapUrl: 'https://unpkg.com/pdfjs-dist@3.11.174/cmaps/',
                cMapPacked: true
            });
            
            this.pdfDoc = await loadingTask.promise;
            console.log(`PDF loaded successfully: ${this.pdfDoc.numPages} pages`);
            
            this.currentPage = 1;
            
            // Initialize continuous view instead of single page
            await this.initializeContinuousView();
            
            // Handle any pending citation scroll
            if (this.pendingScrollPage) {
                setTimeout(() => {
                    this.scrollToPage(this.pendingScrollPage);
                    if (this.pendingCitedText) {
                        this.highlightCitation(this.pendingCitedText, this.pendingScrollPage);
                    }
                    // Clear pending actions
                    this.pendingScrollPage = null;
                    this.pendingCitedText = null;
                }, 500); // Small delay to ensure PDF is fully rendered
            }
            
        } catch (error) {
            console.error('Error loading PDF:', error);
            this.pdfContainer.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-8 m-4 text-center">
                    <div class="text-red-800 text-lg font-semibold mb-2">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        Failed to load PDF: ${filename}
                    </div>
                    <div class="text-red-600 text-sm">${error.message}</div>
                </div>
            `;
        }
    }
    
    setupPDFCanvas() {
        console.log('Setting up PDF canvas container');
        
        // Create container for PDF pages
        this.pdfPagesContainer = document.createElement('div');
        this.pdfPagesContainer.className = 'pdf-pages-container';
        this.pdfPagesContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            padding: 20px;
            min-height: 100%;
            width: 100%;
        `;
        
        this.renderedPages = new Map(); // Cache for rendered pages
        this.pageHeights = new Map(); // Store page heights for scrolling
        this.visiblePages = new Set(); // Track which pages are visible
        
        console.log('PDF canvas container created');
    }
    
    async renderPDFPage() {
        if (!this.pdfDoc) return;
        
        // For continuous view, re-render all visible pages
        await this.rerenderAllPages();
        
        // Update page info
        this.updatePageInfo();
    }
    
    async rerenderAllPages() {
        // Re-render all currently rendered pages with new scale
        const pagesToRerender = Array.from(this.renderedPages.keys());
        
        for (const pageNum of pagesToRerender) {
            await this.renderSinglePage(pageNum, true); // force re-render
        }
    }
    
    async renderSimplePage() {
        try {
            console.log(`Rendering page ${this.currentPage} with scale ${this.scale}`);
            
            const page = await this.pdfDoc.getPage(this.currentPage);
            const viewport = page.getViewport({ scale: this.scale });
            
            // Create main container with enhanced styling
            const pageContainer = document.createElement('div');
            pageContainer.className = 'pdf-page';
            pageContainer.style.cssText = `
                position: relative;
                margin: 20px auto;
                display: inline-block;
                max-width: 95%;
            `;
            
            // Create canvas for PDF
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.style.cssText = 'display: block; max-width: 100%; height: auto;';
            
            // Render PDF to canvas with enhanced quality
            await page.render({
                canvasContext: context,
                viewport: viewport,
                intent: 'display'
            }).promise;
            
            // Add canvas to container
            pageContainer.appendChild(canvas);
            
            // Create enhanced text layer
            const textLayerDiv = document.createElement('div');
            textLayerDiv.className = 'textLayer';
            textLayerDiv.style.cssText = `
                position: absolute;
                left: 0;
                top: 0;
                right: 0;
                bottom: 0;
                overflow: hidden;
                opacity: 0.15;
                line-height: 1.0;
                pointer-events: auto;
                font-size: 1px;
            `;
            
            // Add text layer to container
            pageContainer.appendChild(textLayerDiv);
            
            // Render text layer with simplified approach
            try {
                const textContent = await page.getTextContent();
                
                // Use the basic renderTextLayer method that's more reliable
                pdfjsLib.renderTextLayer({
                    textContent: textContent,
                    container: textLayerDiv,
                    viewport: viewport,
                    textDivs: []
                });
                
                console.log(`Text layer rendered successfully for page ${this.currentPage}`);
            } catch (textError) {
                console.warn(`Text layer rendering failed for page ${this.currentPage}:`, textError);
                // Continue without text layer - PDF will still be viewable
            }
            
            // Clear container and add the page
            this.pdfContainer.innerHTML = '';
            this.pdfContainer.appendChild(pageContainer);
            
            // Update zoom level display
            this.updateZoomDisplay();
            
            console.log(`Page ${this.currentPage} rendered successfully`);
            
        } catch (error) {
            console.error('Error rendering page:', error);
            this.pdfContainer.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-8 m-4 text-center">
                    <div class="text-red-800 text-lg font-semibold mb-2">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        Failed to render page ${this.currentPage}
                    </div>
                    <div class="text-red-600 text-sm">${error.message}</div>
                </div>
            `;
        }
    }
    
    async initializeContinuousView() {
        try {
            console.log(`[DEBUG] initializeContinuousView called for ${this.pdfDoc.numPages} pages`);
            
            // Clear container
            this.pdfContainer.innerHTML = '';
            this.pdfContainer.className = 'overflow-auto h-full';
            
            // Show loading message with progress
            this.pdfContainer.innerHTML = `
                <div id="pdf-loading" class="flex items-center justify-center h-64">
                    <div class="text-center">
                        <div class="spinner mx-auto mb-4"></div>
                        <div class="text-gray-600">Loading ${this.pdfDoc.numPages} PDF pages...</div>
                        <div id="progress" class="text-sm text-gray-500 mt-2">Page 0 of ${this.pdfDoc.numPages}</div>
                    </div>
                </div>
            `;
            
            const progressDiv = document.getElementById('progress');
            
            // Create container for all pages
            const pagesContainer = document.createElement('div');
            pagesContainer.className = 'pdf-pages-container';
            pagesContainer.style.cssText = 'padding: 20px;';
            
            // Render all pages
            console.log(`[DEBUG] Rendering all ${this.pdfDoc.numPages} pages`);
            for (let i = 1; i <= this.pdfDoc.numPages; i++) {
                try {
                    if (progressDiv) {
                        progressDiv.textContent = `Page ${i} of ${this.pdfDoc.numPages}`;
                    }
                    
                    const pageElement = await this.renderPageElement(i);
                    pagesContainer.appendChild(pageElement);
                    
                    
                    if (i % 10 === 0) {
                        console.log(`Rendered ${i}/${this.pdfDoc.numPages} pages`);
                    }
                } catch (error) {
                    console.error(`Failed to render page ${i}:`, error);
                    
                    // Add error placeholder
                    const errorDiv = document.createElement('div');
                    errorDiv.id = `pdf-page-${i}`;
                    errorDiv.className = 'pdf-page-error';
                    errorDiv.innerHTML = `
                        <div class="bg-red-50 border border-red-200 rounded p-4 m-4 text-center">
                            <div class="text-red-800">Failed to load page ${i}</div>
                        </div>
                    `;
                    pagesContainer.appendChild(errorDiv);
                }
            }
            
            // Replace loading message with rendered pages
            this.pdfContainer.innerHTML = '';
            this.pdfContainer.appendChild(pagesContainer);
            
            // Set up scroll tracking for current page indicator
            this.currentPage = 1;
            this.updatePageInfo();
            
            // Add simple scroll listener to update page indicator
            this.pdfContainer.addEventListener('scroll', this.throttle(() => {
                this.updateCurrentPageFromScroll();
            }, 200));
            
            console.log(`Simple PDF viewer loaded successfully with ${this.pdfDoc.numPages} pages`);
            
        } catch (error) {
            console.error('Error initializing PDF view:', error);
            this.pdfContainer.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-8 m-4 text-center">
                    <div class="text-red-800 text-lg font-semibold mb-2">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        Failed to load PDF
                    </div>
                    <div class="text-red-600 text-sm">${error.message}</div>
                </div>
            `;
        }
    }
    
    async renderPageElement(pageNum) {
        console.log(`[DEBUG] renderPageElement called for page ${pageNum}`);
        const page = await this.pdfDoc.getPage(pageNum);
        console.log(`[DEBUG] Got page ${pageNum} from PDF`);
        const viewport = page.getViewport({ scale: this.scale });
        console.log(`[DEBUG] Created viewport for page ${pageNum}:`, viewport.width, 'x', viewport.height);
        
        // Create page container
        const pageContainer = document.createElement('div');
        pageContainer.id = `pdf-page-${pageNum}`;
        pageContainer.className = 'pdf-page';
        pageContainer.setAttribute('data-page', pageNum);
        pageContainer.style.cssText = `
            position: relative;
            border: 1px solid #ccc;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            background: white;
            margin: 20px auto;
            max-width: 100%;
        `;
        
        // Create canvas
        console.log(`[DEBUG] Creating canvas for page ${pageNum}...`);
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.cssText = 'display: block; max-width: 100%; height: auto;';
        console.log(`[DEBUG] Canvas created for page ${pageNum}:`, canvas.width, 'x', canvas.height);
        
        // Render PDF page
        console.log(`[DEBUG] Starting PDF render for page ${pageNum}...`);
        await page.render({
            canvasContext: context,
            viewport: viewport
        }).promise;
        console.log(`[DEBUG] PDF render completed for page ${pageNum}`);
        
        // Create text layer using PDF.js built-in API
        const textLayerDiv = document.createElement('div');
        textLayerDiv.className = 'textLayer';
        textLayerDiv.style.cssText = `
            position: absolute;
            left: 0;
            top: 0;
            width: ${canvas.width}px;
            height: ${canvas.height}px;
            overflow: hidden;
            opacity: 0.2;
            line-height: 1.0;
            transform-origin: top left;
            pointer-events: auto;
            transform: scale(1);
        `;
        
        // Get text content and render using PDF.js API with proper scaling
        const textContent = await page.getTextContent();
        
        // Clear any existing content
        textLayerDiv.innerHTML = '';
        
        // Create a new array to store text divs (required by PDF.js)
        const textDivs = [];
        
        const textLayerRenderTask = pdfjsLib.renderTextLayer({
            textContent: textContent,
            container: textLayerDiv,
            viewport: viewport,
            textDivs: textDivs,
            // Ensure proper scaling and positioning
            enhanceTextSelection: true
        });
        
        // Wait for text layer rendering to complete
        await textLayerRenderTask.promise;
        
        pageContainer.appendChild(canvas);
        pageContainer.appendChild(textLayerDiv);
        
        return pageContainer;
    }
    
    
    updateCurrentPageFromScroll() {
        // Simple method to find which page is most visible
        const pages = document.querySelectorAll('.pdf-page');
        const containerRect = this.pdfContainer.getBoundingClientRect();
        const containerCenter = containerRect.top + containerRect.height / 2;
        
        let closestPage = 1;
        let closestDistance = Infinity;
        
        pages.forEach((page, index) => {
            const pageRect = page.getBoundingClientRect();
            const pageCenter = pageRect.top + pageRect.height / 2;
            const distance = Math.abs(pageCenter - containerCenter);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPage = index + 1;
            }
        });
        
        if (closestPage !== this.currentPage) {
            this.currentPage = closestPage;
            this.updatePageInfo();
        }
    }
    
    async renderSinglePage(pageNum, forceRerender = false) {
        if (!forceRerender && this.renderedPages.has(pageNum)) {
            return this.renderedPages.get(pageNum);
        }
        if (this.renderedPages.has(pageNum)) {
            return this.renderedPages.get(pageNum);
        }
        
        try {
            console.log(`Rendering page ${pageNum}...`);
            
            const page = await this.pdfDoc.getPage(pageNum);
            const viewport = page.getViewport({ scale: this.scale });
            
            // Create page container
            const pageContainer = document.createElement('div');
            pageContainer.className = 'pdf-page-container';
            pageContainer.id = `pdf-page-${pageNum}`;
            pageContainer.style.cssText = `
                position: relative;
                border: 1px solid #ccc;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                background: white;
                margin: 20px auto;
                max-width: 100%;
                user-select: text;
                -webkit-user-select: text;
                -moz-user-select: text;
                -ms-user-select: text;
            `;
            
            // Create canvas for PDF rendering
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.style.cssText = 'display: block; max-width: 100%; height: auto;';
            
            // Render PDF page to canvas
            console.log(`[DEBUG] Starting canvas render for page ${pageNum}...`);
            try {
                const renderContext = {
                    canvasContext: context,
                    viewport: viewport
                };
                await page.render(renderContext).promise;
                console.log(`[DEBUG] Canvas rendered successfully for page ${pageNum}, now creating text layer...`);
            } catch (renderError) {
                console.error(`[DEBUG] Canvas render failed for page ${pageNum}:`, renderError);
                throw renderError;
            }
NY            
            // Create text layer for selectable text (optional, can be skipped if causing issues)
            try {
                console.log(`Creating text layer for page ${pageNum}...`);
                
                const textLayerDiv = document.createElement('div');
                textLayerDiv.className = 'textLayer';
                textLayerDiv.style.cssText = `
                    position: absolute;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    opacity: 0.3;
                    line-height: 1.0;
                    pointer-events: auto;
                    user-select: text;
                    -webkit-user-select: text;
                    -moz-user-select: text;
                    -ms-user-select: text;
                    z-index: 10;
                    background-color: rgba(255, 0, 0, 0.1);
                `;
                
                console.log(`Getting text content for page ${pageNum}...`);
                const textContent = await page.getTextContent();
                console.log(`Text content for page ${pageNum}:`, textContent.items.length, 'items');
                
                // Debug: show some text content
                if (textContent.items.length > 0) {
                    console.log(`First few text items:`, textContent.items.slice(0, 3));
                }
                
                console.log(`Rendering text layer for page ${pageNum}...`);
                
                // Try the modern PDF.js text layer API
                if (pdfjsLib.renderTextLayer) {
                    pdfjsLib.renderTextLayer({
                        textContent: textContent,
                        container: textLayerDiv,
                        viewport: viewport,
                        textDivs: []
                    });
                } else {
                    console.warn('renderTextLayer not available, trying alternative approach');
                    // Fallback: manually create text divs
                    textContent.items.forEach((textItem, index) => {
                        const textDiv = document.createElement('div');
                        textDiv.textContent = textItem.str;
                        textDiv.style.cssText = `
                            position: absolute;
                            color: rgba(0, 0, 0, 0.8);
                            font-size: 12px;
                            left: ${textItem.transform[4]}px;
                            top: ${textItem.transform[5]}px;
                        `;
                        textLayerDiv.appendChild(textDiv);
                    });
                }
                
                pageContainer.appendChild(textLayerDiv);
                console.log(`Text layer added to page container for page ${pageNum}`);
                
                // Debug: check if the text layer is actually in the DOM
                setTimeout(() => {
                    const addedLayer = pageContainer.querySelector('.textLayer');
                    console.log(`Text layer in DOM for page ${pageNum}:`, !!addedLayer, addedLayer?.children?.length || 0, 'children');
                }, 100);
                
            } catch (textError) {
                console.error(`Failed to render text layer for page ${pageNum}:`, textError);
                // Continue without text layer
            }
            
            // Add page number indicator
            const pageLabel = document.createElement('div');
            pageLabel.className = 'page-label';
            pageLabel.textContent = `Page ${pageNum}`;
            pageLabel.style.cssText = `
                position: absolute;
                top: -30px;
                left: 10px;
                font-size: 12px;
                color: #666;
                background: rgba(249, 249, 249, 0.9);
                padding: 4px 8px;
                border-radius: 4px;
                z-index: 10;
            `;
            
            // Assemble page
            pageContainer.appendChild(canvas);
            pageContainer.appendChild(pageLabel);
            
            // Replace placeholder or append
            const placeholder = document.getElementById(`pdf-placeholder-${pageNum}`);
            if (placeholder) {
                console.log(`Replacing placeholder for page ${pageNum}`);
                placeholder.replaceWith(pageContainer);
            } else {
                console.log(`Appending page ${pageNum} directly to container`);
                this.pdfPagesContainer.appendChild(pageContainer);
            }
            
            // Verify the page was added
            // Verify the page was added and check parent chain
            console.log(`Page ${pageNum} added:`, {
                width: pageContainer.offsetWidth,
                height: pageContainer.offsetHeight,
                visible: pageContainer.offsetParent !== null,
                parentContainer: pageContainer.parentElement?.className,
                inDocument: document.body.contains(pageContainer)
            });
            
            // Cache rendered page
            this.renderedPages.set(pageNum, pageContainer);
            this.pageHeights.set(pageNum, viewport.height + 60); // height + margins
            this.visiblePages.add(pageNum);
            
            console.log(`Successfully rendered page ${pageNum}`);
            return pageContainer;
            
        } catch (error) {
            console.error(`Error rendering page ${pageNum}:`, error);
            
            // Create error placeholder
            const errorContainer = document.createElement('div');
            errorContainer.id = `pdf-page-${pageNum}`;
            errorContainer.className = 'pdf-page-error';
            errorContainer.style.cssText = `
                border: 1px solid #fca5a5;
                background: #fef2f2;
                margin: 20px auto;
                padding: 40px;
                text-align: center;
                color: #dc2626;
                max-width: 100%;
            `;
            errorContainer.innerHTML = `
                <div class="text-lg font-semibold mb-2">Failed to load page ${pageNum}</div>
                <div class="text-sm">${error.message}</div>
            `;
            
            // Replace placeholder
            const placeholder = document.getElementById(`pdf-placeholder-${pageNum}`);
            if (placeholder) {
                placeholder.replaceWith(errorContainer);
            }
            
            return null;
        }
    }
    
    createPagePlaceholder(pageNum) {
        const placeholder = document.createElement('div');
        placeholder.id = `pdf-placeholder-${pageNum}`;
        placeholder.className = 'pdf-page-placeholder';
        placeholder.style.cssText = `
            height: 800px;
            border: 1px dashed #ccc;
            background: #f9f9f9;
            margin: 20px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 14px;
            max-width: 100%;
        `;
        placeholder.textContent = `Loading page ${pageNum}...`;
        
        this.pdfPagesContainer.appendChild(placeholder);
    }
    
    async handleScroll() {
        const container = this.pdfContainer;
        const scrollTop = container.scrollTop;
        const containerHeight = container.clientHeight;
        
        // Find which pages are in viewport
        let accumulatedHeight = 20; // initial padding
        let currentViewportPage = 1;
        
        for (let i = 1; i <= this.pdfDoc.numPages; i++) {
            const pageHeight = this.pageHeights.get(i) || 800; // estimated height
            
            // Check if page is in viewport
            const pageTop = accumulatedHeight;
            const pageBottom = accumulatedHeight + pageHeight;
            
            if (pageBottom > scrollTop && pageTop < scrollTop + containerHeight) {
                currentViewportPage = i;
                break;
            }
            
            accumulatedHeight += pageHeight + 20; // add gap between pages
        }
        
        // Update current page indicator
        if (currentViewportPage !== this.currentPage) {
            this.currentPage = currentViewportPage;
            this.updatePageInfo();
        }
        
        // Lazy load pages near viewport
        const range = 2; // Load 2 pages before/after viewport
        const loadPromises = [];
        
        for (let i = Math.max(1, currentViewportPage - range); 
             i <= Math.min(this.pdfDoc.numPages, currentViewportPage + range); 
             i++) {
            if (!this.renderedPages.has(i)) {
                loadPromises.push(this.renderSinglePage(i));
            }
        }
        
        // Load pages in parallel
        await Promise.all(loadPromises);
    }
    
    
    async handleZoomChange() {
        if (!this.pdfDoc) return;
        
        // Store current scroll position
        const scrollRatio = this.pdfContainer.scrollTop / this.pdfContainer.scrollHeight;
        
        // Re-render all pages with new scale
        await this.renderPDFPage();
        
        // Restore scroll position proportionally
        setTimeout(() => {
            this.pdfContainer.scrollTop = scrollRatio * this.pdfContainer.scrollHeight;
        }, 100);
    }
    
    async fitToWidth() {
        if (!this.pdfDoc) return;
        
        const page = await this.pdfDoc.getPage(1); // use first page for width calculation
        const viewport = page.getViewport({ scale: 1.0 });
        
        // Calculate scale to fit width
        const containerWidth = this.pdfContainer.clientWidth - 80; // account for padding and margins
        const optimalScale = containerWidth / viewport.width;
        
        this.scale = Math.min(optimalScale, 3.0); // cap at 3x zoom
        await this.handleZoomChange();
    }
    
    updateZoomDisplay() {
        if (this.zoomLevel) {
            this.zoomLevel.textContent = Math.round(this.scale * 100) + '%';
        }
    }
    
    updatePageInfo() {
        if (this.pageInfo && this.pdfDoc) {
            this.pageInfo.textContent = `Page ${this.currentPage} of ${this.pdfDoc.numPages}`;
        }
    }
    
    // Throttle function to limit scroll event frequency
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
    
    
    updateChatMode() {
        let mode = 'Collection Mode';
        let description = 'Chatting with entire collection';
        let bgColor = 'bg-blue-50';
        let textColor = 'text-blue-800';
        let descColor = 'text-blue-600';
        
        if (this.currentPageRange) {
            mode = 'Page Range Mode';
            description = `Chatting with pages ${this.currentPageRange} in ${this.currentPDF}`;
            bgColor = 'bg-orange-50';
            textColor = 'text-orange-800';
            descColor = 'text-orange-600';
        } else if (this.currentTOCSections.length > 0) {
            mode = this.currentTOCSections.length === 1 ? 'TOC Section Mode' : 'Multiple TOC Sections Mode';
            if (this.currentTOCSections.length === 1) {
                description = `Chatting with "${this.currentTOCSections[0]}" in ${this.currentPDF}`;
            } else {
                description = `Chatting with ${this.currentTOCSections.length} sections in ${this.currentPDF}`;
            }
            bgColor = 'bg-green-50';
            textColor = 'text-green-800';
            descColor = 'text-green-600';
        } else if (this.currentPDF) {
            mode = 'PDF Mode';
            description = `Chatting with ${this.currentPDF}`;
            bgColor = 'bg-purple-50';
            textColor = 'text-purple-800';
            descColor = 'text-purple-600';
        }
        
        const selectedModel = this.modelSelect ? this.modelSelect.value : 'claude';
        const modelDisplay = (selectedModel === 'gpt-5') ? 'GPT-5' : 
                            (selectedModel === 'gpt-4o') ? 'GPT-4 Omni' : 
                            (selectedModel === 'claude-3-5-sonnet-20241022') ? 'Claude 3.5 Sonnet (New)' : 
                            'Claude Sonnet';
        
        this.chatModeIndicator.className = `${bgColor} p-3 rounded-lg`;
        this.chatModeIndicator.innerHTML = `
            <div class="text-sm font-medium ${textColor}">${mode}</div>
            <div class="text-xs ${descColor}">${description}</div>
            <div class="text-xs text-blue-500 mt-1">Using ${modelDisplay}</div>
        `;
        
        // Update mode buttons to reflect current state
        if (this.currentPDF) {
            this.updateModeButtons();
            this.updateModeTooltip();
        }
    }
        
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || !this.currentCollection) return;
        
        // Clear input and show typing indicator
        this.chatInput.value = '';
        this.showTypingIndicator();
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    collection: this.currentCollection,
                    pdf_filename: this.currentPDF || null,
                    toc_sections: this.currentTOCSections.length > 0 ? this.currentTOCSections : null,
                    toc_section: this.currentTOCSections.length === 1 ? this.currentTOCSections[0] : null, // For backward compatibility
                    page_range: this.currentPageRange,
                    model: this.modelSelect ? this.modelSelect.value : 'claude',
                    verbosity: this.currentVerbosity,
                    enhanced: false
                })
            });
            
            const result = await response.json();
            
            // Hide typing indicator and add the assistant's response
            this.hideTypingIndicator();
            this.addMessage(result.response, 'assistant', result.sources);
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, there was an error processing your message.', 'error');
        }
    }
    
    addMessage(content, type, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message mb-4';
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="flex justify-end">
                    <div class="bg-blue-600 text-white p-3 rounded-lg max-w-xs">
                        ${this.escapeHtml(content)}
                    </div>
                </div>
            `;
        } else if (type === 'assistant') {
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="mt-2 text-xs text-gray-600">
                        <details class="cursor-pointer">
                            <summary>Sources (${sources.length})</summary>
                            <div class="mt-1 space-y-1">
                                ${sources.map(source => `
                                    <div class="bg-gray-50 p-2 rounded">
                                        <div class="font-medium">${source.pdf_filename || 'Unknown'}</div>
                                        <div class="text-xs">${source.content?.substring(0, 100)}...</div>
                                    </div>
                                `).join('')}
                            </div>
                        </details>
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="flex justify-start">
                    <div class="bg-gray-100 p-3 rounded-lg max-w-lg">
                        <div class="prose prose-sm">${this.formatMarkdown(content)}</div>
                        ${sourcesHtml}
                    </div>
                </div>
            `;
        } else if (type === 'error') {
            messageDiv.innerHTML = `
                <div class="flex justify-center">
                    <div class="bg-red-100 text-red-800 p-3 rounded-lg">
                        <i class="fas fa-exclamation-triangle mr-2"></i>${this.escapeHtml(content)}
                    </div>
                </div>
            `;
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        this.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.classList.add('hidden');
    }
    
    async clearChat() {
        if (!this.currentCollection) {
            console.error('No collection selected');
            return;
        }
        
        try {
            const response = await fetch(`/api/clear-chat/${this.currentCollection}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                // Clear the visual interface
                this.chatMessages.innerHTML = `
                    <div class="chat-message bg-blue-50 p-3 rounded-lg mb-6">
                        <div class="text-sm text-blue-800 font-medium">Chat cleared!</div>
                        <div class="text-sm text-blue-600 mt-1">Start a new conversation.</div>
                    </div>
                `;
                console.log('Chat history cleared successfully');
            } else {
                console.error('Failed to clear chat history');
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }
    
    async loadChatHistory() {
        if (!this.currentCollection) {
            // No collection selected, show greeting
            this.showInitialGreeting();
            return;
        }
        
        try {
            const response = await fetch(`/api/chat-history/${this.currentCollection}`);
            const data = await response.json();
            
            console.log('Chat history loaded:', data.messages?.length, 'messages');
            
            if (data.messages && data.messages.length > 0) {
                // Clear current messages
                this.chatMessages.innerHTML = '';
                
                // Add each message from history
                for (const message of data.messages) {
                    if (message.type === 'user') {
                        this.addMessage(message.content, 'user');
                    } else if (message.type === 'assistant') {
                        this.addMessage(message.content, 'assistant', message.sources);
                    }
                }
            } else {
                // No history, show welcome message
                this.chatMessages.innerHTML = `
                    <div class="text-gray-500 text-center py-8">
                        No chat history. Start a conversation!
                    </div>
                `;
            }
            
            // Scroll to bottom
            this.scrollToBottom();
            
        } catch (error) {
            console.error('Error loading chat history:', error);
            this.chatMessages.innerHTML = `
                <div class="text-red-500 text-center py-8">
                    Error loading chat history
                </div>
            `;
        }
    }
    
    scrollToBottom() {
        const chatContainer = this.chatMessages.parentElement;
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatMarkdown(text) {
        // Parse citations first and store them
        const parsedCitations = [];
        const doubleQuoteRegex = /<citation\s+pdf_name="([^"]+)"\s+page_number="([^"]+)"\s+chunk_id="([^"]+)"\s+cited_text="(.*?)">/g;
        const singleQuoteRegex = /<citation\s+pdf_name='([^']+)'\s+page_number='([^']+)'\s+chunk_id='([^']+)'\s+cited_text='(.*?)'>/g;
        
        // Replace citations with unique placeholders and parse them immediately
        let textWithPlaceholders = text;
        
        // Handle double quotes
        textWithPlaceholders = textWithPlaceholders.replace(doubleQuoteRegex, (match) => {
            const parsedCitation = this.parseCitations(match);
            const placeholder = `CITATIONPLACEHOLDER${parsedCitations.length}CITATIONPLACEHOLDER`;
            parsedCitations.push(parsedCitation);
            return placeholder;
        });
        
        // Handle single quotes
        textWithPlaceholders = textWithPlaceholders.replace(singleQuoteRegex, (match) => {
            const parsedCitation = this.parseCitations(match);
            const placeholder = `CITATIONPLACEHOLDER${parsedCitations.length}CITATIONPLACEHOLDER`;
            parsedCitations.push(parsedCitation);
            return placeholder;
        });
        
        // Apply full markdown parsing with marked.js
        let formattedText;
        try {
            // Configure marked for better security and formatting
            marked.setOptions({
                breaks: true,        // Convert \n to <br>
                gfm: true,          // GitHub Flavored Markdown
                sanitize: false      // We'll handle our own sanitization
            });
            
            formattedText = marked.parse(textWithPlaceholders);
        } catch (error) {
            console.warn('Markdown parsing failed, falling back to basic formatting:', error);
            // Fallback to basic formatting
            formattedText = textWithPlaceholders
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }
        
        // Restore parsed citations by replacing placeholders
        parsedCitations.forEach((parsedCitation, index) => {
            const placeholder = `CITATIONPLACEHOLDER${index}CITATIONPLACEHOLDER`;
            formattedText = formattedText.replace(new RegExp(placeholder, 'g'), parsedCitation);
        });
        
        return `<div class="markdown-content">${formattedText}</div>`;
    }
    
    parseCitations(text) {
        // Parse citation tags and convert to clickable elements
        // Handle both single and double quotes, but consistently within each tag
        const doubleQuoteRegex = /<citation\s+pdf_name="([^"]+)"\s+page_number="([^"]+)"\s+chunk_id="([^"]+)"\s+cited_text="(.*?)">/g;
        const singleQuoteRegex = /<citation\s+pdf_name='([^']+)'\s+page_number='([^']+)'\s+chunk_id='([^']+)'\s+cited_text='(.*?)'>/g;
        
        // Try double quotes first, then single quotes
        let result = text.replace(doubleQuoteRegex, (_, pdfName, pageNumber, chunkId, citedText) => {
            return this.createCitationSpan(pdfName, pageNumber, chunkId, citedText);
        });
        
        result = result.replace(singleQuoteRegex, (_, pdfName, pageNumber, chunkId, citedText) => {
            return this.createCitationSpan(pdfName, pageNumber, chunkId, citedText);
        });
        
        return result;
    }
    
    createCitationSpan(pdfName, pageNumber, chunkId, citedText) {
        // Extract just the PDF filename without extension for display
        let displayName = pdfName.replace(/\.pdf$/i, '');
        
        // Shorten long PDF names for more compact citation tags
        if (displayName.length > 25) {
            displayName = displayName.substring(0, 22) + '...';
        }
        
        // Create tooltip with cited text if available
        let titleAttribute = '';
        if (citedText && citedText.trim()) {
            let tooltipText = citedText.trim();
            // Limit tooltip length and add ellipsis
            if (tooltipText.length > 150) {
                tooltipText = tooltipText.substring(0, 147) + '...';
            } else {
                tooltipText = tooltipText + '...';
            }
            // Escape quotes for HTML attribute
            const escapedTooltipText = tooltipText.replace(/'/g, '&#39;').replace(/"/g, '&quot;');
            titleAttribute = `title="${escapedTooltipText}"`;
        }
        
        return `<span class="citation-tag" onclick="app.handleCitationClick('${pdfName}', ${pageNumber}, '${chunkId}', '${citedText.replace(/'/g, "\\'")})')" ${titleAttribute}><i class="fas fa-link"></i> ${displayName} p.${pageNumber}</span>`;
    }
    
    
    handleCitationClick(pdfName, pageNumber, chunkId, citedText = '') {
        console.log(`Citation clicked: ${pdfName}, page ${pageNumber}, chunk ${chunkId}`, citedText ? `cited text: ${citedText}` : '');
        
        // Add hearts animation if it's an anniversary
        if (this.isAnniversary) {
            // Get the citation element that was clicked
            const citationElements = document.querySelectorAll('.citation-tag');
            let clickedCitation = null;
            
            // Find which citation was clicked (this is a bit hacky but works)
            citationElements.forEach(el => {
                if (el.onclick && el.onclick.toString().includes(chunkId)) {
                    clickedCitation = el;
                }
            });
            
            if (clickedCitation) {
                const rect = clickedCitation.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                
                // Create 3 hearts for citation clicks (fewer than send button)
                for (let i = 0; i < 3; i++) {
                    setTimeout(() => {
                        this.createSingleHeart(centerX, centerY);
                    }, i * 80);
                }
            }
        }
        
        // Store the target page and cited text for after PDF loads
        this.pendingScrollPage = parseInt(pageNumber);
        this.pendingCitedText = citedText;
        
        // First, select the PDF if it's not already selected
        if (this.currentPDF !== pdfName) {
            // Find the PDF in the current collection and select it
            const pdfElement = document.querySelector(`[data-filename="${pdfName}"]`);
            if (pdfElement) {
                // Find the PDF data from the collection
                const currentPdfs = this.getLoadedPDFs();
                const pdfData = currentPdfs.find(pdf => pdf.filename === pdfName);
                if (pdfData) {
                    // Select the PDF and it will handle scrolling after load
                    this.selectPDF(pdfData);
                } else {
                    pdfElement.click(); // Fallback to click
                }
            } else {
                // If PDF not found in current view, show message
                console.warn(`PDF ${pdfName} not found in current collection`);
                this.showError(`PDF ${pdfName} not found in current collection`);
                return;
            }
        } else {
            // PDF already selected, scroll immediately
            this.scrollToPage(parseInt(pageNumber));
            this.highlightCitation(citedText, parseInt(pageNumber));
        }
    }
    
    async scrollToPage(pageNumber) {
        if (!this.pdfDoc) {
            console.warn('No PDF document loaded');
            return;
        }
        
        console.log(`Scrolling to page ${pageNumber}...`);
        
        // With the new simple viewer, all pages are already rendered
        // Just find the page element and scroll to it
        const pageElement = document.getElementById(`pdf-page-${pageNumber}`);
        
        if (pageElement) {
            // Scroll to the page
            pageElement.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
            
            // Highlight the page immediately (no need to wait for rendering)
            setTimeout(() => {
                pageElement.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5)';
                setTimeout(() => {
                    pageElement.style.boxShadow = '';
                }, 3000);
            }, 500);
            
            console.log(`Successfully scrolled to and highlighted page ${pageNumber}`);
        } else {
            console.warn(`Page ${pageNumber} element not found`);
            this.showError(`Could not find page ${pageNumber}`);
        }
    }
    
    attemptHighlight(pageNumber, attempts) {
        if (attempts >= 5) {
            console.log(`Could not highlight page ${pageNumber} after ${attempts} attempts`);
            return;
        }
        
        // Look for the rendered page
        const pageElement = document.getElementById(`pdf-page-${pageNumber}`);
        
        if (pageElement) {
            // Found it! Highlight the page
            pageElement.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5)';
            setTimeout(() => {
                pageElement.style.boxShadow = '';
            }, 3000);
            console.log(`Successfully highlighted page ${pageNumber}`);
        } else {
            // Not rendered yet, try again
            console.log(`Page ${pageNumber} not yet rendered, attempt ${attempts + 1}/5`);
            setTimeout(() => {
                this.attemptHighlight(pageNumber, attempts + 1);
            }, 1000);
        }
    }
    
    highlightPageAfterScroll(pageNumber) {
        // Try to find the rendered page element
        const pageElement = document.getElementById(`pdf-page-${pageNumber}`) ||
                           this.findPageElementByNumber(pageNumber);
        
        if (pageElement) {
            // Add temporary highlight effect
            pageElement.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5)';
            setTimeout(() => {
                pageElement.style.boxShadow = '';
            }, 3000);
            console.log(`Highlighted page ${pageNumber}`);
        } else {
            // Page might still be a placeholder, try again after a bit more time
            console.log(`Page ${pageNumber} not yet rendered, trying again...`);
            setTimeout(() => {
                this.highlightPageAfterScroll(pageNumber);
            }, 1000);
        }
    }
    
    async ensurePageRendered(pageNumber) {
        // Check if page is already rendered
        if (this.renderedPages && this.renderedPages.has(pageNumber)) {
            console.log(`Page ${pageNumber} already rendered`);
            return; // Already rendered
        }
        
        // Check if there's a placeholder that needs to be replaced
        const placeholder = document.getElementById(`pdf-placeholder-${pageNumber}`);
        if (placeholder) {
            console.log(`Found placeholder for page ${pageNumber}, rendering actual page...`);
            
            // Render the page using the same method as lazy loading
            await this.renderSinglePage(pageNumber);
            
            // Trigger a scroll event to update the lazy loading system
            // This ensures the lazy loading knows about the newly rendered page
            setTimeout(() => {
                if (this.handleScroll) {
                    this.handleScroll();
                }
            }, 100);
            
            console.log(`Successfully rendered page ${pageNumber}`);
        } else {
            console.log(`No placeholder found for page ${pageNumber}, it may already be rendered or not exist`);
        }
    }
    
    findPageElementByNumber(pageNumber) {
        // Try to find page by position in continuous view
        const allPages = document.querySelectorAll('.pdf-page');
        return allPages[pageNumber - 1] || null; // pages are 1-indexed
    }
    
    
    async highlightCitation(citedText, pageNumber = null) {
        if (!citedText || !citedText.trim()) {
            console.warn('Cannot highlight citation: missing cited text');
            return;
        }
        
        console.log(`Highlighting cited text: ${citedText.substring(0, 100)}...`);
        
        try {
            // Clear any existing highlights first
            this.clearChunkHighlights();
            
            // Use the cited text directly for highlighting
            const searchText = citedText.trim();
            
            if (searchText.length < 5) {
                console.warn('Cited text too short for reliable highlighting');
                return;
            }
            
            console.log('Searching for cited text:', searchText.substring(0, 100) + '...');
            
            // Find and highlight the text
            let highlighted = false;
            
            // If we have a page number from the citation, focus on that page and adjacent pages
            let pageRange = [];
            if (pageNumber && pageNumber > 0) {
                pageRange = [pageNumber - 1, pageNumber, pageNumber + 1].filter(p => p > 0);
            } else {
                // If no page number, search current page and adjacent pages
                pageRange = [this.currentPage - 1, this.currentPage, this.currentPage + 1].filter(p => p > 0);
            }
            
            // Try to find the text on the specified pages
            for (const pageNum of pageRange) {
                if (this.highlightTextInPage(searchText, pageNum)) {
                    highlighted = true;
                    console.log(`Found cited text match on page ${pageNum}`);
                    break;
                }
            }
            
            if (highlighted) {
                console.log('Successfully highlighted chunk text');
                
                // Set timeout to fade the highlight after 5 seconds
                setTimeout(() => {
                    this.fadeChunkHighlights();
                }, 5000);
                
                // Set timeout to clear highlights after 10 seconds  
                setTimeout(() => {
                    this.clearChunkHighlights();
                }, 10000);
            } else {
                console.warn('Could not find chunk text in PDF for highlighting');
            }
            
        } catch (error) {
            console.error('Error highlighting citation:', error);
        }
    }
    
    highlightTextInPage(searchText, targetPageNumber) {
        console.log(`Looking for text in page ${targetPageNumber}: "${searchText.substring(0, 100)}..."`);
        
        // Try multiple matching strategies with increasing aggressiveness
        const strategies = [
            () => this.exactTextMatch(searchText, targetPageNumber),
            () => this.normalizedTextMatch(searchText, targetPageNumber),
            () => this.wordsOnlyMatch(searchText, targetPageNumber),
            () => this.partialTextMatch(searchText, targetPageNumber),
            () => this.keywordMatch(searchText, targetPageNumber)
        ];
        
        for (const strategy of strategies) {
            if (strategy()) {
                return true;
            }
        }
        
        console.log('All matching strategies failed');
        return false;
    }
    
    exactTextMatch(searchText, targetPageNumber) {
        console.log('Trying exact text match...');
        return this.findAndHighlightText(searchText, targetPageNumber, false);
    }
    
    normalizedTextMatch(searchText, targetPageNumber) {
        console.log('Trying normalized text match...');
        
        // Aggressive text normalization
        const normalizeText = (text) => {
            return text
                .replace(/\s+/g, ' ')                    // Normalize whitespace
                .replace(/[""'']/g, '"')                 // Normalize quotes
                .replace(/[–—]/g, '-')                   // Normalize dashes
                .replace(/[^\w\s\-".,;:!?()\[\]]/g, '')  // Remove special chars
                .toLowerCase()
                .trim();
        };
        
        const normalizedSearch = normalizeText(searchText);
        return this.findAndHighlightText(normalizedSearch, targetPageNumber, true, normalizeText);
    }
    
    wordsOnlyMatch(searchText, targetPageNumber) {
        console.log('Trying words-only match...');
        
        // Extract just the words, ignoring punctuation and spacing
        const extractWords = (text) => {
            return text.match(/\b\w+\b/g)?.join(' ').toLowerCase() || '';
        };
        
        const wordsOnlySearch = extractWords(searchText);
        if (wordsOnlySearch.length < 10) return false; // Too short
        
        return this.findAndHighlightText(wordsOnlySearch, targetPageNumber, true, extractWords);
    }
    
    partialTextMatch(searchText, targetPageNumber) {
        console.log('Trying partial text match with smaller chunks...');
        
        // Try with smaller chunks of the search text
        const words = searchText.split(/\s+/);
        const chunkSizes = [15, 10, 7, 5]; // Try different chunk sizes
        
        for (const chunkSize of chunkSizes) {
            if (words.length < chunkSize) continue;
            
            const chunk = words.slice(0, chunkSize).join(' ');
            console.log(`Trying ${chunkSize}-word chunk: "${chunk.substring(0, 50)}..."`);
            
            if (this.normalizedTextMatch(chunk, targetPageNumber)) {
                return true;
            }
        }
        
        return false;
    }
    
    keywordMatch(searchText, targetPageNumber) {
        console.log('Trying keyword-based match...');
        
        // Extract key phrases (3+ character words)
        const keywords = searchText
            .match(/\b\w{3,}\b/g)
            ?.slice(0, 5) // Take first 5 keywords
            ?.join(' ');
            
        if (!keywords || keywords.length < 10) return false;
        
        console.log(`Searching for keywords: "${keywords}"`);
        return this.normalizedTextMatch(keywords, targetPageNumber);
    }
    
    findAndHighlightText(searchText, targetPageNumber, caseInsensitive = true, textProcessor = null) {
        const textLayers = document.querySelectorAll('.textLayer');
        let found = false;
        
        textLayers.forEach((textLayer, pageIndex) => {
            const actualPageNumber = pageIndex + 1;
            
            // Skip if we have a target page and this isn't it, but also try adjacent pages
            if (targetPageNumber && Math.abs(actualPageNumber - targetPageNumber) > 1) {
                return;
            }
            
            const textElements = textLayer.querySelectorAll('span, div');
            if (textElements.length === 0) return;
            
            // Build text map
            const textMap = [];
            let fullPageText = '';
            
            textElements.forEach(element => {
                const elementText = element.textContent.trim();
                if (elementText) {
                    textMap.push({
                        element: element,
                        text: elementText,
                        startIndex: fullPageText.length
                    });
                    fullPageText += elementText + ' ';
                }
            });
            
            if (fullPageText.length === 0) return;
            
            // Apply text processor if provided
            const processedPageText = textProcessor ? textProcessor(fullPageText) : fullPageText;
            const processedSearchText = textProcessor ? textProcessor(searchText) : searchText;
            
            console.log(`Page ${actualPageNumber}: Searching in ${processedPageText.length} chars`);
            console.log(`Search text: "${processedSearchText.substring(0, 100)}..."`);
            console.log(`Page text sample: "${processedPageText.substring(0, 200)}..."`);
            
            // Try to find the text
            let searchIndex = -1;
            if (caseInsensitive) {
                searchIndex = processedPageText.toLowerCase().indexOf(processedSearchText.toLowerCase());
            } else {
                searchIndex = processedPageText.indexOf(processedSearchText);
            }
            
            if (searchIndex !== -1) {
                console.log(`✓ Found text at index ${searchIndex} on page ${actualPageNumber}`);
                
                // Map back to original text if we used a processor
                let actualStart = searchIndex;
                let actualEnd = searchIndex + processedSearchText.length;
                
                if (textProcessor) {
                    // For processed text, highlight more generously around the found area
                    actualStart = Math.max(0, searchIndex - 50);
                    actualEnd = Math.min(fullPageText.length, searchIndex + processedSearchText.length + 50);
                }
                
                // Highlight elements that overlap with found text
                let highlightedCount = 0;
                textMap.forEach(({ element, text, startIndex }) => {
                    const elementEndIndex = startIndex + text.length;
                    
                    if (startIndex <= actualEnd && elementEndIndex >= actualStart) {
                        element.classList.add('chunk-highlight');
                        highlightedCount++;
                    }
                });
                
                console.log(`Highlighted ${highlightedCount} elements`);
                
                if (highlightedCount > 0) {
                    found = true;
                    
                    // Scroll to first highlighted element
                    const firstHighlight = textLayer.querySelector('.chunk-highlight');
                    if (firstHighlight) {
                        firstHighlight.scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'center',
                            inline: 'nearest'
                        });
                    }
                }
            } else {
                console.log(`✗ Page ${actualPageNumber}: Text not found`);
            }
        });
        
        return found;
    }
    
    clearChunkHighlights() {
        document.querySelectorAll('.chunk-highlight, .chunk-highlight-fade').forEach(element => {
            element.classList.remove('chunk-highlight', 'chunk-highlight-fade');
        });
    }
    
    fadeChunkHighlights() {
        document.querySelectorAll('.chunk-highlight').forEach(element => {
            element.classList.remove('chunk-highlight');
            element.classList.add('chunk-highlight-fade');
        });
    }
    
    showTemporaryMessage(message) {
        // Create a temporary toast message
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-yellow-500 text-white px-4 py-2 rounded shadow-lg z-50';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showError(message) {
        this.addMessage(message, 'error');
    }
    
    // Mode Management Methods
    enableModeSelector() {
        if (this.modeToggleContainer) {
            this.modeToggleContainer.classList.remove('opacity-50', 'pointer-events-none');
            this.modeToggleContainer.classList.add('opacity-100', 'pointer-events-auto');
        }
        
        if (this.tocModeBtn) {
            this.tocModeBtn.classList.remove('cursor-not-allowed');
            this.tocModeBtn.classList.add('cursor-pointer');
        }
        
        if (this.pageRangeModeBtn) {
            this.pageRangeModeBtn.classList.remove('cursor-not-allowed');
            this.pageRangeModeBtn.classList.add('cursor-pointer');
        }
        
        if (this.noPdfMessage) {
            this.noPdfMessage.classList.add('hidden');
        }
        
        // Update tooltip for enabled state
        this.updateModeTooltip();
    }
    
    disableModeSelector() {
        if (this.modeToggleContainer) {
            this.modeToggleContainer.classList.add('opacity-50', 'pointer-events-none');
            this.modeToggleContainer.classList.remove('opacity-100', 'pointer-events-auto');
        }
        
        if (this.tocModeBtn) {
            this.tocModeBtn.classList.add('cursor-not-allowed');
            this.tocModeBtn.classList.remove('cursor-pointer');
        }
        
        if (this.pageRangeModeBtn) {
            this.pageRangeModeBtn.classList.add('cursor-not-allowed');
            this.pageRangeModeBtn.classList.remove('cursor-pointer');
        }
        
        if (this.noPdfMessage) {
            this.noPdfMessage.classList.remove('hidden');
        }
        
        // Update tooltip for disabled state
        this.updateModeTooltip();
    }
    
    switchToPDFMode() {
        // Default mode - show TOC view but don't select anything
        this.currentMode = 'pdf';
        
        // Clear all selections
        this.currentTOCSections = [];
        this.currentPageRange = null;
        
        // Clear UI selections
        if (this.tocList) {
            const tocItems = this.tocList.querySelectorAll('div');
            tocItems.forEach(item => {
                item.classList.remove('bg-green-100', 'border-green-300');
            });
        }
        
        if (this.pageRangeInput) this.pageRangeInput.value = '';
        if (this.currentPageRangeDiv) this.currentPageRangeDiv.classList.add('hidden');
        
        // Update button states
        this.updateModeButtons();
        
        // Show TOC view by default (but nothing selected)
        if (this.tocView) this.tocView.classList.remove('hidden');
        if (this.pageRangeView) this.pageRangeView.classList.add('hidden');
    }
    
    switchToTOCMode() {
        // Only switch UI, actual TOC selections are handled by TOC click handlers
        if (this.tocView) this.tocView.classList.remove('hidden');
        if (this.pageRangeView) this.pageRangeView.classList.add('hidden');
        
        // Clear page range
        this.currentPageRange = null;
        if (this.pageRangeInput) this.pageRangeInput.value = '';
        if (this.currentPageRangeDiv) this.currentPageRangeDiv.classList.add('hidden');
        
        this.updateModeButtons();
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    switchToPageRangeMode() {
        // Clear TOC selections and switch to page range mode
        this.currentTOCSections = [];
        
        // Clear TOC UI selections
        if (this.tocList) {
            const tocItems = this.tocList.querySelectorAll('div');
            tocItems.forEach(item => {
                item.classList.remove('bg-green-100', 'border-green-300');
            });
        }
        
        // Show page range view
        if (this.tocView) this.tocView.classList.add('hidden');
        if (this.pageRangeView) this.pageRangeView.classList.remove('hidden');
        
        this.updateModeButtons();
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    updateModeButtons() {
        // Reset all button states
        if (this.tocModeBtn) {
            this.tocModeBtn.classList.remove('bg-blue-500', 'bg-orange-500', 'bg-purple-500', 'text-white');
            this.tocModeBtn.classList.add('text-gray-600');
        }
        
        if (this.pageRangeModeBtn) {
            this.pageRangeModeBtn.classList.remove('bg-blue-500', 'bg-orange-500', 'bg-purple-500', 'text-white');
            this.pageRangeModeBtn.classList.add('text-gray-600');
        }
        
        // Highlight active mode based on which view is currently visible
        if (this.pageRangeView && !this.pageRangeView.classList.contains('hidden')) {
            // Page range view is active
            if (this.pageRangeModeBtn) {
                this.pageRangeModeBtn.classList.add('bg-orange-500', 'text-white');
                this.pageRangeModeBtn.classList.remove('text-gray-600');
            }
        } else if (this.tocView && !this.tocView.classList.contains('hidden')) {
            // TOC view is active (default)
            if (this.tocModeBtn) {
                this.tocModeBtn.classList.add('bg-blue-500', 'text-white');
                this.tocModeBtn.classList.remove('text-gray-600');
            }
        }
    }
    
    updateModeTooltip() {
        if (!this.modeTooltip) return;
        
        if (!this.currentPDF) {
            // No PDF selected
            this.modeTooltip.textContent = "Select a PDF to choose chat mode";
        } else if (this.currentMode === 'page_range' || (this.pageRangeView && !this.pageRangeView.classList.contains('hidden'))) {
            // Page range mode active
            this.modeTooltip.textContent = "Enter page range (e.g., 1-10) to chat with specific pages";
        } else if (this.currentMode === 'toc' && this.currentTOCSections.length > 0) {
            // TOC sections selected
            if (this.currentTOCSections.length === 1) {
                this.modeTooltip.textContent = `Click "${this.currentTOCSections[0]}" again to deselect and return to PDF mode`;
            } else {
                this.modeTooltip.textContent = `Ctrl/Cmd+click sections to deselect. Clear all to return to PDF mode`;
            }
        } else if (this.currentMode === 'pdf' || this.currentTOCSections.length === 0) {
            // PDF mode or TOC view but nothing selected
            if (!this.tocView.classList.contains('hidden')) {
                this.modeTooltip.textContent = "Click: single section | Ctrl/Cmd+click: multiple sections";
            } else {
                this.modeTooltip.textContent = "Switch between TOC sections and page ranges";
            }
        } else {
            // Fallback
            this.modeTooltip.textContent = "Switch between TOC sections and page ranges";
        }
    }
    
    applyPageRange() {
        const pageRange = this.pageRangeInput.value.trim();
        if (!pageRange) {
            this.showError('Please enter a page range (e.g., "1-10")');
            return;
        }
        
        // Validate page range format
        const rangePattern = /^\d+-\d+$/;
        if (!rangePattern.test(pageRange)) {
            this.showError('Invalid format. Use "start-end" like "1-10"');
            return;
        }
        
        const [startPage, endPage] = pageRange.split('-').map(num => parseInt(num));
        if (startPage >= endPage) {
            this.showError('Start page must be less than end page');
            return;
        }
        
        // Store current page range and clear TOC selections
        this.currentPageRange = pageRange;
        this.currentTOCSections = [];
        this.currentMode = 'page_range';
        
        // Clear TOC UI selections
        document.querySelectorAll('#tocList > div').forEach(item => {
            item.classList.remove('bg-green-100', 'border-green-300');
        });
        
        // Update UI
        if (this.pageRangeDisplay) this.pageRangeDisplay.textContent = pageRange;
        if (this.currentPageRangeDiv) this.currentPageRangeDiv.classList.remove('hidden');
        
        // Scroll to the start page of the range
        if (this.pdfDoc && startPage > 0) {
            console.log(`Scrolling to page range start: page ${startPage}`);
            this.scrollToPage(startPage);
        }
        
        // Update chat mode
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    clearPageRange() {
        this.currentPageRange = null;
        this.currentMode = 'pdf';  // Return to PDF mode
        if (this.pageRangeInput) this.pageRangeInput.value = '';
        if (this.currentPageRangeDiv) this.currentPageRangeDiv.classList.add('hidden');
        this.updateChatMode();
        this.updateModeTooltip();
    }
    
    showLoadingMessage(message) {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'pdf-loading';
        loadingDiv.className = 'flex items-center justify-center h-64';
        loadingDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner mx-auto mb-4"></div>
                <div class="text-gray-600">${message}</div>
            </div>
        `;
        this.pdfPagesContainer.appendChild(loadingDiv);
    }
    
    hideLoadingMessage() {
        const loadingDiv = document.getElementById('pdf-loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }
    
    checkBirthday() {
        const today = new Date();
        const month = today.getMonth() + 1; // January is 0
        const day = today.getDate();
        
        // Check for birthday (October 8th = 10/8, but testing with September 5th = 9/5)
        if (month === 10 && day === 8) {
            this.isBirthday = true;
            this.showBirthdayPopup();
        } else {
            this.isBirthday = false;
        }
        
        // Check for anniversary (June 7, 2025 - 7th of every month)
        if (day === 7) {
            this.isAnniversary = true;
            this.showAnniversaryPopup();
        } else {
            this.isAnniversary = false;
        }
    }
    
    showBirthdayPopup() {
        // Create birthday popup overlay
        const overlay = document.createElement('div');
        overlay.id = 'birthday-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            backdrop-filter: blur(5px);
        `;
        
        // Create birthday popup
        const popup = document.createElement('div');
        popup.style.cssText = `
            background: linear-gradient(135deg, #ff6b9d, #ff8fab, #ffa8cc);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(255, 107, 157, 0.3);
            animation: birthdayBounce 0.8s ease-out;
            max-width: 400px;
            position: relative;
        `;
        
        popup.innerHTML = `
            <div style="font-size: 48px; margin-bottom: 20px; animation: heartFloat 2s ease-in-out infinite;">
                💖✨💖
            </div>
            <h2 style="color: white; font-size: 32px; margin: 0 0 10px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: bold;">
                Happy Birthday Kris!
            </h2>
            <p style="color: white; font-size: 18px; margin: 20px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🎉 I like you 🎂
            </p>
            <button id="birthday-close" style="
                background: white;
                color: #ff6b9d;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            ">
                Close 💕
            </button>
        `;
        
        overlay.appendChild(popup);
        document.body.appendChild(overlay);
        
        // Add CSS animations
        if (!document.getElementById('birthday-styles')) {
            const style = document.createElement('style');
            style.id = 'birthday-styles';
            style.textContent = `
                @keyframes birthdayBounce {
                    0% { transform: scale(0.3) rotate(-10deg); opacity: 0; }
                    50% { transform: scale(1.05) rotate(5deg); }
                    100% { transform: scale(1) rotate(0deg); opacity: 1; }
                }
                
                @keyframes heartFloat {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-10px); }
                }
                
                @keyframes fadeOut {
                    0% { opacity: 1; }
                    100% { opacity: 0; }
                }
                
                @keyframes popupFadeOut {
                    0% { opacity: 1; transform: scale(1); }
                    100% { opacity: 0; transform: scale(0.8); }
                }
                
                #birthday-close:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
                    background: #fff5f8;
                }
            `;
            document.head.appendChild(style);
        }
        
        // Close popup functionality
        document.getElementById('birthday-close').addEventListener('click', () => {
            overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
            popup.style.animation = 'popupFadeOut 0.3s ease-out forwards';
            setTimeout(() => overlay.remove(), 300);
        });
        
        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
                popup.style.animation = 'popupFadeOut 0.3s ease-out forwards';
                setTimeout(() => overlay.remove(), 300);
            }
        });
        
        // Auto-close after 10 seconds
        setTimeout(() => {
            if (document.getElementById('birthday-overlay')) {
                overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
                popup.style.animation = 'popupFadeOut 0.3s ease-out forwards';
                setTimeout(() => overlay.remove(), 300);
            }
        }, 10000);
    }
    
    createHeartsFromButton() {
        if (!this.sendButton) return;
        
        // Get button position
        const buttonRect = this.sendButton.getBoundingClientRect();
        const buttonCenterX = buttonRect.left + buttonRect.width / 2;
        const buttonCenterY = buttonRect.top + buttonRect.height / 2;
        
        // Create multiple hearts
        for (let i = 0; i < 6; i++) {
            setTimeout(() => {
                this.createSingleHeart(buttonCenterX, buttonCenterY);
            }, i * 100); // Stagger the hearts
        }
    }
    
    createSingleHeart(startX, startY) {
        const heart = document.createElement('div');
        heart.innerHTML = '💖';
        heart.style.cssText = `
            position: fixed;
            left: ${startX}px;
            top: ${startY}px;
            font-size: 20px;
            pointer-events: none;
            z-index: 9999;
            animation: heartFlyUp 2s ease-out forwards;
        `;
        
        // Add random horizontal movement
        const randomX = (Math.random() - 0.5) * 200; // -100px to +100px
        heart.style.setProperty('--random-x', `${randomX}px`);
        
        document.body.appendChild(heart);
        
        // Add the flying animation if not already added
        if (!document.getElementById('heart-fly-styles')) {
            const style = document.createElement('style');
            style.id = 'heart-fly-styles';
            style.textContent = `
                @keyframes heartFlyUp {
                    0% {
                        transform: translate(0, 0) scale(0.5);
                        opacity: 1;
                    }
                    50% {
                        transform: translate(var(--random-x), -150px) scale(1);
                        opacity: 1;
                    }
                    100% {
                        transform: translate(var(--random-x), -250px) scale(0.3);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Remove heart after animation
        setTimeout(() => {
            if (heart.parentNode) {
                heart.remove();
            }
        }, 2000);
    }
    
    showAnniversaryPopup() {
        const today = new Date();
        const startDate = new Date(2025, 5, 7); // June 7, 2025 (month is 0-indexed)
        
        // Calculate months since we started dating
        let months = (today.getFullYear() - startDate.getFullYear()) * 12;
        months += today.getMonth() - startDate.getMonth();
        
        // If we haven't reached the anniversary day yet this month, subtract 1
        if (today.getDate() < startDate.getDate()) {
            months--;
        }
        
        // Only show if it's been at least 1 month
        if (months < 1) return;
        
        // Create anniversary popup overlay
        const overlay = document.createElement('div');
        overlay.id = 'anniversary-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            backdrop-filter: blur(5px);
        `;
        
        // Create anniversary popup
        const popup = document.createElement('div');
        popup.style.cssText = `
            background: linear-gradient(135deg, #ffd700, #ffb347, #ff69b4);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(255, 215, 0, 0.3);
            animation: anniversaryBounce 0.8s ease-out;
            max-width: 400px;
            position: relative;
        `;
        
        const monthText = months === 1 ? "1 month" : `${months} months`;
        
        popup.innerHTML = `
            <div style="font-size: 48px; margin-bottom: 20px; animation: sparkleFloat 2s ease-in-out infinite;">
                ✨💕✨
            </div>
            <h2 style="color: white; font-size: 28px; margin: 0 0 10px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: bold;">
                Happy ${monthText} Anniversary!
            </h2>
            <p style="color: white; font-size: 18px; margin: 20px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🌟 U are pretty cool 💫
            </p>
            <button id="anniversary-close" style="
                background: white;
                color: #ffd700;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            ">
                Close ✨
            </button>
        `;
        
        overlay.appendChild(popup);
        document.body.appendChild(overlay);
        
        // Add CSS animations for anniversary
        if (!document.getElementById('anniversary-styles')) {
            const style = document.createElement('style');
            style.id = 'anniversary-styles';
            style.textContent = `
                @keyframes anniversaryBounce {
                    0% { transform: scale(0.3) rotate(-10deg); opacity: 0; }
                    50% { transform: scale(1.05) rotate(5deg); }
                    100% { transform: scale(1) rotate(0deg); opacity: 1; }
                }
                
                @keyframes sparkleFloat {
                    0%, 100% { transform: translateY(0px) rotate(0deg); }
                    50% { transform: translateY(-10px) rotate(180deg); }
                }
                
                @keyframes anniversaryFadeOut {
                    0% { opacity: 1; }
                    100% { opacity: 0; }
                }
                
                @keyframes anniversaryPopupFadeOut {
                    0% { opacity: 1; transform: scale(1); }
                    100% { opacity: 0; transform: scale(0.8); }
                }
                
                #anniversary-close:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
                    background: #fff9e6;
                }
            `;
            document.head.appendChild(style);
        }
        
        // Close popup functionality
        document.getElementById('anniversary-close').addEventListener('click', () => {
            overlay.style.animation = 'anniversaryFadeOut 0.3s ease-out forwards';
            popup.style.animation = 'anniversaryPopupFadeOut 0.3s ease-out forwards';
            setTimeout(() => overlay.remove(), 300);
        });
        
        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.style.animation = 'anniversaryFadeOut 0.3s ease-out forwards';
                popup.style.animation = 'anniversaryPopupFadeOut 0.3s ease-out forwards';
                setTimeout(() => overlay.remove(), 300);
            }
        });
        
        // Auto-close after 12 seconds (a bit longer for anniversary)
        setTimeout(() => {
            if (document.getElementById('anniversary-overlay')) {
                overlay.style.animation = 'anniversaryFadeOut 0.3s ease-out forwards';
                popup.style.animation = 'anniversaryPopupFadeOut 0.3s ease-out forwards';
                setTimeout(() => overlay.remove(), 300);
            }
        }, 12000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Set PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    
    // Initialize app and make it globally available for citation clicks
    window.app = new PDFChatApp();
});