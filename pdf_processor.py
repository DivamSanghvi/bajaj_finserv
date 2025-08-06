import os
import fitz
import numpy as np
import cv2
import easyocr
from langdetect import detect
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_dir = "vector_stores"
        os.makedirs(self.vector_store_dir, exist_ok=True)
        self.vector_stores = {}
        self.allow_dangerous_deserialization = True
        logger.info("PDFProcessor initialized with vector store directory: %s", self.vector_store_dir)

    def extract_text_pymupdf(self, pdf_path, pages=None):
        try:
            document = fitz.open(pdf_path)
            text = ""
            max_pages = pages if pages else len(document)
            for page_num in range(min(max_pages, len(document))):
                page = document.load_page(page_num)
                page_text = page.get_text("text")
                if isinstance(page_text, str):
                    text += page_text + "\n"
                else:
                    logger.warning(f"Page {page_num} returned non-string text: {type(page_text)}")
            document.close()
            return text
        except Exception as e:
            logger.error(f"Error during PyMuPDF extraction: {e}")
            return ""

    def perform_ocr_on_pdf(self, pdf_path):
        try:
            pages = convert_from_path(pdf_path, 300)
            reader = easyocr.Reader(['en', 'hi'])
            extracted_text = ""
            for page in pages:
                open_cv_image = np.array(page)[:, :, ::-1].copy()
                result = reader.readtext(open_cv_image)
                for detection in result:
                    if isinstance(detection[1], str):
                        extracted_text += detection[1] + "\n"
                    else:
                        logger.warning(f"OCR returned non-string text: {type(detection[1])}")
            return extracted_text
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return None

    def process_pdf(self, pdf_path, resource_id, project_id=None, original_filename=None):
        try:
            logger.info(f"🔄 PROCESS_PDF FUNCTION:")
            logger.info(f"   📋 PDF path: {pdf_path}")
            logger.info(f"   📋 Resource ID: {resource_id}")
            logger.info(f"   📋 Project ID: {project_id}")
            logger.info(f"   📋 Original filename: '{original_filename}'")
            
            # Extract text using PyMuPDF
            text = self.extract_text_pymupdf(pdf_path)
            
            # If text extraction fails or returns empty, try OCR
            if not text.strip():
                logger.info(f"Text extraction failed for {pdf_path}, trying OCR")
                text = self.perform_ocr_on_pdf(pdf_path)
            
            if not text.strip():
                logger.error(f"Failed to extract text from {pdf_path}")
                return []
            
            # Create documents from the text with enhanced metadata
            documents = self.create_documents(text, resource_id, project_id, original_filename)
            
            if not documents:
                logger.warning(f"No documents created from {pdf_path}")
                return []
            
            logger.info(f"Successfully processed {pdf_path}, created {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def chunk_text(self, text):
        try:
            if not isinstance(text, str):
                logger.error(f"Expected string for chunking, got {type(text)}")
                return []
                
            # Optimized text splitter for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Better chunks for accuracy
                chunk_overlap=100,  # More overlap for better context
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                is_separator_regex=False,
                keep_separator=True
            )
            chunks = text_splitter.split_text(text)
            
            # Post-process chunks to fix word boundaries
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Clean up the chunk
                chunk = chunk.strip()
                if not chunk:
                    continue
                    
                # Ensure chunk doesn't start or end with partial words
                if i > 0 and chunk and chunk[0].islower() and processed_chunks:
                    # Check if this looks like a continuation of the previous chunk
                    prev_chunk = processed_chunks[-1]
                    if prev_chunk and not prev_chunk.endswith(('.', '!', '?', '\n')):
                        # Merge with previous chunk to avoid word splitting
                        processed_chunks[-1] = prev_chunk + chunk
                        continue
                
                processed_chunks.append(chunk)
            
            logger.info(f"Split text into {len(processed_chunks)} chunks (original: {len(chunks)})")
            return processed_chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []

    def create_documents(self, pdf_text, resource_id, project_id=None, original_filename=None):
        try:
            if not isinstance(pdf_text, str):
                logger.error(f"Expected string for document creation, got {type(pdf_text)}")
                return []
            
            # Debug logging for parameters
            logger.info(f"🔍 CREATE_DOCUMENTS DEBUG:")
            logger.info(f"   📋 resource_id: {resource_id} (type: {type(resource_id)})")
            logger.info(f"   📋 project_id: {project_id} (type: {type(project_id)})")
            logger.info(f"   📋 original_filename: '{original_filename}' (type: {type(original_filename)})")
                
            all_docs = []
            chunks = self.chunk_text(pdf_text)
            if chunks is None:
                return all_docs
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    # Minimal metadata with only required fields
                    metadata = {
                        "chunk_id": i,
                        "resource_id": str(resource_id),  # Always store as string
                    }
                    
                    # Add resource name if provided
                    if original_filename:
                        metadata["resource_name"] = original_filename
                        logger.info(f"✅ Added resource_name to metadata: {original_filename}")
                    else:
                        logger.warning(f"⚠️  original_filename is None/empty, skipping resource_name in metadata")
                    
                    # Debug log the metadata for first chunk
                    if i == 0:
                        logger.info(f"📊 Sample metadata for chunk 0: {metadata}")
                    
                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    all_docs.append(doc)
            
            logger.info(f"Created {len(all_docs)} documents from text chunks for resource_id: {resource_id} (filename: {original_filename})")
            return all_docs
        except Exception as e:
            logger.error(f"Error creating documents: {str(e)}")
            return []

    def get_vector_store_path(self, project_id):
        return os.path.join(self.vector_store_dir, f"project_{project_id}")

    def save_or_update_vector_store(self, documents, project_id):
        try:
            if not documents:
                logger.warning(f"No documents to save for project {project_id}")
                return False

            # Create project directory if it doesn't exist
            project_dir = self.get_vector_store_path(project_id)
            os.makedirs(project_dir, exist_ok=True)
            logger.info(f"Created/verified project directory: {project_dir}")

            # Check if vector store exists
            if project_id in self.vector_stores:
                # Update existing vector store
                self.vector_stores[project_id].add_documents(documents)
                logger.info(f"Updated existing vector store for project {project_id}")
            else:
                # Create new vector store
                self.vector_stores[project_id] = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created new vector store for project {project_id}")
            
            # Save to disk
            self.vector_stores[project_id].save_local(project_dir)
            logger.info(f"Successfully saved vector store for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False

    def load_and_retrieve_filtered_documents(self, query, project_id, target_resource_ids=None):
        """Load FAISS vector store and retrieve relevant documents."""
        try:
            project_dir = self.get_vector_store_path(project_id)
            logger.info(f"Getting relevant documents for project {project_id}")
            
            # Always reload from disk to ensure we have the latest documents (including newly added ones)
            if os.path.exists(os.path.join(project_dir, "index.faiss")):
                try:
                    # Load from disk - try with allow_dangerous_deserialization first, fallback if not supported
                    try:
                        self.vector_stores[project_id] = FAISS.load_local(project_dir, self.embeddings, allow_dangerous_deserialization=True)
                    except TypeError:
                        # Fallback for older versions that don't support allow_dangerous_deserialization
                        self.vector_stores[project_id] = FAISS.load_local(project_dir, self.embeddings)
                    logger.info(f"🔄 Reloaded vector store for project {project_id} from disk")
                except Exception as e:
                    logger.error(f"Error loading vector store from disk: {str(e)}")
                    return []
            else:
                logger.warning(f"No vector store found for project {project_id}")
                return []

            store = self.vector_stores[project_id]
            print("Index loaded successfully.")
            
            # Get all documents from the vector store
            all_documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                           for doc in store.docstore._dict.values()]
            
            # Filter documents by target resource IDs if specified
            if target_resource_ids and len(target_resource_ids) > 0:
                # Convert resource IDs to strings for comparison (as they're stored as strings in metadata)
                target_resource_ids_str = [str(rid) for rid in target_resource_ids]
                
                # Debug: Show what we're looking for vs what's available
                available_resource_ids = set(doc.metadata.get("resource_id") for doc in all_documents)
                print(f"🔍 DEBUG FILTERING:")
                print(f"   🎯 Looking for resource IDs: {target_resource_ids_str} (type: {type(target_resource_ids_str[0])})")
                print(f"   📋 Available in documents: {available_resource_ids}")
                
                filtered_docs = [doc for doc in all_documents if doc.metadata.get("resource_id") in target_resource_ids_str]
                print(f"✅ filtered_docs: Found {len(filtered_docs)} documents matching target resource IDs: {target_resource_ids}")
            else:
                filtered_docs = all_documents
                print(f"No target resource IDs specified, using all {len(filtered_docs)} documents")
            
            if not filtered_docs:
                logger.warning(f"No documents found after filtering for project {project_id}")
                return []
            
            # Create BM25 retriever from filtered documents
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = 6  # Increased for better accuracy
            
            # Create FAISS retriever - if filtering is applied, create from filtered docs only
            if target_resource_ids and len(target_resource_ids) > 0:
                # When filtering by specific resources, create FAISS from filtered documents only
                if len(filtered_docs) > 0:
                    filtered_faiss_store = FAISS.from_documents(filtered_docs, self.embeddings)
                    faiss_retriever = filtered_faiss_store.as_retriever(search_kwargs={"k": 6})
                    print(f"🎯 Created FAISS retriever from {len(filtered_docs)} filtered documents")
                else:
                    # Fallback to empty results if no filtered docs
                    return []
            else:
                # When no filtering, use the full vector store
                faiss_retriever = store.as_retriever(search_kwargs={"k": 6})
                print(f"📋 Using full FAISS retriever with all documents")
            
            # Create ensemble retriever optimized for accuracy
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.2, 0.8]  # More weight to FAISS for better semantic understanding
            )
            
            # Get relevant documents
            results = ensemble_retriever.get_relevant_documents(query)
            
            # Validation: If filtering was applied, ensure no unwanted resource IDs leaked through
            if target_resource_ids and len(target_resource_ids) > 0:
                # Check if any results have resource IDs not in our target list
                result_resource_ids = [doc.metadata.get('resource_id') for doc in results]
                unwanted_ids = set(result_resource_ids) - set(target_resource_ids_str)
                if unwanted_ids:
                    print(f"⚠️  WARNING: Found unwanted resource IDs in results: {unwanted_ids}")
                    # Filter out unwanted results
                    results = [doc for doc in results if doc.metadata.get('resource_id') in target_resource_ids_str]
                    print(f"🔧 Filtered out unwanted documents. Final count: {len(results)}")
            
            # Print detailed information about the extracted documents to terminal
            print("\n" + "="*80)
            print(f"🔍 ENHANCED VECTOR SEARCH RESULTS FOR PROJECT {project_id}")
            print(f"📝 Query: {query}")
            if target_resource_ids:
                print(f"🎯 Target Resource IDs: {target_resource_ids}")
                print(f"🔒 FILTERED MODE: Only documents from specified resources")
            else:
                print(f"🌐 UNFILTERED MODE: Using all available documents")
            print(f"📊 Number of documents extracted: {len(results)}")
            print("="*80)
            
            for i, doc in enumerate(results, 1):
                print(f"\n📄 DOCUMENT {i}:")
                print(f"   Resource ID: {doc.metadata.get('resource_id', 'Unknown')}")
                
                # Get resource name from metadata or fetch from database if missing
                resource_name = doc.metadata.get('resource_name', 'Unknown')
                print(f"   Resource Name: {resource_name}")
                print(f"   Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
                print(f"   Content Preview (first 200 chars):")
                content_preview = doc.page_content[:200].replace('\n', ' ').strip()
                print(f"   \"{content_preview}{'...' if len(doc.page_content) > 200 else ''}\"")
                print(f"   Full Content Length: {len(doc.page_content)} characters")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"   Metadata: {doc.metadata}")
                print("-" * 40)
            
            print("="*80 + "\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    # Keep the old method for backward compatibility
    def get_relevant_documents(self, query, project_id, target_resource_ids=None):
        """Wrapper method for backward compatibility"""
        return self.load_and_retrieve_filtered_documents(query, project_id, target_resource_ids)

    def remove_resource_from_vector_store(self, project_id, resource_id):
        """Remove a resource's documents from the vector store."""
        project_store_path = self.get_vector_store_path(project_id)
        
        try:
            if os.path.exists(project_store_path):
                # Load existing vector store - try with allow_dangerous_deserialization first, fallback if not supported
                try:
                    vectorstore = FAISS.load_local(project_store_path, self.embeddings, allow_dangerous_deserialization=True)
                except TypeError:
                    # Fallback for older versions that don't support allow_dangerous_deserialization
                    vectorstore = FAISS.load_local(project_store_path, self.embeddings)
                
                # Get all documents
                documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                           for doc in vectorstore.docstore._dict.values()]
                
                # Filter out documents from the deleted resource
                filtered_docs = [doc for doc in documents 
                               if doc.metadata.get('resource_id') != str(resource_id)]
                
                # Create new vector store with filtered documents
                if filtered_docs:
                    new_vectorstore = FAISS.from_documents(filtered_docs, self.embeddings)
                    new_vectorstore.save_local(project_store_path)
                else:
                    # If no documents left, remove the vector store directory
                    import shutil
                    shutil.rmtree(project_store_path)
                
                logger.info(f"Successfully removed resource {resource_id} from vector store")
                return True
        except Exception as e:
            logger.error(f"Error removing resource from vector store: {e}")
            return False

def capture_browser_errors(url):
    """
    Captures browser console errors using Selenium.
    
    Args:
        url (str): The URL to check for console errors
        
    Returns:
        list: List of console errors found
    """
    try:
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Enable console logging
        chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
        
        # Initialize the driver
        from chromedriver_autoinstaller import install
        install()
        driver = webdriver.Chrome(options=chrome_options)
        
        # Store console logs
        console_errors = []
        
        try:
            # Navigate to the URL
            driver.get(url)
            
            # Wait for page to load (adjust timeout as needed)
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            
            # Get console logs
            logs = driver.get_log('browser')
            
            # Filter for errors and warnings
            for log in logs:
                if log['level'] in ['SEVERE', 'WARNING']:
                    console_errors.append({
                        'level': log['level'],
                        'message': log['message'],
                        'timestamp': log['timestamp']
                    })
                    logger.error(f"Browser Console {log['level']}: {log['message']}")
            
            return console_errors
            
        finally:
            # Always close the driver
            driver.quit()
            
    except Exception as e:
        logger.error(f"Error capturing browser console errors: {str(e)}")
        return [{
            'level': 'ERROR',
            'message': f"Failed to capture browser errors: {str(e)}",
            'timestamp': None
        }]

def check_browser_errors(url):
    """
    Checks for browser console errors and prints them to console.
    
    Args:
        url (str): The URL to check for console errors
    """
    errors = capture_browser_errors(url)
    
    if errors:
        print("\n=== Browser Console Errors ===")
        for error in errors:
            print(f"\nLevel: {error['level']}")
            print(f"Message: {error['message']}")
            if error['timestamp']:
                print(f"Timestamp: {error['timestamp']}")
        print("\n===========================")
    else:
        print("\nNo browser console errors found.")

pdf_processor = PDFProcessor() 