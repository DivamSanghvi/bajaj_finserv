#!/usr/bin/env python3
"""
Optimized version of main.py with reduced resource usage for debugging
"""
import os
import requests
import json
import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import PyPDF2
from io import BytesIO

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
def setup_logging():
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"optimized_api_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
app = FastAPI()

# Simplified models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Authentication
async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = authorization.replace("Bearer ", "")
    expected_token = "343c934c163f8f87a6a809c5c79729281f6fdbf03592227539766d3097f11fcd"
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def extract_text_from_pdf_simple(pdf_content):
    """Simple PDF text extraction using PyPDF2 only"""
    try:
        start_time = time.time()
        logger.info("📄 Starting simple PDF text extraction...")
        
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        max_pages = min(len(pdf_reader.pages), 20)  # Limit to 20 pages for speed
        
        for i, page in enumerate(pdf_reader.pages[:max_pages]):
            page_text = page.extract_text()
            text += page_text + "\n"
            
            if (i + 1) % 5 == 0:
                logger.info(f"📄 Processed page {i+1}/{max_pages}")
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ PDF text extraction completed in {extraction_time:.2f}s")
        logger.info(f"📊 Extracted {len(text)} characters from {max_pages} pages")
        
        return text
    except Exception as e:
        logger.error(f"❌ PDF extraction error: {e}")
        return ""

def simple_text_chunking(text, chunk_size=1000, overlap=100):
    """Simple text chunking without heavy dependencies"""
    try:
        start_time = time.time()
        logger.info("🔪 Starting simple text chunking...")
        
        if len(text) < chunk_size:
            logger.info("📝 Text is small, returning as single chunk")
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size - 200:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        chunking_time = time.time() - start_time
        logger.info(f"✅ Text chunking completed in {chunking_time:.2f}s")
        logger.info(f"📊 Created {len(chunks)} chunks")
        
        return chunks[:50]  # Limit to 50 chunks for memory efficiency
    except Exception as e:
        logger.error(f"❌ Text chunking error: {e}")
        return [text[:2000]]  # Fallback to truncated text

def generate_simple_answer(question, context):
    """Generate answer using OpenAI without complex processing"""
    try:
        start_time = time.time()
        logger.info(f"🤖 Generating answer for: {question[:50]}...")
        
        # Limit context size to prevent token limits
        max_context = 3000
        if len(context) > max_context:
            context = context[:max_context] + "..."
            logger.info(f"📝 Truncated context to {max_context} characters")
        
        prompt = f"""
Based on the following policy document context, answer the question concisely and accurately.

Context:
{context}

Question: {question}

Answer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        
        answer = response.choices[0].message.content.strip()
        
        answer_time = time.time() - start_time
        logger.info(f"✅ Answer generated in {answer_time:.2f}s")
        
        return answer
    except Exception as e:
        logger.error(f"❌ Answer generation error: {e}")
        return "Error generating answer. Please try again."

def find_relevant_chunks(question, chunks):
    """Simple keyword-based relevance without embeddings"""
    try:
        start_time = time.time()
        logger.info("🔍 Finding relevant chunks using keyword matching...")
        
        # Extract keywords from question
        question_lower = question.lower()
        keywords = [word.strip('?.,!') for word in question_lower.split() 
                   if len(word) > 3 and word not in ['what', 'when', 'where', 'does', 'this']]
        
        logger.info(f"🔑 Keywords: {keywords}")
        
        # Score chunks based on keyword matches
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            if score > 0:
                chunk_scores.append((score, i, chunk))
        
        # Sort by score and return top chunks
        chunk_scores.sort(reverse=True)
        relevant_chunks = [chunk for score, i, chunk in chunk_scores[:5]]
        
        search_time = time.time() - start_time
        logger.info(f"✅ Found {len(relevant_chunks)} relevant chunks in {search_time:.2f}s")
        
        return relevant_chunks
    except Exception as e:
        logger.error(f"❌ Chunk search error: {e}")
        return chunks[:3]  # Fallback to first 3 chunks

@app.get("/")
async def root():
    return {"message": "Optimized FastAPI working!", "mode": "lightweight"}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run_optimized(request: HackRxRequest, token: str = Depends(verify_token)):
    overall_start = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("📥 OPTIMIZED REQUEST RECEIVED:")
        logger.info(f"📄 Documents URL: {request.documents}")
        logger.info(f"❓ Questions ({len(request.questions)}):")
        for i, question in enumerate(request.questions, 1):
            logger.info(f"   {i}. {question}")
        logger.info("=" * 60)
        
        # STEP 1: Download PDF
        step_start = time.time()
        logger.info("🔽 STEP 1: Downloading PDF...")
        response = requests.get(request.documents, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF")
        logger.info(f"✅ STEP 1: PDF downloaded ({len(response.content)} bytes) in {time.time() - step_start:.2f}s")

        # STEP 2: Extract text (simplified)
        step_start = time.time()
        logger.info("📄 STEP 2: Extracting text from PDF...")
        text = extract_text_from_pdf_simple(response.content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        logger.info(f"✅ STEP 2: Text extracted in {time.time() - step_start:.2f}s")

        # STEP 3: Simple text chunking
        step_start = time.time()
        logger.info("🔪 STEP 3: Chunking text...")
        chunks = simple_text_chunking(text)
        logger.info(f"✅ STEP 3: Text chunked in {time.time() - step_start:.2f}s")

        # STEP 4: Answer questions
        step_start = time.time()
        logger.info("🤖 STEP 4: Generating answers...")
        answers = []
        
        for i, question in enumerate(request.questions):
            q_start = time.time()
            logger.info(f"🔍 Processing question {i+1}: {question}")
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question, chunks)
            context = "\n\n".join(relevant_chunks)
            
            # Generate answer
            answer = generate_simple_answer(question, context)
            answers.append(answer)
            
            logger.info(f"✅ Question {i+1} answered in {time.time() - q_start:.2f}s")
        
        logger.info(f"✅ STEP 4: All answers generated in {time.time() - step_start:.2f}s")

        total_time = time.time() - overall_start
        logger.info("=" * 60)
        logger.info("📤 OPTIMIZED RESPONSE READY:")
        logger.info(f"📝 Generated {len(answers)} answers")
        logger.info(f"⏱️ TOTAL TIME: {total_time:.2f} seconds")
        logger.info("=" * 60)

        return HackRxResponse(answers=answers)

    except Exception as e:
        total_time = time.time() - overall_start
        logger.error(f"❌ ERROR after {total_time:.2f}s: {str(e)}")
        import traceback
        logger.error(f"❌ TRACEBACK:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
