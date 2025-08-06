import os
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import numpy as np
import faiss
from pdf_processor import pdf_processor
from pydantic import BaseModel
from typing import List
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Database setup
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# FAISS setup (example: 128-dim vectors, L2 index)
dim = 128
faiss_index = faiss.IndexFlatL2(dim)

# Pydantic models for request/response
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    return {"message": "FastAPI + Postgres + FAISS setup working!"}

@app.post("/faiss/add")
async def add_vector():
    # Example: add a random vector
    vec = np.random.rand(1, dim).astype('float32')
    faiss_index.add(vec)
    return {"status": "vector added", "vector": vec.tolist()}

@app.get("/faiss/search")
async def search_vector():
    # Example: search for a random vector
    query = np.random.rand(1, dim).astype('float32')
    D, I = faiss_index.search(query, k=1)
    return {"query": query.tolist(), "distances": D.tolist(), "indices": I.tolist()}

@app.post("/pdf/upload_and_process")
async def upload_and_process_pdf(
    file: UploadFile = File(...),
    resource_id: str = Form(...),
    project_id: str = Form(...)
):
    # Save uploaded file
    upload_dir = "uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # Process PDF
    docs = pdf_processor.process_pdf(file_path, resource_id, project_id, file.filename)
    if not docs:
        return JSONResponse(status_code=400, content={"error": "Failed to process PDF or extract text."})
    # Save to vector store
    pdf_processor.save_or_update_vector_store(docs, project_id)
    return {"message": f"Processed and indexed {file.filename}", "num_chunks": len(docs)}

@app.post("/pdf/search")
async def search_pdf(
    query: str = Form(...),
    project_id: str = Form(...),
    resource_ids: str = Form(None)  # comma-separated
):
    resource_id_list = resource_ids.split(",") if resource_ids else None
    results = pdf_processor.get_relevant_documents(query, project_id, resource_id_list)
    return {"results": [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]}

def generate_llm_answer(question, context):
    prompt = (
        f"Based on the provided insurance policy document context, answer the following question with specific details, exact numbers, and policy terms. If the information is not found in the context, say 'Information not found in the provided document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a clear, specific answer with exact details from the policy document:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,  # Lower temperature for more consistent answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest):
    try:
        # Download PDF from URL
        response = requests.get(request.documents, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from URL")
        
        # Save PDF temporarily
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, "temp_document.pdf")
        
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        # Process PDF
        resource_id = "hackrx_doc"
        project_id = "hackrx_project"
        docs = pdf_processor.process_pdf(pdf_path, resource_id, project_id, "hackrx_document.pdf")
        
        if not docs:
            raise HTTPException(status_code=400, detail="Failed to process PDF or extract text")
        
        # Save to vector store
        pdf_processor.save_or_update_vector_store(docs, project_id)
        
        # Generate answers for each question using LLM (optimized for accuracy)
        answers = []
        for question in request.questions:
            # Search for relevant documents
            results = pdf_processor.get_relevant_documents(question, project_id)
            if results:
                # Use top 4 chunks for better accuracy
                context = "\n\n".join([doc.page_content for doc in results[:4]])
                answer = generate_llm_answer(question, context)
            else:
                answer = "No relevant information found for this question."
            answers.append(answer)
        
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 