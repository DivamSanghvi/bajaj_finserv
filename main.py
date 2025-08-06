import os
import requests
import json
import re
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import numpy as np
import faiss
from pdf_processor import pdf_processor
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
def setup_logging():
    """Set up file logging for requests and responses."""
    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique log file for each session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"api_log_{timestamp}.txt")
    
    # Configure logging with DEBUG level for detailed analysis
    logging.basicConfig(
        level=logging.DEBUG,  # Changed from INFO to DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Global cache for vector stores to avoid redundant reloading
vector_store_cache = {}

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

class DetailedAnalysisRequest(BaseModel):
    documents: str
    questions: List[str]

class DetailedAnalysisResponse(BaseModel):
    analyses: List[Dict[str, Any]]

class PolicyAnalysis(BaseModel):
    clause_id: str
    relevance_score: float
    clause_type: str  # 'inclusion', 'exclusion', 'condition', 'general'
    matched_criteria: List[str]
    extracted_rules: Dict[str, Any]
    reasoning: str

class DecisionResult(BaseModel):
    decision: str  # 'approved', 'rejected', 'requires_review'
    confidence_score: float
    reasoning: str
    risk_factors: List[str]
    recommendations: List[str]

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

def parse_json_response(text: str) -> dict:
    """Safely parse JSON from a string that might contain markdown."""
    # Try to extract JSON from markdown code blocks
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON response. Response text: {text}")
        return {"error": "Failed to parse LLM response", "raw_response": text}

def analyze_policy_clauses(question: str, document_chunks: List[str]) -> List[PolicyAnalysis]:
    """Analyze policy clauses against the question using structured approach."""
    try:
        context_text = "\n\n".join(document_chunks)
        
        # Limit context to avoid token limits
        if len(context_text) > 3000:
            context_text = context_text[:3000] + "..."

        prompt = f"""
        Analyze these policy document sections against the insurance question. Return a JSON array of relevant clauses.

        Question: {question}

        Policy Sections:
        {context_text}

        For each relevant section, return a JSON array with this structure:
        [{{
            "clause_id": "A unique reference or the first 10 words of the clause",
            "relevance_score": 0.95,
            "clause_type": "inclusion",
            "matched_criteria": ["grace period", "premium payment"],
            "extracted_rules": {{
                "waiting_period_months": 30,
                "coverage_amount": null,
                "exclusions_mentioned": [],
                "conditions_mentioned": ["timely payment"]
            }},
            "reasoning": "A brief explanation of how this clause applies to the question. Use single quotes for any internal quotes."
        }}]

        Instructions:
        1. Only include clauses that are directly relevant to the question
        2. Be precise with waiting periods and coverage amounts
        3. If no clauses are relevant, return an empty array []
        4. Focus on the most important policy terms and conditions
        5. Use realistic relevance scores between 0.1 and 1.0
        6. In reasoning, use single quotes for internal quotes, not double quotes

        Return only the JSON array, no additional text or markdown formatting.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,
        )

        result_text = response.choices[0].message.content.strip()
        logger.debug(f"Raw LLM response: {result_text}")
        
        result = parse_json_response(result_text)
        logger.debug(f"Parsed result: {result}")

        if isinstance(result, list):
            return [PolicyAnalysis(**clause) for clause in result]
        else:
            return []

    except Exception as e:
        logger.error(f"Error analyzing policy clauses: {str(e)}")
        return []

def generate_structured_answer(question: str, analyzed_clauses: List[PolicyAnalysis]) -> str:
    """Generate a structured answer based on policy analysis."""
    try:
        if not analyzed_clauses:
            return "Information not found in the provided document."

        # Sort by relevance score
        sorted_clauses = sorted(analyzed_clauses, key=lambda x: x.relevance_score, reverse=True)

        # Get the most relevant clause
        top_clause = sorted_clauses[0]

        # Determine answer type based on clause type
        if top_clause.clause_type == "exclusion":
            answer_start = "No"
        elif top_clause.clause_type == "inclusion":
            answer_start = "Yes"
        elif top_clause.clause_type == "condition":
            answer_start = "Yes, with conditions"
        else:
            answer_start = "Yes"

        # Extract key information
        waiting_period = top_clause.extracted_rules.get("waiting_period_months")
        coverage_amount = top_clause.extracted_rules.get("coverage_amount")

        # Build the answer
        answer_parts = [answer_start]

        if waiting_period:
            answer_parts.append(f"there is a {waiting_period}-month waiting period")
        if coverage_amount:
            answer_parts.append(f"with coverage up to {coverage_amount}")

        # Add reasoning with proper quote handling
        if top_clause.reasoning and top_clause.reasoning != "No reasoning provided":
            # Clean the reasoning text
            clean_reasoning = top_clause.reasoning.replace('\\"', '"').replace('"', '"').replace('"', '"')
            answer_parts.append(f"According to {top_clause.clause_id}: '{clean_reasoning}'")

        # Clean and format the answer
        answer = ", ".join(answer_parts) + "."
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        answer = answer.replace('\\', '')
        answer = ' '.join(answer.split())

        return answer
        
    except Exception as e:
        logger.error(f"Error in generate_structured_answer: {str(e)}")
        return "Information not found in the provided document."

def generate_decision_reasoning(question: str, analyzed_clauses: List[PolicyAnalysis]) -> DecisionResult:
    """Generate detailed decision reasoning based on policy analysis."""
    if not analyzed_clauses:
        return DecisionResult(
            decision="requires_review",
            confidence_score=0.0,
            reasoning="No relevant policy clauses found",
            risk_factors=["Insufficient information"],
            recommendations=["Request additional documentation"]
        )

    # Sort by relevance and type
    exclusions = [c for c in analyzed_clauses if c.clause_type == "exclusion"]
    inclusions = [c for c in analyzed_clauses if c.clause_type == "inclusion"]
    conditions = [c for c in analyzed_clauses if c.clause_type == "condition"]

    # Decision logic based on Gemini approach
    decision = "requires_review"
    confidence_score = 0.5
    reasoning_parts = []
    risk_factors = []
    recommendations = []

    # Rule 1: Exclusion Priority
    if exclusions:
        decision = "rejected"
        confidence_score = 0.9
        reasoning_parts.append("Exclusion clause applies")
        risk_factors.append("Policy exclusion identified")
        recommendations.append("Review exclusion details")

    # Rule 2: Waiting Period Validation
    waiting_periods = []
    for clause in analyzed_clauses:
        waiting_period = clause.extracted_rules.get("waiting_period_months")
        if waiting_period:
            waiting_periods.append(waiting_period)
    
    if waiting_periods:
        max_waiting_period = max(waiting_periods)
        reasoning_parts.append(f"Waiting period of {max_waiting_period} months applies")
        if decision != "rejected":
            decision = "approved"
            confidence_score = 0.8

    # Rule 3: Approval Logic
    if inclusions and decision != "rejected":
        decision = "approved"
        confidence_score = 0.85
        reasoning_parts.append("Coverage clause applies")

    # Rule 4: Conditions
    if conditions:
        reasoning_parts.append("Conditions must be met")
        if decision == "approved":
            confidence_score = 0.7
        recommendations.append("Verify all conditions are satisfied")

    # Build final reasoning
    reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Analysis completed"
    
    return DecisionResult(
        decision=decision,
        confidence_score=confidence_score,
        reasoning=reasoning,
        risk_factors=risk_factors,
        recommendations=recommendations
    )

def generate_llm_answer(question, context):
    """Enhanced LLM answer generation with structured analysis."""
    try:
        # First, try the structured analysis approach
        document_chunks = context.split("\n\n")
        analyzed_clauses = analyze_policy_clauses(question, document_chunks)
        
        if analyzed_clauses:
            # Generate structured answer
            answer = generate_structured_answer(question, analyzed_clauses)
            return answer
        else:
            # Fallback to the original simple approach if structured analysis fails
            prompt = (
                f"Answer the question with specific details and citations from the provided document. Be concise and to-the-point.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Instructions:\n"
                f"1. Start your answer with 'Yes' or 'No' as the first word\n"
                f"2. Give direct, concise answers (1-3 sentences maximum)\n"
                f"3. Include the exact section/clause name when available\n"
                f"4. Quote the most relevant text from the document in quotes\n"
                f"5. If information is not found, say 'Information not found in the provided document'\n"
                f"6. Focus on the key facts, not explanations\n"
                f"7. Do not use line breaks or special formatting\n"
                f"8. Use single quotes for internal quotes, not double quotes\n"
                f"9. DO NOT overgeneralize or assume context unless explicitly stated\n"
                f"10. Use precise terms as defined in the document\n\n"
                f"Example format:\n"
                f"Answer: Yes, [direct answer]. According to [Section/Clause]: '[exact quote]'.\n"
                f"Answer: No, [direct answer]. According to [Section/Clause]: '[exact quote]'.\n\n"
                f"Answer:"
            )
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
            
            # Enhanced cleaning up the response
            answer = answer.replace('\n', ' ').replace('\r', ' ')  # Remove line breaks
            answer = answer.replace('\\', '')  # Remove backslashes
            answer = ' '.join(answer.split())  # Remove extra whitespace
            
            # Fix quote issues
            answer = answer.replace('\\"', '"')  # Fix escaped quotes
            answer = answer.replace('\\"', '"')  # Fix escaped quotes (alternative)
            answer = answer.replace('"', '"').replace('"', '"')  # Fix smart quotes
            answer = answer.replace('"', '"').replace('"', '"')  # Fix smart quotes
            
            # Remove any remaining escaped characters
            answer = answer.replace('\\n', ' ')
            answer = answer.replace('\\t', ' ')
            answer = answer.replace('\\r', ' ')
            
            # Final cleanup
            answer = ' '.join(answer.split())  # Remove any remaining extra spaces
            
            return answer
            
    except Exception as e:
        logger.error(f"Error in generate_llm_answer: {str(e)}")
        # Final fallback
        return f"Error generating answer: {str(e)}"

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

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, token: str = Depends(verify_token)):
    try:
        # Log the request body for debugging
        logger.info("=" * 50)
        logger.info("📥 RECEIVED REQUEST:")
        logger.info(f"📄 Documents URL: {request.documents}")
        logger.info(f"❓ Questions ({len(request.questions)}):")
        for i, question in enumerate(request.questions, 1):
            logger.info(f"   {i}. {question}")
        logger.info("=" * 50)
        
        # Download PDF from URL
        response = requests.get(request.documents, timeout=30)
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
        for i, question in enumerate(request.questions):
            logger.info(f"🔍 Processing question {i+1}/{len(request.questions)}: {question}")
            
            # Search for relevant documents
            results = pdf_processor.get_relevant_documents(question, project_id)
            if results:
                # Use top 6 chunks for better accuracy
                context = "\n\n".join([doc.page_content for doc in results[:6]])
                answer = generate_llm_answer(question, context)
                logger.info(f"✅ Answer {i+1}: {answer}")
            else:
                answer = "Information not found in the provided document."
                logger.info(f"❌ Answer {i+1}: {answer}")
            answers.append(answer)

        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        logger.info("=" * 50)
        logger.info("📤 SENDING RESPONSE:")
        logger.info(f"📝 Generated {len(answers)} answers")
        logger.info("=" * 50)

        return HackRxResponse(answers=answers)

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/hackrx/analyze", response_model=DetailedAnalysisResponse)
async def hackrx_detailed_analysis(request: DetailedAnalysisRequest, token: str = Depends(verify_token)):
    """Enhanced endpoint providing detailed policy analysis with structured insights."""
    try:
        # Log the request body for debugging
        logger.info("=" * 50)
        logger.info("📥 RECEIVED DETAILED ANALYSIS REQUEST:")
        logger.info(f"📄 Documents URL: {request.documents}")
        logger.info(f"❓ Questions ({len(request.questions)}):")
        for i, question in enumerate(request.questions, 1):
            logger.info(f"   {i}. {question}")
        logger.info("=" * 50)
        
        # Download PDF from URL
        response = requests.get(request.documents, timeout=30)
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

        # Generate detailed analysis for each question
        analyses = []
        for i, question in enumerate(request.questions):
            logger.info(f"🔍 Processing detailed analysis for question {i+1}/{len(request.questions)}: {question}")
            
            # Search for relevant documents
            results = pdf_processor.get_relevant_documents(question, project_id)
            
            if results:
                # Use top 6 chunks for better accuracy
                context = "\n\n".join([doc.page_content for doc in results[:6]])
                document_chunks = context.split("\n\n")
                
                # Analyze policy clauses
                analyzed_clauses = analyze_policy_clauses(question, document_chunks)
                logger.info(f"📋 Found {len(analyzed_clauses)} relevant clauses")
                
                # Generate decision reasoning
                decision_result = generate_decision_reasoning(question, analyzed_clauses)
                logger.info(f"🎯 Decision: {decision_result.decision} (confidence: {decision_result.confidence_score})")
                
                # Generate concise answer
                answer = generate_structured_answer(question, analyzed_clauses)
                logger.info(f"✅ Answer: {answer}")
                
                # Build comprehensive analysis
                analysis = {
                    "question": question,
                    "answer": answer,
                    "decision": decision_result.dict(),
                    "clauses_analyzed": len(analyzed_clauses),
                    "relevant_clauses": [
                        {
                            "clause_id": clause.clause_id,
                            "relevance_score": clause.relevance_score,
                            "clause_type": clause.clause_type,
                            "extracted_rules": clause.extracted_rules,
                            "reasoning": clause.reasoning
                        }
                        for clause in analyzed_clauses[:3]  # Top 3 most relevant
                    ]
                }
            else:
                logger.info(f"❌ No relevant information found for question {i+1}")
                analysis = {
                    "question": question,
                    "answer": "Information not found in the provided document.",
                    "decision": {
                        "decision": "requires_review",
                        "confidence_score": 0.0,
                        "reasoning": "No relevant information found",
                        "risk_factors": ["Document may not contain relevant information"],
                        "recommendations": ["Verify document content or try different search terms"]
                    },
                    "clauses_analyzed": 0,
                    "relevant_clauses": []
                }
            
            analyses.append(analysis)

        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        logger.info("=" * 50)
        logger.info("📤 SENDING DETAILED ANALYSIS RESPONSE:")
        logger.info(f"📝 Generated {len(analyses)} detailed analyses")
        logger.info("=" * 50)

        return DetailedAnalysisResponse(analyses=analyses)

    except Exception as e:
        logger.error(f"❌ ERROR in detailed analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 