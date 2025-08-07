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
# PDF processor will be imported lazily to reduce startup memory
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variable for lazy PDF processor initialization
_pdf_processor = None

def get_pdf_processor():
    """Lazy initialization of PDF processor to reduce startup memory usage."""
    global _pdf_processor
    if _pdf_processor is None:
        from pdf_processor import pdf_processor
        _pdf_processor = pdf_processor
    return _pdf_processor

# Enhanced Prompt Templates for RAG System
QUERY_STRUCTURING_PROMPT = """
You are an expert document analyst specializing in insurance, legal, and compliance documents. 

TASK: Parse the user query and extract structured information for semantic search.

INPUT QUERY: "{query}"

INSTRUCTIONS:
1. Extract key entities: age, gender, medical procedures, locations, policy duration, amounts, dates
2. Identify query intent: coverage check, waiting period, conditions, exclusions, definitions
3. Determine document sections to focus on: benefits, exclusions, definitions, conditions, sub-limits
4. Generate semantic search keywords including synonyms and related terms

OUTPUT FORMAT (JSON):
{{
    "entities": {{
        "demographics": [],
        "medical_info": [],
        "policy_info": [],
        "temporal": []
    }},
    "intent": "coverage_verification|definition_lookup|condition_check|calculation",
    "focus_sections": [],
    "search_terms": [],
    "query_complexity": "simple|moderate|complex"
}}

Return only valid JSON, no additional text.
"""

ENHANCED_ANALYSIS_PROMPT = """
You are an expert insurance policy analyst. Analyze the policy document sections to extract precise, comprehensive information.

QUERY: {question}

POLICY SECTIONS:
{context_text}

TASK: Extract ALL relevant clauses and conditions. For each relevant section, return detailed analysis.

CRITICAL REQUIREMENTS:
1. Include EXACT numerical values (waiting periods, amounts, percentages) with precise formatting
2. Capture ALL conditions, requirements, exceptions, and limitations
3. Extract complete coverage details including sub-limits and restrictions
4. Identify cross-references to other policy sections or external documents
5. Note specific terminology and definitions used in the policy
6. Include both positive coverage AND exclusions/limitations

Return a JSON array with this EXACT structure:
[{{
    "clause_id": "Exact section name or first 15 words of clause",
    "relevance_score": 0.95,
    "clause_type": "inclusion|exclusion|condition|definition|calculation",
    "matched_criteria": ["specific", "terms", "from", "query"],
    "extracted_rules": {{
        "waiting_period_months": null,
        "waiting_period_specific": "exact text if not in months",
        "coverage_amount": null,
        "coverage_percentage": null,
        "age_restrictions": [],
        "geographical_restrictions": [],
        "time_restrictions": [],
        "exclusions_mentioned": [],
        "conditions_mentioned": [],
        "sub_limits": {{}},
        "cross_references": [],
        "specific_requirements": []
    }},
    "exact_text_quote": "Verbatim text from policy document",
    "reasoning": "How this clause directly addresses the query"
}}]

FOCUS AREAS:
- Waiting periods and time-based restrictions
- Coverage amounts, percentages, and sub-limits
- Age, gender, and demographic requirements
- Pre-existing disease conditions
- Geographic and network restrictions
- Exclusions and exceptions
- Cross-references to schedules, tables, or other sections

Return only the JSON array, no markdown or additional text.
"""

COMPREHENSIVE_RESPONSE_PROMPT = """
You are an expert insurance policy analyst. Analyze the question type and respond according to these EXACT patterns:

QUERY: {original_query}
ANALYZED CLAUSES: {analyzed_clauses_json}

CRITICAL RESPONSE PATTERNS:

1. QUESTION STARTING WITH "What is/are..." → Direct explanation with details
   Example: "What is the grace period?" → "A grace period of thirty days is provided for premium payment after the due date..."

2. QUESTION STARTING WITH "Does/Is/Are..." → Start with "Yes" or "No" + explanation  
   Example: "Does this policy cover...?" → "Yes, the policy covers..."

3. QUESTION STARTING WITH "How..." → Direct explanation
   Example: "How does the policy define...?" → "A hospital is defined as..."

EXACT FORMAT REQUIREMENTS:
✅ Match the question type pattern exactly
✅ Include ALL numerical values with exact formatting: thirty-six (36) months, two (2) years
✅ Include specific section references when available: According to Section X.X
✅ Use exact terminology from policy document
✅ Include ALL conditions, limitations, and requirements
✅ No backslashes or escape characters in response

SAMPLE RESPONSES TO COPY:

"What is the grace period for premium payment?"
→ "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."

"Does this policy cover maternity expenses?"
→ "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."

"What is the waiting period for pre-existing diseases?"
→ "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."

Provide response following the EXACT pattern for the question type:
"""

VALIDATION_PROMPT = """
Validate the response follows the correct pattern for the question type:

QUERY: {query}
RESPONSE: {response}

PATTERN VALIDATION:
□ "What is/are..." questions → Start with direct explanation (NOT "Yes/No")
□ "Does/Is/Are..." questions → Start with "Yes" or "No" + explanation
□ "How..." questions → Start with direct explanation  
□ NO backslashes or escape characters (\\)
□ Numerical formatting: thirty-six (36) months, two (2) years
□ Include section references when available

SAMPLE CORRECT PATTERNS:
"What is the grace period?" → "A grace period of thirty days is provided..."
"Does this policy cover?" → "Yes, the policy covers..." OR "No, this policy does not cover..."

If response follows correct pattern: return "VALIDATED"
If needs improvement: provide corrected version with proper pattern.

OUTPUT:
"""

# Insurance-specific terminology and patterns
INSURANCE_SPECIFIC_TERMS = [
    "waiting period", "pre-existing diseases", "sum insured", "deductible",
    "co-payment", "sub-limits", "room rent", "ICU charges", "AYUSH treatment",
    "maternity benefits", "organ donor", "no claim discount", "grace period",
    "network hospitals", "cashless treatment", "reimbursement", "policy inception",
    "continuous coverage", "medical advice", "lawful medical termination",
    "caesarean delivery", "normal delivery", "complications", "direct complications"
]

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

def structure_query(question: str) -> dict:
    """Enhanced query structuring for better semantic search."""
    try:
        prompt = QUERY_STRUCTURING_PROMPT.format(query=question)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Minimal for query structuring
            temperature=0.1,
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.debug(f"Query structuring result: {result_text}")
        
        structured_query = parse_json_response(result_text)
        return structured_query if not structured_query.get("error") else {}
        
    except Exception as e:
        logger.error(f"Error in query structuring: {str(e)}")
        return {}

def enhanced_analyze_policy_clauses(question: str, document_chunks: List[str]) -> List[PolicyAnalysis]:
    """Enhanced policy clause analysis with comprehensive extraction."""
    try:
        context_text = "\n\n".join(document_chunks)
        
        # Increase context limit for better analysis
        if len(context_text) > 6000:
            context_text = context_text[:6000] + "..."

        prompt = ENHANCED_ANALYSIS_PROMPT.format(
            question=question,
            context_text=context_text
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,  # Back to original limit
            temperature=0.1,
        )

        result_text = response.choices[0].message.content.strip()
        logger.debug(f"Enhanced analysis result: {result_text}")
        
        result = parse_json_response(result_text)
        logger.debug(f"Parsed enhanced result: {result}")

        if isinstance(result, list):
            # Convert to PolicyAnalysis objects, handling new fields
            policy_analyses = []
            for clause in result:
                try:
                    # Map new structure to existing PolicyAnalysis model
                    policy_analysis = PolicyAnalysis(
                        clause_id=clause.get("clause_id", "Unknown"),
                        relevance_score=clause.get("relevance_score", 0.5),
                        clause_type=clause.get("clause_type", "general"),
                        matched_criteria=clause.get("matched_criteria", []),
                        extracted_rules=clause.get("extracted_rules", {}),
                        reasoning=clause.get("reasoning", "No reasoning provided")
                    )
                    policy_analyses.append(policy_analysis)
                except Exception as e:
                    logger.warning(f"Error creating PolicyAnalysis object: {e}")
                    continue
            return policy_analyses
        else:
            return []

    except Exception as e:
        logger.error(f"Error in enhanced policy clause analysis: {str(e)}")
        return []

def generate_comprehensive_answer(question: str, analyzed_clauses: List[PolicyAnalysis]) -> str:
    """Generate comprehensive answer using enhanced prompting."""
    try:
        if not analyzed_clauses:
            return "Information not found in the provided document."

        # Convert analyzed clauses to JSON for the prompt
        clauses_data = []
        for clause in analyzed_clauses:
            clauses_data.append({
                "clause_id": clause.clause_id,
                "relevance_score": clause.relevance_score,
                "clause_type": clause.clause_type,
                "matched_criteria": clause.matched_criteria,
                "extracted_rules": clause.extracted_rules,
                "reasoning": clause.reasoning
            })
        
        analyzed_clauses_json = json.dumps(clauses_data, indent=2)
        
        prompt = COMPREHENSIVE_RESPONSE_PROMPT.format(
            original_query=question,
            analyzed_clauses_json=analyzed_clauses_json
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,  # Back to original limit
            temperature=0.1,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Enhanced cleanup to fix backslash issues and preserve formatting
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        answer = answer.replace('\\', '')  # Remove all backslashes
        answer = answer.replace('\\"', '"')  # Fix escaped quotes
        answer = answer.replace('"', '"').replace('"', '"')  # Fix smart quotes
        answer = ' '.join(answer.split())  # Remove extra whitespace
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in generate_comprehensive_answer: {str(e)}")
        return "Information not found in the provided document."

def validate_and_improve_response(question: str, response: str, source_clauses: List[PolicyAnalysis]) -> str:
    """Validate and potentially improve the generated response."""
    try:
        # Convert source clauses to readable format
        source_summary = []
        for clause in source_clauses[:3]:  # Top 3 most relevant
            source_summary.append({
                "clause_id": clause.clause_id,
                "clause_type": clause.clause_type,
                "extracted_rules": clause.extracted_rules,
                "reasoning": clause.reasoning
            })
        
        source_clauses_text = json.dumps(source_summary, indent=2)
        
        prompt = VALIDATION_PROMPT.format(
            query=question,
            response=response,
            source_clauses=source_clauses_text
        )
        
        validation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,  # Reduced for cost efficiency
            temperature=0.1,
        )
        
        validation_result = validation_response.choices[0].message.content.strip()
        
        # If validation suggests improvements, use the improved version
        if validation_result != "VALIDATED" and len(validation_result) > 50:
            logger.info(f"Response improved through validation")
            return validation_result
        else:
            logger.info(f"Response validated as accurate")
            return response
            
    except Exception as e:
        logger.error(f"Error in response validation: {str(e)}")
        return response  # Return original if validation fails

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
    """Enhanced LLM answer generation with comprehensive structured analysis."""
    try:
        # Step 1: Structure the query for better processing
        structured_query = structure_query(question)
        logger.debug(f"Structured query: {structured_query}")
        
        # Step 2: Enhanced analysis approach
        document_chunks = context.split("\n\n")
        analyzed_clauses = enhanced_analyze_policy_clauses(question, document_chunks)
        
        if analyzed_clauses:
            # Step 3: Generate comprehensive answer
            answer = generate_comprehensive_answer(question, analyzed_clauses)
            
            # Step 4: Skip validation to save tokens, use direct answer
            # final_answer = validate_and_improve_response(question, answer, analyzed_clauses)
            
            logger.info(f"Generated enhanced answer with {len(analyzed_clauses)} analyzed clauses")
            return answer
        else:
            # Enhanced fallback approach with pattern-based prompting
            enhanced_fallback_prompt = f"""
You are an expert insurance policy analyst. Follow these EXACT response patterns based on question type:

CONTEXT:
{context}

QUESTION: {question}

RESPONSE PATTERNS:

1. "What is/are..." → Direct explanation with details
   Example: "What is the grace period?" → "A grace period of thirty days is provided..."

2. "Does/Is/Are..." → Start with "Yes" or "No" + explanation  
   Example: "Does this policy cover...?" → "Yes, the policy covers..." OR "No, this policy does not cover..."

3. "How..." → Direct explanation
   Example: "How does the policy define...?" → "[Term] is defined as..."

REQUIREMENTS:
✅ Follow the exact pattern for the question type
✅ Include numerical values exactly: thirty-six (36) months, two (2) years  
✅ Reference policy sections when available
✅ Include ALL conditions and limitations
✅ Use exact terminology from document
✅ NO backslashes or escape characters

SAMPLE RESPONSES:
"A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
"Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy."
"There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases."

Provide response following the pattern:
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": enhanced_fallback_prompt}],
                max_tokens=200,  # Back to original limit
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
            
            # Enhanced cleanup to fix backslash issues
            answer = answer.replace('\n', ' ').replace('\r', ' ')
            answer = answer.replace('\\', '')  # Remove all backslashes
            answer = answer.replace('\\"', '"')  # Fix escaped quotes
            answer = answer.replace('"', '"').replace('"', '"')  # Fix smart quotes
            answer = ' '.join(answer.split())  # Remove extra whitespace
            
            logger.info("Generated answer using enhanced fallback approach")
            return answer
            
    except Exception as e:
        logger.error(f"Error in generate_llm_answer: {str(e)}")
        return "Error generating answer. Please try again."

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
    docs = get_pdf_processor().process_pdf(file_path, resource_id, project_id, file.filename)
    if not docs:
        return JSONResponse(status_code=400, content={"error": "Failed to process PDF or extract text."})
    # Save to vector store
    get_pdf_processor().save_or_update_vector_store(docs, project_id)
    return {"message": f"Processed and indexed {file.filename}", "num_chunks": len(docs)}

@app.post("/pdf/search")
async def search_pdf(
    query: str = Form(...),
    project_id: str = Form(...),
    resource_ids: str = Form(None)  # comma-separated
):
    resource_id_list = resource_ids.split(",") if resource_ids else None
    results = get_pdf_processor().get_relevant_documents(query, project_id, resource_id_list)
    return {"results": [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, token: str = Depends(verify_token)):
    import time
    start_time = time.time()
    
    try:
        # Log the request body for debugging
        logger.info("=" * 50)
        logger.info("📥 RECEIVED REQUEST:")
        logger.info(f"📄 Documents URL: {request.documents}")
        logger.info(f"❓ Questions ({len(request.questions)}):")
        for i, question in enumerate(request.questions, 1):
            logger.info(f"   {i}. {question}")
        logger.info("=" * 50)
        
        # STEP 1: Download PDF from URL
        step_start = time.time()
        logger.info("🔽 STEP 1: Starting PDF download...")
        response = requests.get(request.documents, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from URL")
        logger.info(f"✅ STEP 1 COMPLETED: PDF downloaded ({len(response.content)} bytes) in {time.time() - step_start:.2f}s")

        # STEP 2: Save PDF temporarily
        step_start = time.time()
        logger.info("💾 STEP 2: Saving PDF temporarily...")
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, "temp_document.pdf")

        with open(pdf_path, "wb") as f:
            f.write(response.content)
        logger.info(f"✅ STEP 2 COMPLETED: PDF saved to {pdf_path} in {time.time() - step_start:.2f}s")

        # STEP 3: Process PDF (extract text and chunk)
        step_start = time.time()
        logger.info("📄 STEP 3: Starting PDF processing (text extraction + chunking)...")
        resource_id = "hackrx_doc"
        project_id = "hackrx_project"
        docs = get_pdf_processor().process_pdf(pdf_path, resource_id, project_id, "hackrx_document.pdf")

        if not docs:
            raise HTTPException(status_code=400, detail="Failed to process PDF or extract text")
        logger.info(f"✅ STEP 3 COMPLETED: PDF processed, created {len(docs)} document chunks in {time.time() - step_start:.2f}s")

        # STEP 4: Save to vector store (embeddings + FAISS)
        step_start = time.time()
        logger.info("🧠 STEP 4: Starting vector store creation (embeddings + FAISS indexing)...")
        get_pdf_processor().save_or_update_vector_store(docs, project_id)
        logger.info(f"✅ STEP 4 COMPLETED: Vector store created and saved in {time.time() - step_start:.2f}s")
        
        # STEP 5: Generate answers for each question using LLM
        step_start = time.time()
        logger.info("🤖 STEP 5: Starting LLM answer generation...")
        answers = []
        for i, question in enumerate(request.questions):
            q_start = time.time()
            logger.info(f"🔍 Processing question {i+1}/{len(request.questions)}: {question}")
            
            # Search for relevant documents
            results = get_pdf_processor().get_relevant_documents(question, project_id)
            if results:
                # Use top 8 chunks for better accuracy and comprehensive coverage
                context = "\n\n".join([doc.page_content for doc in results[:8]])
                answer = generate_llm_answer(question, context)
                logger.info(f"✅ Answer {i+1} generated in {time.time() - q_start:.2f}s: {answer}")
            else:
                answer = "Information not found in the provided document."
                logger.info(f"❌ Answer {i+1} (no context) in {time.time() - q_start:.2f}s: {answer}")
            answers.append(answer)
        logger.info(f"✅ STEP 5 COMPLETED: All {len(answers)} answers generated in {time.time() - step_start:.2f}s")

        # STEP 6: Clean up
        step_start = time.time()
        logger.info("🧹 STEP 6: Cleaning up temporary files...")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        logger.info(f"✅ STEP 6 COMPLETED: Cleanup finished in {time.time() - step_start:.2f}s")

        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("📤 SENDING RESPONSE:")
        logger.info(f"📝 Generated {len(answers)} answers")
        logger.info(f"⏱️ TOTAL PROCESSING TIME: {total_time:.2f} seconds")
        logger.info("=" * 50)

        return HackRxResponse(answers=answers)

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ ERROR after {total_time:.2f}s: {str(e)}")
        logger.error(f"❌ ERROR TYPE: {type(e).__name__}")
        
        # Log which step failed
        import traceback
        logger.error(f"❌ FULL TRACEBACK:\n{traceback.format_exc()}")
        
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
        docs = get_pdf_processor().process_pdf(pdf_path, resource_id, project_id, "hackrx_document.pdf")

        if not docs:
            raise HTTPException(status_code=400, detail="Failed to process PDF or extract text")

        # Save to vector store
        get_pdf_processor().save_or_update_vector_store(docs, project_id)

        # Generate detailed analysis for each question
        analyses = []
        for i, question in enumerate(request.questions):
            logger.info(f"🔍 Processing detailed analysis for question {i+1}/{len(request.questions)}: {question}")
            
            # Search for relevant documents
            results = get_pdf_processor().get_relevant_documents(question, project_id)
            
            if results:
                # Use top 8 chunks for better accuracy and comprehensive coverage
                context = "\n\n".join([doc.page_content for doc in results[:8]])
                document_chunks = context.split("\n\n")
                
                # Analyze policy clauses
                analyzed_clauses = enhanced_analyze_policy_clauses(question, document_chunks)
                logger.info(f"📋 Found {len(analyzed_clauses)} relevant clauses")
                
                # Generate decision reasoning
                decision_result = generate_decision_reasoning(question, analyzed_clauses)
                logger.info(f"🎯 Decision: {decision_result.decision} (confidence: {decision_result.confidence_score})")
                
                # Generate comprehensive answer (validation disabled to save tokens)
                answer = generate_comprehensive_answer(question, analyzed_clauses)
                # final_answer = validate_and_improve_response(question, answer, analyzed_clauses)
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

# Add startup configuration for deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 