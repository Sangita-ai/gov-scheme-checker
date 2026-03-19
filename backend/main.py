"""
Government Scheme Eligibility Checker — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

import os
import sys
import json
import uuid
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv


sys.path.insert(0, os.path.dirname(__file__))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import database
import agents
from vector_store import get_vector_store_count

SCHEME_CACHE: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load scheme cache at startup."""
    global SCHEME_CACHE
    print("🚀 Starting Government Scheme Eligibility Checker API...")

    
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("⚠️  GROQ_API_KEY not set. Chat features will be limited.")
    else:
        print("✅ Groq API key found.")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("⚠️  GEMINI_API_KEY not set. Translation fallback unavailable.")
    else:
        print("✅ Gemini API key found.")

    
    SCHEME_CACHE = database.get_all_schemes()
    print(f"✅ Loaded {len(SCHEME_CACHE)} schemes from database.")

    if len(SCHEME_CACHE) == 0:
        print("⚠️  No schemes found! Please run: python seed_db.py")

    vs_count = get_vector_store_count()
    print(f"✅ Vector store has {vs_count} embeddings.")

    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Gov Scheme Eligibility Checker",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "null",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


request_counts: dict = defaultdict(list)
RATE_LIMIT = 30  


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    window = 60  
    request_counts[ip] = [t for t in request_counts[ip] if now - t < window]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return True
    request_counts[ip].append(now)
    return False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language: str = Field(default="en", pattern="^(en|hi|bn)$")


class ChatResponse(BaseModel):
    reply: str
    profile_complete: int
    schemes_found: int
    session_id: str
    profile: dict


class UserProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    state: Optional[str] = None
    occupation: Optional[str] = None
    income_annual_inr: Optional[float] = None
    caste_category: Optional[str] = None
    is_bpl: Optional[bool] = None
    has_bank_account: Optional[bool] = None
    has_aadhar: Optional[bool] = None
    is_farmer: Optional[bool] = None
    land_owned_acres: Optional[float] = None
    number_of_children: Optional[int] = None
    has_disability: Optional[bool] = None


class EligibilityRequest(BaseModel):
    profile: dict


class FeedbackRequest(BaseModel):
    scheme_id: str
    helpful: bool
    comment: Optional[str] = ""

@app.get("/")
def root():
    """Root endpoint — confirms API is running. Fixes 404 on browser hits."""
    return {
        "message": "Gov Scheme Checker API is running!",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    """Health check — shows current state of all services."""
    return {
        "status": "ok",
        "version": "1.0",
        "schemes_loaded": len(SCHEME_CACHE),
        "vector_store_ready": get_vector_store_count() > 0,
        "vector_store_count": get_vector_store_count(),
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Main conversational endpoint — collects profile and returns AI response."""

    
    client_ip = request.client.host if request.client else "unknown"
    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment."
        )

    
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        return ChatResponse(
            reply=(
                "⚠️ No API key configured. Please add your GROQ_API_KEY "
                "to the .env file. Get a free key at https://console.groq.com"
            ),
            profile_complete=0,
            schemes_found=0,
            session_id=body.session_id or str(uuid.uuid4()),
            profile={},
        )

    
    if len(SCHEME_CACHE) == 0:
        return ChatResponse(
            reply="⚠️ Database not ready. Please run: python seed_db.py",
            profile_complete=0,
            schemes_found=0,
            session_id=body.session_id or str(uuid.uuid4()),
            profile={},
        )

    
    session_id = body.session_id or str(uuid.uuid4())
    session = database.get_session(session_id)
    if not session:
        session = database.create_session(session_id, body.language)

    profile = session.get("profile_json") or {}
    history = session.get("chat_history") or []

    
    history.append({"role": "user", "content": body.message})

    
    try:
        result = agents.parse_input(
            user_message=body.message,
            current_profile=profile,
            chat_history=history[-10:],  
            language=body.language,
        )
        updated_profile = result.get("updated_profile", profile)
        next_question = result.get("next_question", "Could you tell me more?")
        pct = result.get("profile_complete_pct", 0)
    except Exception as e:
        print(f"❌ Agent error: {e}")
        updated_profile = profile
        next_question = "I had trouble understanding that. Could you rephrase?"
        pct = 0

    
    schemes_found = 0
    if pct >= 70:
        try:
            eligible_count = len([
                r for r in agents.check_eligibility(updated_profile, SCHEME_CACHE)
                if r["status"] == "ELIGIBLE"
            ])
            schemes_found = eligible_count

            
            if body.language == "hi":
                next_question += (
                    f"\n\n🎉 आपकी प्रोफ़ाइल {pct}% पूरी हो गई है! "
                    f"मुझे {eligible_count} योजनाएं मिली हैं जिनके लिए आप पात्र हैं। "
                    f"'परिणाम देखें' बटन दबाएं।"
                )
            elif body.language == "bn":
                next_question += (
                    f"\n\n🎉 আপনার প্রোফাইল {pct}% সম্পূর্ণ হয়েছে! "
                    f"আমি {eligible_count}টি প্রকল্প পেয়েছি যার জন্য আপনি যোগ্য। "
                    f"'ফলাফল দেখুন' বোতামটি চাপুন।"
                )
            else:
                next_question += (
                    f"\n\n🎉 Your profile is {pct}% complete! "
                    f"I found **{eligible_count} scheme(s)** you may be eligible for. "
                    f"Click **'View Results →'** to see them!"
                )
        except Exception as e:
            print(f"❌ Eligibility count error: {e}")

    
    history.append({"role": "assistant", "content": next_question})

    
    database.save_session(session_id, updated_profile, history[-30:], body.language)

    return ChatResponse(
        reply=next_question,
        profile_complete=pct,
        schemes_found=schemes_found,
        session_id=session_id,
        profile=updated_profile,
    )


@app.post("/api/check-eligibility")
async def check_eligibility(body: EligibilityRequest):
    """Run full rule-based eligibility check against all 25 schemes."""
    if len(SCHEME_CACHE) == 0:
        raise HTTPException(
            status_code=503,
            detail="Database not ready. Please run: python seed_db.py"
        )

    if not body.profile:
        raise HTTPException(status_code=400, detail="Profile is empty")

    results = agents.check_eligibility(body.profile, SCHEME_CACHE)
    eligible = [r for r in results if r["status"] == "ELIGIBLE"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    not_eligible = [r for r in results if r["status"] == "NOT_ELIGIBLE"]

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "eligible": len(eligible),
            "partial": len(partial),
            "not_eligible": len(not_eligible),
        },
    }


@app.get("/api/scheme/{scheme_id}")
async def get_scheme(
    scheme_id: str,
    language: str = "en",
    session_id: Optional[str] = None,
):
    """Get a single scheme's details and generate a personalized application guide."""
    scheme = database.get_scheme_by_id(scheme_id)
    if not scheme:
        raise HTTPException(
            status_code=404,
            detail=f"Scheme '{scheme_id}' not found. Check /api/schemes for valid IDs."
        )

    
    profile = {}
    if session_id:
        session = database.get_session(session_id)
        if session:
            profile = session.get("profile_json") or {}

    
    guide = agents.generate_guide(scheme, profile, language)

    
    docs = scheme.get("documents", "[]")
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except Exception:
            docs = []
    scheme["documents"] = docs

    return {"scheme": scheme, "guide": guide}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Return session profile and chat history for a given session ID."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "profile": session.get("profile_json", {}),
        "history": session.get("chat_history", []),
        "language": session.get("language", "en"),
    }


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Save thumbs-up/down feedback for a scheme."""
    database.save_feedback(body.scheme_id, body.helpful, body.comment or "")
    return {"success": True}


@app.get("/api/schemes")
async def list_schemes():
    """Return a summary list of all schemes (used by frontend landing page)."""
    if len(SCHEME_CACHE) == 0:
        return {"schemes": [], "count": 0}

    summary = [
        {
            "id": s["id"],
            "name_en": s["name_en"],
            "ministry": s["ministry"],
            "benefit_type": s["benefit_type"],
            "benefit_summary": s["benefit_summary"],
            "scheme_type": s["scheme_type"],
            "state": s["state"],
        }
        for s in SCHEME_CACHE
    ]
    return {"schemes": summary, "count": len(summary)}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )