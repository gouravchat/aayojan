"""
AayojanAI Backend — FastAPI + Gemini SDK
Serves as the AI brain for both the React landing page and Flutter app.
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from gemini_client import GeminiClient

load_dotenv()


# ─── Rate Limiter ─────────────────────────────────────────────────────────────
class RateLimiter:
    """Simple in-memory rate limiter per IP."""
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Clean old entries
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
        if len(self.requests[ip]) >= self.max_requests:
            return False
        self.requests[ip].append(now)
        return True

rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
chat_limiter = RateLimiter(max_requests=10, window_seconds=60)  # Stricter for AI endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Gemini client on startup."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set — running in echo mode")
    app.state.gemini = GeminiClient(api_key=api_key, model=model)
    yield


app = FastAPI(
    title="AayojanAI API",
    description="Gemini-powered catering chatbot backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restricted to known origins only
origins = os.getenv("CORS_ORIGINS", "https://aayojan.online,https://www.aayojan.online,https://aayojanv1.github.io").split(",")
if os.getenv("ENV", "production") == "development":
    origins.append("http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ─── Security Middleware ──────────────────────────────────────────────────────
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Rate limiting
    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()
    
    # Stricter limit for AI endpoints
    if request.url.path.startswith("/api/chat") or request.url.path.startswith("/api/menu") or request.url.path.startswith("/api/price"):
        if not chat_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please wait a minute."})
    else:
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})
    
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# ─── Models ──────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if len(v) > 5000:
            raise ValueError("message too long (max 5000 chars)")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if len(v) > 50:
            raise ValueError("too many messages (max 50)")
        if len(v) == 0:
            raise ValueError("at least one message required")
        return v


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


class MenuRequest(BaseModel):
    party_type: str
    guest_count: int = 100
    pincode: str = "700156"


class MenuResponse(BaseModel):
    party_type: str
    description: str
    items: list[dict]


class PriceRequest(BaseModel):
    party_type: str
    guest_count: int
    pincode: str
    selected_items: list[dict]


class PriceResponse(BaseModel):
    per_plate_estimate: int
    summary: str
    pricing_reason: str


class CatererRankRequest(BaseModel):
    party_type: str
    guest_count: int
    pincode: str
    per_plate_budget: int
    selected_items: list[str]
    candidates: list[dict]


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversational endpoint — used by both React and Flutter."""
    gemini: GeminiClient = app.state.gemini
    try:
        messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
        reply = await gemini.chat(
            messages=messages_dicts,
            system_prompt=req.system_prompt,
        )
        return ChatResponse(reply=reply, session_id=req.session_id)
    except Exception as e:
        error_msg = str(e)
        print(f"[WARN] Chat error: {error_msg[:200]}", flush=True)
        # Return a friendly fallback instead of 500
        if "quota" in error_msg.lower() or "429" in error_msg or "resource" in error_msg.lower():
            fallback = "I'm getting a lot of requests right now! The AI quota is temporarily exhausted (free tier: 20 requests/day). Please try again in a few minutes or tomorrow. Meanwhile, you can browse caterers using the service cards below!"
        else:
            fallback = f"Something went wrong connecting to the AI. Please try again. (Error: {error_msg[:100]})"
        return ChatResponse(reply=fallback, session_id=req.session_id)


@app.post("/api/menu/generate", response_model=MenuResponse)
async def generate_menu(req: MenuRequest):
    """Generate a curated menu for a party type."""
    gemini: GeminiClient = app.state.gemini
    try:
        result = await gemini.generate_menu(
            party_type=req.party_type,
            guest_count=req.guest_count,
            pincode=req.pincode,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/price/estimate", response_model=PriceResponse)
async def estimate_price(req: PriceRequest):
    """Estimate fair per-plate pricing for selected menu."""
    gemini: GeminiClient = app.state.gemini
    try:
        result = await gemini.estimate_price(
            party_type=req.party_type,
            guest_count=req.guest_count,
            pincode=req.pincode,
            selected_items=req.selected_items,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caterers/rank")
async def rank_caterers(req: CatererRankRequest):
    """Rank caterers using AI based on party context."""
    gemini: GeminiClient = app.state.gemini
    try:
        result = await gemini.rank_caterers(
            party_type=req.party_type,
            guest_count=req.guest_count,
            pincode=req.pincode,
            per_plate_budget=req.per_plate_budget,
            selected_items=req.selected_items,
            candidates=req.candidates,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
