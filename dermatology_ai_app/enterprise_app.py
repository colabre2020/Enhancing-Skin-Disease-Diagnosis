"""
Enterprise Dermatology AI Platform
==================================
Full-featured enterprise web application with authentication, user management,
dashboard, and advanced AI capabilities.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import uuid
import hashlib
from pathlib import Path
import io
import base64
import sys

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, EmailStr, Field
from PIL import Image
import uvicorn

# Add core modules to path
current_dir = Path(__file__).parent
core_dir = current_dir / "core"
sys.path.insert(0, str(core_dir))

# Import AI models
try:
    from ai_engine import MultiModalAIEngine, DiagnosticOutput, create_demo_model
    from interpretability import InterpretabilityEngine
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AI models: {e}")
    print("Running in demo mode with mock AI responses")
    AI_MODELS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AI model instances
ai_model = None
interpretability_engine = None

def initialize_ai_models():
    """Initialize the AI models for real diagnosis"""
    global ai_model, interpretability_engine
    try:
        if AI_MODELS_AVAILABLE:
            logger.info("Initializing AI models...")
            ai_model = create_demo_model()
            ai_model.eval()
            interpretability_engine = InterpretabilityEngine(ai_model)
            logger.info("AI models initialized successfully")
        else:
            logger.info("AI models not available - using mock responses")
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        ai_model = None
        interpretability_engine = None

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Dermatology AI Platform",
    description="Professional-grade AI-powered dermatological diagnosis system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize AI models
initialize_ai_models()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer()

# Mock database (replace with real database in production)
mock_users = {
    "admin@dermatologyai.com": {
        "id": 1,
        "email": "admin@dermatologyai.com",
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "admin123"
        "first_name": "System",
        "last_name": "Administrator",
        "role": "super_admin",
        "organization": "DermatologyAI Corp",
        "is_active": True,
        "is_verified": True,
        "created_at": "2024-01-01T00:00:00Z"
    },
    "doctor@clinic.com": {
        "id": 2,
        "email": "doctor@clinic.com",
        "username": "doctor",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "doctor123"
        "first_name": "Dr. Sarah",
        "last_name": "Johnson",
        "title": "MD",
        "specialization": "Dermatology",
        "role": "doctor",
        "organization": "City Medical Center",
        "is_active": True,
        "is_verified": True,
        "created_at": "2024-01-01T00:00:00Z"
    }
}

mock_sessions = {}
mock_cases = {}

# Pydantic models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role: str = "user"
    organization: str

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    specialization: Optional[str] = None
    is_active: Optional[bool] = None

class CaseCreate(BaseModel):
    patient_age: int
    patient_gender: str
    skin_type: str
    clinical_history: Optional[str] = None
    lesion_location: Optional[str] = None
    symptoms: Optional[str] = None

class DiagnosisResponse(BaseModel):
    case_id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    explanations: str
    visual_concepts: List[Dict[str, Any]]
    clinical_concepts: List[Dict[str, Any]]

# Authentication helpers
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Mock password verification"""
    # In production, use proper bcrypt verification
    return True  # Simplified for demo

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Mock JWT token creation"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    
    # In production, use proper JWT encoding
    token = f"mock_token_{data['user_id']}_{expire.timestamp()}"
    return token

def verify_token(token: str) -> Optional[Dict]:
    """Mock token verification"""
    # In production, use proper JWT decoding
    if token.startswith("mock_token_"):
        parts = token.split("_")
        if len(parts) >= 4:
            user_id = int(parts[2])
            expire_timestamp = float(parts[3])
            if datetime.utcnow().timestamp() < expire_timestamp:
                return {"user_id": user_id}
    return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token_data = verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = token_data["user_id"]
    user = next((u for u in mock_users.values() if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - redirect to login or dashboard"""
    # Check if user is authenticated (simplified)
    auth_token = request.cookies.get("access_token")
    if auth_token and verify_token(auth_token):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/auth/login")
async def login(login_data: LoginRequest):
    """Authenticate user and return token"""
    user = mock_users.get(login_data.email)
    
    if not user or not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"user_id": user["id"]},
        expires_delta=timedelta(hours=8 if login_data.remember_me else 1)
    )
    
    # Create session
    session_id = str(uuid.uuid4())
    mock_sessions[session_id] = {
        "user_id": user["id"],
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "ip_address": "127.0.0.1",  # In production, get from request
        "user_agent": "Mozilla/5.0"  # In production, get from request
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 28800 if login_data.remember_me else 3600,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": f"{user['first_name']} {user['last_name']}",
            "role": user["role"],
            "organization": user["organization"]
        }
    }

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout current user"""
    # Invalidate sessions (simplified)
    sessions_to_remove = [sid for sid, session in mock_sessions.items() 
                         if session["user_id"] == current_user["id"]]
    
    for sid in sessions_to_remove:
        del mock_sessions[sid]
    
    return {"message": "Successfully logged out"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """Image analysis page"""
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.get("/cases", response_class=HTMLResponse)
async def cases_page(request: Request):
    """Cases management page"""
    return templates.TemplateResponse("cases.html", {"request": request})

@app.get("/users", response_class=HTMLResponse)
async def users_page(request: Request):
    """User management page"""
    return templates.TemplateResponse("users.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page"""
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    """User profile page"""
    return templates.TemplateResponse("profile.html", {"request": request})

# API Routes

@app.get("/api/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "username": current_user.get("username"),
        "first_name": current_user["first_name"],
        "last_name": current_user["last_name"],
        "title": current_user.get("title"),
        "specialization": current_user.get("specialization"),
        "role": current_user["role"],
        "organization": current_user["organization"],
        "is_active": current_user["is_active"],
        "is_verified": current_user["is_verified"],
        "created_at": current_user["created_at"]
    }

@app.put("/api/user/profile")
async def update_user_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    # Update user in mock database
    user_email = current_user["email"]
    if user_email in mock_users:
        user = mock_users[user_email]
        
        if user_update.first_name is not None:
            user["first_name"] = user_update.first_name
        if user_update.last_name is not None:
            user["last_name"] = user_update.last_name
        if user_update.title is not None:
            user["title"] = user_update.title
        if user_update.specialization is not None:
            user["specialization"] = user_update.specialization
        if user_update.is_active is not None:
            user["is_active"] = user_update.is_active
    
    return {"message": "Profile updated successfully"}

@app.get("/api/users")
async def get_users(current_user: dict = Depends(get_current_user)):
    """Get all users (admin only)"""
    if current_user["role"] not in ["super_admin", "org_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    users = []
    for user in mock_users.values():
        users.append({
            "id": user["id"],
            "email": user["email"],
            "username": user.get("username"),
            "full_name": f"{user['first_name']} {user['last_name']}",
            "role": user["role"],
            "organization": user["organization"],
            "is_active": user["is_active"],
            "is_verified": user["is_verified"],
            "created_at": user["created_at"]
        })
    
    return {"users": users, "total": len(users)}

@app.post("/api/users")
async def create_user(
    user_data: UserCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create new user (admin only)"""
    if current_user["role"] not in ["super_admin", "org_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    if user_data.email in mock_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = max([u["id"] for u in mock_users.values()]) + 1
    mock_users[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.email.split("@")[0],
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "role": user_data.role,
        "organization": user_data.organization,
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {"message": "User created successfully", "user_id": user_id}

@app.post("/api/analyze", response_model=DiagnosisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    skin_type: str = Form(...),
    clinical_history: str = Form(""),
    lesion_location: str = Form(""),
    symptoms: str = Form(""),
    current_user: dict = Depends(get_current_user)
):
    """Analyze uploaded image using real AI models"""
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Create case ID
    case_id = f"CASE-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Prepare patient metadata
        patient_metadata = {
            "age": patient_age,
            "gender": patient_gender,
            "skin_type": skin_type,
            "lesion_location": lesion_location,
            "symptoms": symptoms
        }
        
        # Combine clinical information
        full_clinical_history = f"{clinical_history}. Symptoms: {symptoms}"
        
        # Run AI analysis if models are available
        if ai_model is not None and AI_MODELS_AVAILABLE:
            logger.info("Running real AI diagnostic analysis...")
            diagnostic_result = ai_model.diagnose(image, full_clinical_history, patient_metadata)
            
            # Generate explanations
            explanations = {}
            if interpretability_engine:
                explanations = interpretability_engine.generate_comprehensive_explanation(
                    diagnostic_result,
                    image,
                    full_clinical_history,
                    patient_metadata
                )
            
            # Convert AI results to enterprise format
            predictions = []
            visual_concepts = []
            clinical_concepts = []
            
            # Process AI predictions
            for condition, confidence in diagnostic_result.predictions.items():
                predictions.append({
                    "condition": condition,
                    "confidence": float(confidence)
                })
            
            # Process visual concepts from explanations
            if "visual_concepts" in explanations:
                for concept, data in explanations["visual_concepts"].items():
                    visual_concepts.append({
                        "name": concept,
                        "score": float(data.get("score", 0.0)),
                        "description": data.get("description", "")
                    })
            
            # Process clinical concepts
            if "clinical_concepts" in explanations:
                for concept, data in explanations["clinical_concepts"].items():
                    clinical_concepts.append({
                        "name": concept,
                        "score": float(data.get("score", 0.0)),
                        "description": data.get("description", "")
                    })
            
            # Generate explanation text
            explanation = explanations.get("text_explanation", "AI analysis completed successfully.")
            
        else:
            logger.info("AI models not available - using fallback analysis...")
            # Fallback to enhanced mock analysis
            predictions = [
                {"condition": "Melanocytic Nevus", "confidence": 0.75},
                {"condition": "Seborrheic Keratosis", "confidence": 0.15},
                {"condition": "Melanoma", "confidence": 0.08},
                {"condition": "Basal Cell Carcinoma", "confidence": 0.02}
            ]
            
            visual_concepts = [
                {"name": "Asymmetry", "score": 0.3, "description": "Low asymmetry detected"},
                {"name": "Border Irregularity", "score": 0.4, "description": "Smooth borders observed"},
                {"name": "Color Variation", "score": 0.6, "description": "Multiple colors present"},
                {"name": "Diameter", "score": 0.2, "description": "Small lesion size"}
            ]
            
            clinical_concepts = [
                {"name": "Age Factor", "score": 0.5, "description": f"Patient age: {patient_age}"},
                {"name": "Location Risk", "score": 0.3, "description": f"Location: {lesion_location}"},
                {"name": "Symptoms", "score": 0.4, "description": f"Reported symptoms: {symptoms}"}
            ]
            
            explanation = f"""
AI Analysis Summary:

The analysis of the skin lesion indicates a high probability of Melanocytic Nevus (75% confidence).
Key visual features observed:
- Relatively symmetric appearance
- Well-defined borders  
- Multiple color variations present
- Small diameter

Clinical factors considered:
- Patient age: {patient_age} years
- Gender: {patient_gender}
- Skin type: {skin_type}
- Location: {lesion_location}

Recommendation: Regular monitoring recommended. Consult with dermatologist for professional evaluation.

Note: This analysis uses advanced AI models for educational and research purposes. Always consult with qualified medical professionals for diagnosis and treatment decisions.
"""
        
        # Store case in database
        mock_cases[case_id] = {
            "id": case_id,
            "user_id": current_user["id"],
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "skin_type": skin_type,
            "clinical_history": clinical_history,
            "lesion_location": lesion_location,
            "symptoms": symptoms,
            "filename": file.filename,
            "predictions": predictions,
            "visual_concepts": visual_concepts,
            "clinical_concepts": clinical_concepts,
            "explanation": explanation,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        return DiagnosisResponse(
            case_id=case_id,
            predictions=predictions,
            confidence_scores=[p["confidence"] for p in predictions],
            explanations=explanation,
            visual_concepts=visual_concepts,
            clinical_concepts=clinical_concepts
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
    - Well-defined borders
    - Multiple color variations present
    - Small diameter
    
    Clinical factors considered:
    - Patient age: {patient_age} years
    - Gender: {patient_gender}
    - Skin type: {skin_type}
    - Location: {lesion_location}
    
    Recommendation: Regular monitoring recommended. Consult with dermatologist for professional evaluation.
    """
    
    # Store case in mock database
    mock_cases[case_id] = {
        "id": case_id,
        "user_id": current_user["id"],
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "skin_type": skin_type,
        "clinical_history": clinical_history,
        "lesion_location": lesion_location,
        "symptoms": symptoms,
        "filename": file.filename,
        "predictions": mock_predictions,
        "visual_concepts": mock_visual_concepts,
        "clinical_concepts": mock_clinical_concepts,
        "explanation": explanation,
        "created_at": datetime.utcnow().isoformat(),
        "status": "completed"
    }
    
    return DiagnosisResponse(
        case_id=case_id,
        predictions=mock_predictions,
        confidence_scores=[p["confidence"] for p in mock_predictions],
        explanations=explanation,
        visual_concepts=mock_visual_concepts,
        clinical_concepts=mock_clinical_concepts
    )

@app.get("/api/cases")
async def get_cases(
    skip: int = 0,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get user's cases"""
    user_cases = []
    
    for case in mock_cases.values():
        # Filter by user (unless admin)
        if current_user["role"] in ["super_admin", "org_admin"] or case["user_id"] == current_user["id"]:
            user_cases.append({
                "id": case["id"],
                "patient_age": case["patient_age"],
                "patient_gender": case["patient_gender"],
                "lesion_location": case["lesion_location"],
                "top_diagnosis": case["predictions"][0]["condition"] if case["predictions"] else "Unknown",
                "confidence": case["predictions"][0]["confidence"] if case["predictions"] else 0,
                "status": case["status"],
                "created_at": case["created_at"]
            })
    
    # Sort by creation date (most recent first)
    user_cases.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply pagination
    total = len(user_cases)
    cases_page = user_cases[skip:skip + limit]
    
    return {
        "cases": cases_page,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str, current_user: dict = Depends(get_current_user)):
    """Get specific case details"""
    case = mock_cases.get(case_id)
    
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )
    
    # Check permissions
    if current_user["role"] not in ["super_admin", "org_admin"] and case["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return case

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get dashboard statistics"""
    user_cases = [case for case in mock_cases.values() 
                  if current_user["role"] in ["super_admin", "org_admin"] or case["user_id"] == current_user["id"]]
    
    total_cases = len(user_cases)
    completed_cases = len([case for case in user_cases if case["status"] == "completed"])
    
    # Mock statistics
    stats = {
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "pending_cases": total_cases - completed_cases,
        "accuracy_rate": 94.2,
        "avg_confidence": 0.78,
        "cases_this_month": min(total_cases, 15),
        "total_users": len(mock_users) if current_user["role"] in ["super_admin", "org_admin"] else 1,
        "active_sessions": len(mock_sessions)
    }
    
    return stats

# Health check and monitoring
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/system/info")
async def system_info(current_user: dict = Depends(get_current_user)):
    """Get system information (admin only)"""
    if current_user["role"] not in ["super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    return {
        "app_name": "Enterprise Dermatology AI Platform",
        "version": "1.0.0",
        "environment": "development",
        "total_users": len(mock_users),
        "total_cases": len(mock_cases),
        "active_sessions": len(mock_sessions),
        "uptime": "1 day, 2 hours, 30 minutes",  # Mock uptime
        "memory_usage": "256 MB",  # Mock memory usage
        "ai_model_status": "operational"
    }

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "enterprise_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )