"""
Web Application Interface
========================

FastAPI-based web application for the AI dermatological diagnosis system.
Implements the clinical integration module described in the research paper.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import sys
from pathlib import Path
import json
import base64
import io
from PIL import Image
from typing import Dict, Optional, List
import logging

# Add core modules to path
current_dir = Path(__file__).parent
core_dir = current_dir / "core"
sys.path.insert(0, str(core_dir))

try:
    from ai_engine import MultiModalAIEngine, DiagnosticOutput, create_demo_model
    from interpretability import InterpretabilityEngine
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Running in demo mode with limited functionality")

try:
    from .lightweight_inference import LightweightDermEngine
except ImportError:
    try:
        from lightweight_inference import LightweightDermEngine
    except ImportError as e:
        print(f"Warning: Could not import lightweight inference module: {e}")
        LightweightDermEngine = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Dermatological Diagnosis System",
    description="Interpretable AI system for skin disease detection and classification",
    version="1.0.0"
)

# Global model instance (in production, this would be loaded from a saved checkpoint)
model = None
interpretability_engine = None
lightweight_engine = None

def initialize_models():
    """Initialize the AI models"""
    global model, interpretability_engine, lightweight_engine
    try:
        logger.info("Initializing AI models...")
        model = create_demo_model()
        model.eval()
        interpretability_engine = InterpretabilityEngine(model)
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Create fallback objects for serverless/demo usage
        model = None
        interpretability_engine = None

    if model is None and LightweightDermEngine is not None:
        try:
            lightweight_engine = LightweightDermEngine()
            logger.info("Lightweight inference engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize lightweight inference engine: {e}")
            lightweight_engine = None

# Initialize models on startup
initialize_models()

# Create directories
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page describing the system"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/analyze")
async def analyze_skin_lesion(
    file: UploadFile = File(...),
    clinical_history: str = Form(""),
    patient_age: int = Form(50),
    patient_gender: str = Form("unknown"),
    skin_type: str = Form("type_III"),
    lesion_location: str = Form("unknown"),
    symptoms: str = Form("")
):
    """
    Main endpoint for skin lesion analysis
    Implements the diagnostic workflow from the research paper
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
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
        
        if model is None and lightweight_engine is not None:
            lightweight_result = lightweight_engine.diagnose(
                image,
                full_clinical_history,
                patient_metadata
            )
            return create_lightweight_response(
                image,
                full_clinical_history,
                patient_metadata,
                lightweight_result
            )

        if model is None:
            # Return demo response if model not available
            return create_demo_response(image, full_clinical_history, patient_metadata)
        
        # Run diagnosis
        logger.info("Running diagnostic analysis...")
        diagnostic_result = model.diagnose(image, full_clinical_history, patient_metadata)
        
        # Generate explanations
        explanations = {}
        if interpretability_engine:
            explanations = interpretability_engine.generate_comprehensive_explanation(
                diagnostic_result,
                image,
                full_clinical_history,
                patient_metadata
            )
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Convert NumPy types to Python native types"""
            import numpy as np
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Prepare response with converted types
        response = {
            "success": True,
            "image_data": f"data:image/png;base64,{image_base64}",
            "predictions": convert_numpy_types(diagnostic_result.predictions),
            "confidence": convert_numpy_types(diagnostic_result.confidence_scores),
            "explanation": explanations.get("text_explanation", "No explanation available"),
            "visual_concepts": convert_numpy_types(explanations.get("visual_concepts", {})),
            "clinical_concepts": convert_numpy_types(explanations.get("clinical_concepts", {})),
            "patient_info": patient_metadata,
            "clinical_history": full_clinical_history
        }
        
        # Add attention visualization if available
        if "attention_visualization" in explanations:
            # Convert attention visualization to base64
            att_buffered = io.BytesIO()
            explanations["attention_visualization"].save(att_buffered, format="PNG")
            att_base64 = base64.b64encode(att_buffered.getvalue()).decode()
            response["attention_visualization"] = f"data:image/png;base64,{att_base64}"
        
        logger.info("Analysis completed successfully")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


def create_demo_response(image: Image.Image, clinical_history: str, patient_metadata: Dict) -> JSONResponse:
    """Create a demo response when the actual model is not available"""
    import random
    
    # Demo predictions
    diseases = [
        "melanoma", "melanocytic_nevus", "basal_cell_carcinoma",
        "actinic_keratosis", "benign_keratosis", "dermatofibroma", "vascular_lesion"
    ]
    
    # Generate random but realistic predictions
    predictions = {}
    remaining_prob = 1.0
    for i, disease in enumerate(diseases[:-1]):
        if i == 0:  # Make one disease dominant
            prob = random.uniform(0.4, 0.8)
        else:
            prob = random.uniform(0.01, remaining_prob * 0.3)
        predictions[disease] = prob
        remaining_prob -= prob
    
    predictions[diseases[-1]] = max(remaining_prob, 0.01)
    
    # Normalize to ensure sum = 1
    total = sum(predictions.values())
    predictions = {k: v/total for k, v in predictions.items()}
    
    # Demo explanation
    top_disease = max(predictions.keys(), key=lambda k: predictions[k])
    explanation = f"""**DEMO MODE - Diagnostic Assessment:**

Primary diagnosis: **{top_disease.replace('_', ' ').title()}** (confidence: {predictions[top_disease]:.1%})

This is a demonstration of the AI diagnostic system. The results shown are for illustration purposes only and should not be used for actual medical diagnosis.

**Key Features Analyzed:**
- Visual pattern recognition using multi-modal AI
- Integration of clinical history and patient metadata
- Interpretable explanations with concept attribution
- Confidence scoring and uncertainty quantification

**Recommendations:**
- This is a research prototype - consult a dermatologist for actual diagnosis
- The system demonstrates the framework described in the research paper
- Real clinical deployment would require extensive validation and regulatory approval
"""
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse(content={
        "success": True,
        "demo_mode": True,
        "image_data": f"data:image/png;base64,{image_base64}",
        "predictions": predictions,
        "confidence": {"overall_confidence": random.uniform(0.6, 0.9)},
        "explanation": explanation,
        "visual_concepts": {
            "asymmetry": random.uniform(0.2, 0.8),
            "border_irregularity": random.uniform(0.1, 0.7),
            "color_variation": random.uniform(0.3, 0.9),
            "diameter": random.uniform(0.2, 0.6)
        },
        "clinical_concepts": {
            "patient_age": min(patient_metadata["age"] / 80.0, 1.0),
            "lesion_location": random.uniform(0.3, 0.7),
            "skin_type": random.uniform(0.2, 0.8)
        },
        "patient_info": patient_metadata,
        "clinical_history": clinical_history
    })


def create_lightweight_response(
    image: Image.Image,
    clinical_history: str,
    patient_metadata: Dict,
    lightweight_result
) -> JSONResponse:
    """Create a deterministic lightweight response for serverless environments."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={
        "success": True,
        "lightweight_mode": True,
        "image_data": f"data:image/png;base64,{image_base64}",
        "predictions": lightweight_result.predictions,
        "confidence": lightweight_result.confidence_scores,
        "explanation": lightweight_result.explanation,
        "visual_concepts": lightweight_result.visual_concepts,
        "clinical_concepts": lightweight_result.clinical_concepts,
        "patient_info": patient_metadata,
        "clinical_history": clinical_history
    })


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    mode = "full" if model is not None else ("lightweight" if lightweight_engine is not None else "demo")
    return {
        "status": "healthy",
        "inference_mode": mode,
        "model_loaded": model is not None,
        "interpretability_available": interpretability_engine is not None,
        "lightweight_available": lightweight_engine is not None
    }


@app.get("/api/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None and lightweight_engine is not None:
        return {
            "model_type": "LightweightDermEngine",
            "num_classes": len(lightweight_engine.disease_classes),
            "disease_classes": lightweight_engine.disease_classes,
            "version": "1.0.0-lightweight",
            "note": "Serverless heuristic mode for Vercel"
        }

    if model is None:
        return {"error": "Model not loaded", "demo_mode": True}
    
    return {
        "model_type": "MultiModalAIEngine",
        "num_classes": model.num_classes,
        "disease_classes": model.disease_classes,
        "version": "1.0.0"
    }


@app.post("/api/batch_analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Batch analysis endpoint for processing multiple images
    Useful for research and validation studies
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    for file in files:
        try:
            # Process each file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Use default metadata for batch processing
            default_metadata = {
                "age": 50,
                "gender": "unknown",
                "skin_type": "type_III",
                "lesion_location": "unknown"
            }
            
            if model:
                result = model.diagnose(image, "Batch processing", default_metadata)
                results.append({
                    "filename": file.filename,
                    "predictions": result.predictions,
                    "confidence": result.confidence_scores,
                    "top_prediction": max(result.predictions.keys(), key=lambda k: result.predictions[k])
                })
            elif lightweight_engine:
                result = lightweight_engine.diagnose(image, "Batch processing", default_metadata)
                results.append({
                    "filename": file.filename,
                    "lightweight_mode": True,
                    "predictions": result.predictions,
                    "confidence": result.confidence_scores,
                    "top_prediction": max(result.predictions.keys(), key=lambda k: result.predictions[k])
                })
            else:
                # Demo mode
                results.append({
                    "filename": file.filename,
                    "demo_mode": True,
                    "message": "Demo analysis completed"
                })
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_processed": len(results)}


# Create HTML templates
def create_templates():
    """Create HTML templates for the web interface"""
    
    # Index template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Dermatological Diagnosis System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #667eea;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .prediction-bar {
            height: 20px;
            background: #667eea;
            border-radius: 10px;
            transition: width 0.5s;
        }
        .explanation {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
            white-space: pre-line;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .image-preview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            display: block;
            border-radius: 5px;
        }
        .concepts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .concept-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI-Powered Dermatological Diagnosis System</h1>
        <p>Interpretable Visual Concept Discovery for Skin Disease Detection</p>
    </div>

    <div class="container">
        <h2>Upload Skin Lesion Image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('file').click()">
                <p>Click here to select an image or drag and drop</p>
                <input type="file" id="file" name="file" accept="image/*" style="display: none;" onchange="previewImage(this)">
            </div>
            <img id="imagePreview" class="image-preview" style="display: none;">

            <div class="form-group">
                <label for="clinical_history">Clinical History:</label>
                <textarea id="clinical_history" name="clinical_history" rows="3" 
                         placeholder="Describe the lesion's appearance, duration, and any changes observed..."></textarea>
            </div>

            <div class="form-group">
                <label for="patient_age">Patient Age:</label>
                <input type="number" id="patient_age" name="patient_age" value="50" min="0" max="120">
            </div>

            <div class="form-group">
                <label for="patient_gender">Gender:</label>
                <select id="patient_gender" name="patient_gender">
                    <option value="unknown">Unknown</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                </select>
            </div>

            <div class="form-group">
                <label for="skin_type">Skin Type:</label>
                <select id="skin_type" name="skin_type">
                    <option value="type_I">Type I (Always burns, never tans)</option>
                    <option value="type_II">Type II (Usually burns, tans minimally)</option>
                    <option value="type_III" selected>Type III (Sometimes burns, tans gradually)</option>
                    <option value="type_IV">Type IV (Burns minimally, always tans)</option>
                    <option value="type_V">Type V (Rarely burns, tans darkly)</option>
                    <option value="type_VI">Type VI (Never burns, deeply pigmented)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="lesion_location">Lesion Location:</label>
                <input type="text" id="lesion_location" name="lesion_location" 
                       placeholder="e.g., back, face, arm, leg...">
            </div>

            <div class="form-group">
                <label for="symptoms">Symptoms:</label>
                <input type="text" id="symptoms" name="symptoms" 
                       placeholder="e.g., itching, bleeding, pain, growth...">
            </div>

            <button type="submit" class="btn">Analyze Lesion</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image... This may take a few moments.</p>
        </div>

        <div class="results" id="results">
            <h2>Diagnostic Results</h2>
            <div id="predictions"></div>
            <div id="explanation" class="explanation"></div>
            
            <div class="concepts">
                <div>
                    <h3>Visual Concepts</h3>
                    <div id="visualConcepts"></div>
                </div>
                <div>
                    <h3>Clinical Risk Factors</h3>
                    <div id="clinicalConcepts"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function previewImage(input) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('imagePreview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            // Show loading, hide results
            loading.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Analysis failed: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        });

        function displayResults(data) {
            const results = document.getElementById('results');
            const predictions = document.getElementById('predictions');
            const explanation = document.getElementById('explanation');
            const visualConcepts = document.getElementById('visualConcepts');
            const clinicalConcepts = document.getElementById('clinicalConcepts');
            
            // Display predictions
            predictions.innerHTML = '';
            const sortedPredictions = Object.entries(data.predictions)
                .sort(([,a], [,b]) => b - a);
            
            sortedPredictions.forEach(([disease, confidence]) => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                item.innerHTML = `
                    <span>${disease.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</span>
                    <div style="width: 200px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                        <div class="prediction-bar" style="width: ${confidence * 100}%"></div>
                    </div>
                    <span>${(confidence * 100).toFixed(1)}%</span>
                `;
                predictions.appendChild(item);
            });
            
            // Display explanation
            explanation.textContent = data.explanation || 'No explanation available';
            
            // Display visual concepts
            visualConcepts.innerHTML = '';
            if (data.visual_concepts) {
                Object.entries(data.visual_concepts).forEach(([concept, score]) => {
                    const item = document.createElement('div');
                    item.className = 'concept-item';
                    item.innerHTML = `
                        <span>${concept.replace(/_/g, ' ')}</span>
                        <span>${(score * 100).toFixed(0)}%</span>
                    `;
                    visualConcepts.appendChild(item);
                });
            }
            
            // Display clinical concepts
            clinicalConcepts.innerHTML = '';
            if (data.clinical_concepts) {
                Object.entries(data.clinical_concepts).forEach(([concept, score]) => {
                    const item = document.createElement('div');
                    item.className = 'concept-item';
                    item.innerHTML = `
                        <span>${concept.replace(/_/g, ' ')}</span>
                        <span>${(score * 100).toFixed(0)}%</span>
                    `;
                    clinicalConcepts.appendChild(item);
                });
            }
            
            results.style.display = 'block';
        }
    </script>
</body>
</html>
    """
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(index_html)
    
    # About template
    about_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - AI Dermatological Diagnosis System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .feature {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>About the System</h1>
        <p>AI-Powered Dermatological Diagnosis: Research Implementation</p>
    </div>

    <div class="container">
        <h2>Research Background</h2>
        <p>This application implements the framework described in the research paper:</p>
        <p><strong>"AI-Powered Dermatological Diagnosis: From Interpretable Models to Clinical Implementation - A Comprehensive Framework for Accessible and Trustworthy Skin Disease Detection"</strong></p>
        
        <h2>System Architecture</h2>
        <div class="feature">
            <h3>Multi-Modal AI Engine</h3>
            <p>Combines visual analysis using EfficientNet-B7 with clinical text processing using BERT-based encoders for comprehensive diagnosis.</p>
        </div>
        
        <div class="feature">
            <h3>Interpretability Layer</h3>
            <p>Provides visual attention maps, concept attribution, and natural language explanations to ensure clinical transparency and trust.</p>
        </div>
        
        <div class="feature">
            <h3>Clinical Integration</h3>
            <p>Designed for seamless integration with healthcare workflows, supporting both primary care and specialist use cases.</p>
        </div>
        
        <h2>Key Features</h2>
        <ul>
            <li><strong>Multi-modal Input:</strong> Processes both images and clinical text</li>
            <li><strong>Interpretable Results:</strong> Provides explanations for diagnostic decisions</li>
            <li><strong>Confidence Scoring:</strong> Quantifies diagnostic uncertainty</li>
            <li><strong>Concept Attribution:</strong> Identifies key visual and clinical features</li>
            <li><strong>Clinical Recommendations:</strong> Suggests appropriate follow-up actions</li>
        </ul>
        
        <h2>Disease Classes</h2>
        <p>The system can classify the following skin conditions:</p>
        <ul>
            <li>Melanoma</li>
            <li>Melanocytic Nevus</li>
            <li>Basal Cell Carcinoma</li>
            <li>Actinic Keratosis</li>
            <li>Benign Keratosis</li>
            <li>Dermatofibroma</li>
            <li>Vascular Lesion</li>
        </ul>
        
        <h2>Important Disclaimer</h2>
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <p><strong>Research Prototype:</strong> This system is for research and demonstration purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical concerns.</p>
        </div>
        
        <a href="/" class="btn">Back to Analysis</a>
    </div>
</body>
</html>
    """
    
    with open(templates_dir / "about.html", "w") as f:
        f.write(about_html)


# Create templates on startup only when explicitly enabled
# This avoids write attempts in read-only serverless environments.
if os.getenv("GENERATE_DEMO_TEMPLATES", "0") == "1":
    try:
        create_templates()
    except OSError as e:
        logger.warning(f"Template generation skipped: {e}")


if __name__ == "__main__":
    import uvicorn

    print("Starting AI Dermatological Diagnosis System...")
    print("Access the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
