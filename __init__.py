"""
AI-Powered Dermatological Diagnosis Application
==============================================

This application implements the framework described in the research paper:
"AI-Powered Dermatological Diagnosis: From Interpretable Models to Clinical Implementation"

The system includes:
- Multi-modal AI engine for skin disease classification
- Interpretability layer with visual explanations
- Clinical workflow integration
- Real-time diagnosis and recommendations
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

__version__ = "1.0.0"
__author__ = "AI Dermatology Research Team"
