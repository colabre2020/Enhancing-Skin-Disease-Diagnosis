#!/usr/bin/env python3
"""
Setup and Demo Script for AI Dermatological Diagnosis System
===========================================================

This script sets up the environment and runs a demonstration of the system
described in the research paper.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json

def print_header():
    """Print the application header"""
    print("=" * 80)
    print("🏥 AI-Powered Dermatological Diagnosis System")
    print("=" * 80)
    print("Implementation of the research paper:")
    print("'AI-Powered Dermatological Diagnosis: From Interpretable Models")
    print("to Clinical Implementation - A Comprehensive Framework for")
    print("Accessible and Trustworthy Skin Disease Detection'")
    print("=" * 80)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies that are essential
    core_deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0"
    ]
    
    # Optional dependencies for full functionality
    optional_deps = [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.0",
        "aiofiles>=23.0.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ]
    
    def install_package(package):
        """Install a single package"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            return True
        except subprocess.CalledProcessError:
            return False
    
    # Install core dependencies
    print("Installing core dependencies...")
    for dep in core_deps:
        package_name = dep.split(">=")[0]
        if install_package(dep):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ❌ {package_name} (failed)")
    
    # Install optional dependencies
    print("Installing optional dependencies...")
    failed_optional = []
    for dep in optional_deps:
        package_name = dep.split(">=")[0]
        if install_package(dep):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ⚠️  {package_name} (optional, failed)")
            failed_optional.append(package_name)
    
    if failed_optional:
        print(f"\n⚠️  Some optional dependencies failed to install: {', '.join(failed_optional)}")
        print("The system will run in limited mode without these features.")
    
    print("✅ Dependency installation completed!")

def create_demo_data():
    """Create demonstration data"""
    print("\n🎯 Creating demonstration data...")
    
    try:
        # Import after dependencies are installed
        sys.path.insert(0, str(Path(__file__).parent))
        from training import create_demo_dataset
        
        # Create demo dataset
        demo_dir = Path("demo_data")
        demo_dir.mkdir(exist_ok=True)
        
        print("Generating synthetic skin lesion images...")
        image_paths, labels, histories, metadata = create_demo_dataset(
            num_samples=50,  # Smaller dataset for demo
            output_dir=str(demo_dir)
        )
        
        print(f"✅ Created {len(image_paths)} demo images in {demo_dir}")
        
    except Exception as e:
        print(f"⚠️  Demo data creation failed: {e}")
        print("Continuing without demo data...")

def run_quick_test():
    """Run a quick test of the core functionality"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.ai_engine import create_demo_model
        from PIL import Image
        import numpy as np
        
        # Create demo model
        print("Initializing AI model...")
        model = create_demo_model()
        
        # Create test image
        print("Creating test image...")
        test_image = Image.new('RGB', (224, 224), color='lightpink')
        
        # Test diagnosis
        print("Running diagnostic test...")
        test_metadata = {
            "age": 45,
            "gender": "female", 
            "skin_type": "type_III",
            "lesion_location": "arm"
        }
        
        result = model.diagnose(
            test_image,
            "Test lesion for system validation",
            test_metadata
        )
        
        print("✅ Core functionality test passed!")
        print(f"   Top prediction: {result.explanations['top_prediction']}")
        print(f"   Confidence: {result.confidence_scores['overall_confidence']:.2f}")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        print("Please check the installation and try again.")

def start_web_application():
    """Start the web application"""
    print("\n🌐 Starting web application...")
    print("The application will be available at: http://localhost:8000")
    print("API documentation will be at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the web app
        sys.path.insert(0, str(Path(__file__).parent))
        import web_app
        
        # This will run the FastAPI server
        web_app.uvicorn.run(
            "web_app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except ImportError:
        print("❌ Web application dependencies not available.")
        print("Running in basic demo mode...")
        run_basic_demo()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        print("Falling back to basic demo...")
        run_basic_demo()

def run_basic_demo():
    """Run a basic command-line demo"""
    print("\n🖥️  Running basic command-line demo...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.ai_engine import create_demo_model
        from PIL import Image, ImageDraw
        import random
        
        # Create demo model
        model = create_demo_model()
        
        # Create a synthetic lesion image
        img = Image.new('RGB', (224, 224), color='pink')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple lesion
        x, y = 112, 112
        size = 40
        color = (random.randint(50, 150), random.randint(30, 100), random.randint(20, 80))
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline=(0, 0, 0))
        
        # Save demo image
        demo_img_path = "demo_lesion.png"
        img.save(demo_img_path)
        print(f"Created demo lesion image: {demo_img_path}")
        
        # Run diagnosis
        metadata = {
            "age": 52,
            "gender": "male",
            "skin_type": "type_II", 
            "lesion_location": "back"
        }
        
        clinical_history = "Patient reports a growing lesion on the back, noticed 3 months ago. Irregular borders and color variation observed."
        
        print("\nRunning AI diagnosis...")
        result = model.diagnose(img, clinical_history, metadata)
        
        # Display results
        print("\n" + "="*50)
        print("🏥 DIAGNOSTIC RESULTS")
        print("="*50)
        
        print(f"Patient: {metadata['age']} year old {metadata['gender']}")
        print(f"Location: {metadata['lesion_location']}")
        print(f"Skin Type: {metadata['skin_type']}")
        print()
        
        print("PREDICTIONS:")
        sorted_predictions = sorted(result.predictions.items(), key=lambda x: x[1], reverse=True)
        for disease, probability in sorted_predictions:
            disease_name = disease.replace('_', ' ').title()
            bar_length = int(probability * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {disease_name:<25} {bar} {probability:.1%}")
        
        print()
        print(f"Overall Confidence: {result.confidence_scores['overall_confidence']:.1%}")
        print(f"Prediction Entropy: {result.confidence_scores['prediction_entropy']:.3f}")
        
        print("\n" + result.explanations.get('text_explanation', 'No explanation available'))
        
        print("\n" + "="*50)
        print("✅ Demo completed successfully!")
        print("This demonstrates the core AI diagnostic capabilities.")
        print("For the full web interface, install FastAPI dependencies.")
        
    except Exception as e:
        print(f"❌ Basic demo failed: {e}")
        print("Please check the core dependencies and try again.")

def main():
    """Main setup and demo function"""
    print_header()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create demo data
    create_demo_data()
    
    # Run quick test
    run_quick_test()
    
    # Ask user what to do next
    print("\n🚀 Setup completed! What would you like to do?")
    print("1. Start web application (recommended)")
    print("2. Run basic command-line demo")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            start_web_application()
            break
        elif choice == "2":
            run_basic_demo()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
