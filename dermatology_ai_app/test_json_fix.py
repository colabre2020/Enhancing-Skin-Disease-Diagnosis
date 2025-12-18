#!/usr/bin/env python3
"""
Test script to verify the JSON serialization fix
"""

import requests
import json
from PIL import Image
import io
import base64

def test_api_fix():
    """Test the /analyze endpoint with an image"""
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='tan')
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Prepare the request
    files = {
        'file': ('test_image.png', img_buffer, 'image/png')
    }
    
    data = {
        'clinical_history': 'Test lesion for API verification',
        'age': 45,
        'gender': 'female',
        'skin_type': 'type_III',
        'lesion_location': 'arm',
        'symptoms': 'No symptoms reported'
    }
    
    try:
        print("🧪 Testing API endpoint with image upload...")
        response = requests.post('http://localhost:8000/analyze', files=files, data=data)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ JSON serialization successful!")
            print(f"🎯 Success: {result.get('success', 'Unknown')}")
            
            if result.get('success'):
                print("📋 Predictions received:")
                for disease, prob in result.get('predictions', {}).items():
                    print(f"  • {disease}: {prob:.3f}")
                
                print(f"🔥 Overall confidence: {result.get('confidence', {}).get('overall_confidence', 'N/A')}")
                print("✅ Test PASSED - No JSON serialization errors!")
                return True
            else:
                print(f"❌ API returned error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("🔧 JSON Serialization Fix Test")
    print("=" * 50)
    
    success = test_api_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The JSON serialization fix is working correctly.")
        print("You can now upload images through the web interface without errors.")
    else:
        print("❌ TESTS FAILED!")
        print("There may still be JSON serialization issues.")
    
    print("\n💡 To test manually:")
    print("1. Open http://localhost:8000 in your browser")
    print("2. Upload a skin lesion image")
    print("3. Fill out the clinical information")
    print("4. Click 'Analyze' - it should work without JSON errors")
