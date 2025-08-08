#!/usr/bin/env python3
"""
Demo script for testing the NYC Taxi Duration Prediction API
"""
import requests
import json
import time

# API endpoints
BASE_URL = "http://localhost:9696"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\n🚕 Testing prediction endpoint...")
    
    # Sample request data
    test_data = {
        "PULocationID": 142,  # Times Square
        "DOLocationID": 236,  # Upper East Side
        "trip_distance": 2.5
    }
    
    try:
        response = requests.post(
            PREDICT_URL, 
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Request: {json.dumps(test_data, indent=2)}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            duration = response.json().get("duration")
            print(f"✅ Predicted trip duration: {duration:.2f} minutes")
            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        return False

def test_invalid_request():
    """Test with invalid request data"""
    print("\n🧪 Testing with invalid request...")
    
    invalid_data = {
        "invalid_field": "test"
    }
    
    try:
        response = requests.post(
            PREDICT_URL,
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        print(f"Request: {json.dumps(invalid_data, indent=2)}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 400:
            print("✅ Invalid request handled correctly")
            return True
        else:
            print(f"❌ Expected 400 status code, got {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Invalid request test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 NYC Taxi Duration Prediction API Demo")
    print("=" * 50)
    
    # Check if service is running
    print("Checking if service is running...")
    if not test_health_check():
        print("\n❌ Service is not running. Please start the Flask app first:")
        print("   python src/predict.py")
        return
    
    # Wait a moment for service to be ready
    print("\n⏳ Waiting for service to be ready...")
    time.sleep(2)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health_check():
        tests_passed += 1
    
    if test_prediction():
        tests_passed += 1
        
    if test_invalid_request():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏆 Tests Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the service logs.")

if __name__ == "__main__":
    main()