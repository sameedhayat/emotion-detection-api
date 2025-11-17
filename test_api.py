#!/usr/bin/env python3
"""
Test script for the Emotion Detection API
Run this after starting the server to verify functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test single tweet prediction"""
    print("Testing /predict endpoint...")
    data = {
        "text": "I am so happy and excited about this amazing opportunity!"
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_prediction():
    """Test batch tweet prediction"""
    print("Testing /predict/batch endpoint...")
    data = {
        "texts": [
            "I am so happy and excited!",
            "This makes me really angry and frustrated.",
            "I feel very sad and disappointed.",
            "Things are looking great, I'm optimistic about the future!"
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Emotion Detection API Test Suite")
        print("=" * 60 + "\n")
        
        test_health()
        test_single_prediction()
        test_batch_prediction()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {str(e)}")
