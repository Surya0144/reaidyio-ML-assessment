import requests
import json

# Configuration
API_URL = "http://host.docker.internal:8000/predict"
SAMPLE_DATA = {
    "features": [6.5, 3.0, 5.2, 2.0]  # Virginica
}

def test_prediction():
    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=SAMPLE_DATA)
        response.raise_for_status()
        
        result = response.json()
        print("\n Prediction Successful!")
        print(f"Input: {SAMPLE_DATA['features']}")
        print(f"Class: {result['class_name']} (ID: {result['class_id']})")
        print(f"Confidence: {result['confidence_score']:.2f}")
        
    except requests.exceptions.ConnectionError:
        print(" Error: Could not connect. Is Docker running?")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    test_prediction()