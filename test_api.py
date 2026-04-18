#!/usr/bin/env python3
"""
Test script for TempoQR API
"""

import requests
import json
import os

# Simple entity mapping function
def load_entity_mappings():
    """Load entity mappings from file"""
    entity_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt"))
    entity_mappings = {}
    
    if os.path.exists(entity_file):
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        try:
                            # Handle QID format (Q23008452)
                            id_part = parts[0].strip()
                            entity_name = parts[1].strip()
                            
                            # Remove Q prefix if present
                            if id_part.startswith('Q'):
                                entity_id = int(id_part[1:])  # Remove 'Q' and convert to int
                            else:
                                entity_id = int(id_part)
                            
                            entity_mappings[entity_id] = entity_name
                        except ValueError:
                            continue
    return entity_mappings

def id_to_entity_name(entity_id, entity_mappings):
    """Convert entity ID to name"""
    if entity_id in entity_mappings:
        return entity_mappings[entity_id]
    return f"entity_{entity_id}"

# Load entity mappings
entity_mappings = load_entity_mappings()
print(f"Loaded {len(entity_mappings)} entity mappings")

def test_predict_api():
    """Test the predict API endpoint"""
    url = "http://localhost:8000/api/v1/predict/single"
    
    # Test data
    test_questions = [
        "Who was the president of the United States in 2020?",
        "What happened in 2015?",
        "Who won the Nobel Prize in 2021?",
        "When was Barack Obama born?",
        "What company did Steve Jobs found?"
    ]
    
    for question in test_questions:
        print(f"\n=== Testing: {question} ===")
        
        payload = {
            "question": question,
            "top_k": 3
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Status: Success")
                print(f"Processing time: {result.get('processing_time', 'N/A')}s")
                print(f"Predictions:")
                
                for i, pred in enumerate(result.get('predictions', []), 1):
                    answer = pred.get('answer', 'N/A')
                    # Try to convert entity ID to name
                    if answer.startswith('entity_'):
                        try:
                            entity_id = int(answer.replace('entity_', ''))
                            entity_name = id_to_entity_name(entity_id, entity_mappings)
                            print(f"  {i}. {entity_name} (confidence: {pred.get('confidence', 0) * 100:.1f}%)")
                        except:
                            print(f"  {i}. {answer} (confidence: {pred.get('confidence', 0) * 100:.1f}%)")
                    else:
                        print(f"  {i}. {answer} (confidence: {pred.get('confidence', 0) * 100:.1f}%)")
                    
            else:
                print(f"Status: Error ({response.status_code})")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to server. Make sure the API server is running.")
            print("Run: python src/api/main.py")
        except Exception as e:
            print(f"Error: {e}")

def test_health_api():
    """Test the health check endpoint"""
    url = "http://localhost:8000/health"
    
    print("\n=== Health Check ===")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Message: {result.get('message', 'N/A')}")
            print(f"Models loaded: {result.get('models_loaded', False)}")
            print(f"Device: {result.get('device', 'unknown')}")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def test_model_info():
    """Test the model info endpoint"""
    url = "http://localhost:8000/api/v1/predict/model/info"
    
    print("\n=== Model Info ===")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"Model name: {result.get('model_name', 'N/A')}")
            print(f"Dataset: {result.get('dataset_name', 'N/A')}")
            print(f"Entities: {result.get('num_entities', 'N/A')}")
            print(f"Relations: {result.get('num_relations', 'N/A')}")
            print(f"Timestamps: {result.get('num_timestamps', 'N/A')}")
            print(f"Device: {result.get('device', 'N/A')}")
            print(f"Model loaded: {result.get('model_loaded', False)}")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("TempoQR API Test Script")
    print("=" * 50)
    
    # Test health first
    test_health_api()
    
    # Test model info
    test_model_info()
    
    # Test predictions
    test_predict_api()
    
    print("\n" + "=" * 50)
    print("Test completed!")
