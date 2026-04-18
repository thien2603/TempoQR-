import requests
import json
import time

def test_api():
    """Test TempoQR FastAPI với các câu hỏi mẫu"""
    
    base_url = "http://localhost:8000"
    
    # Test questions
    test_questions = [
        "Who is the CEO of Apple?",
        "Who was president of USA in 2020?", 
        "What company did Steve Jobs found?",
        "When did World War 2 end?",
        "What is the capital of France?",
        "Who invented the telephone?",
        "When was Barack Obama born?",
        "What company makes iPhone?",
        "Who wrote Romeo and Juliet?"
    ]
    
    print("🧪 Testing TempoQR FastAPI")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        
        try:
            # Gửi request đến API
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/ask",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Kiểm tra response
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Answer: {result['answer']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"⏱️  Response time: {response_time:.2f}s")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Error details: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Error: Request timeout")
        except requests.exceptions.ConnectionError:
            print("🔌 Error: Cannot connect to server")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)
    
    print("\n🎉 API Testing completed!")

def test_health():
    """Test API health endpoint"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            print(f"📄 Response: {response.json()}")
        else:
            print(f"❌ API returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def test_performance():
    """Test API performance với nhiều request"""
    print("\n🚀 Performance Test (10 concurrent requests)")
    
    import threading
    import queue
    
    results = queue.Queue()
    
    def make_request(question):
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end = time.time()
            results.put({
                'success': response.status_code == 200,
                'time': end - start,
                'question': question
            })
        except Exception as e:
            results.put({
                'success': False,
                'error': str(e),
                'question': question
            })
    
    # Tạo 10 request đồng thời
    questions = ["Who is the CEO of Apple?"] * 10
    threads = []
    
    start_time = time.time()
    
    for question in questions:
        thread = threading.Thread(target=make_request, args=(question,))
        thread.start()
        threads.append(thread)
    
    # Đợi tất cả threads hoàn thành
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    # Phân tích kết quả
    successful = 0
    total_time = 0
    
    while not results.empty():
        result = results.get()
        if result['success']:
            successful += 1
            total_time += result['time']
    
    print(f"📊 Performance Results:")
    print(f"   Total requests: {len(questions)}")
    print(f"   Successful: {successful}")
    print(f"   Success rate: {successful/len(questions)*100:.1f}%")
    print(f"   Average response time: {total_time/successful:.2f}s" if successful > 0 else "N/A")
    print(f"   Total test time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    print("🧪 TempoQR API Test Suite")
    print("Make sure FastAPI server is running on http://localhost:8000")
    print()
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Basic functionality
    test_api()
    
    # Test 3: Performance
    test_performance()
    
    print("\n🎯 Testing Tips:")
    print("   1. Start server: uvicorn app:app --host 0.0.0.0 --port 8000")
    print("   2. Open browser: http://localhost:8000/docs for API documentation")
    print("   3. Use curl or Python script for testing")
    print("   4. Monitor GPU memory usage during testing")
