#!/usr/bin/env python3
"""
Lightweight test to identify bottlenecks in the API
"""
import requests
import json
import time

# API configuration
API_BASE_URL = "https://bajaj-finserv-x74m.onrender.com"
AUTH_TOKEN = "343c934c163f8f87a6a809c5c79729281f6fdbf03592227539766d3097f11fcd"

def test_with_single_question():
    """Test with just one question to identify bottleneck"""
    print("🧪 Testing with SINGLE question to identify bottleneck...")
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/EDLHLGA23009V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": [
            "What is the grace period?"  # Single simple question
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📤 Sending request to {API_BASE_URL}/hackrx/run")
        print(f"📄 Document URL: {payload['documents']}")
        print(f"❓ Single Question: {payload['questions'][0]}")
        
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"⏱️ Total Time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"📝 Answer: {result.get('answers', ['No answer'])[0]}")
            return True
        else:
            print(f"❌ Request failed:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Request timed out after {elapsed:.2f} seconds")
        print("❌ This confirms there's a bottleneck in the processing pipeline")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Lightweight API Bottleneck Test")
    print("=" * 60)
    
    # Test root endpoint first
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint working!")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return

    print("=" * 60)
    
    # Test with single question
    success = test_with_single_question()
    
    print("\n" + "=" * 60)
    print("🏁 Bottleneck Analysis:")
    
    if success:
        print("✅ API processing working - try with more questions")
    else:
        print("❌ Bottleneck identified! Check these likely causes:")
        print("   1. 🧠 Embedding generation (HuggingFace model loading)")
        print("   2. 📄 PDF parsing (large/complex document)")
        print("   3. 💾 FAISS vector store creation")
        print("   4. 🤖 OpenAI API calls")
        print("   5. 💻 Memory/CPU limits on Render")
        print("\n💡 Check your Render logs for the exact step where it stops!")

if __name__ == "__main__":
    main()
