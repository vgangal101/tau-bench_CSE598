import os
from litellm import completion

# Test connection to local vLLM server
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "dummy"  # vLLM doesn't require real key

print("Testing LiteLLM connection to vLLM server...")
print(f"API Base: {os.environ.get('OPENAI_API_BASE')}")
print(f"Model: qwen3-4b")

try:
    response = completion(
        model="qwen3-4b",
        messages=[{"role": "user", "content": "Hello! Please respond with 'Connection successful'."}],
        custom_llm_provider="openai",
        temperature=0.0,
    )
    
    print("\n✅ SUCCESS! vLLM server is accessible via LiteLLM")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Is vLLM server running? Start with: vllm serve qwen3-4b --host 0.0.0.0 --port 8000")
    print("2. Check server status: curl http://localhost:8000/v1/models")
    print("3. Is the model name correct? Should be: qwen3-4b")
