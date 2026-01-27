#!/usr/bin/env python3
"""
Quick test to verify all vLLM servers are working
"""

from openai import OpenAI
import sys

# Get node names from command line
if len(sys.argv) != 6:
    print("Usage: python test_vllm_servers.py <user_node> <4b_node> <8b_node> <14b_node> <32b_node>")
    print("Example: python test_vllm_servers.py sg007 sg008 sg009 sg010 sg011")
    sys.exit(1)

user_node = sys.argv[1]
node_4b = sys.argv[2]
node_8b = sys.argv[3]
node_14b = sys.argv[4]
node_32b = sys.argv[5]

# Define all servers
servers = {
    "User-Simulator (port 8000)": f"http://{user_node}:8000/v1",
    "Agent-4B (port 8001)": f"http://{node_4b}:8001/v1",
    "Agent-8B (port 8002)": f"http://{node_8b}:8002/v1",
    "Agent-14B (port 8003)": f"http://{node_14b}:8003/v1",
    "Agent-32B (port 8004)": f"http://{node_32b}:8004/v1",
}

print("="*60)
print("Testing All vLLM Servers")
print("="*60)

all_passed = True

for name, url in servers.items():
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        # Create client
        client = OpenAI(base_url=url, api_key="dummy")
        
        # Send test request
        response = client.chat.completions.create(
            model="qwen",  # Model name doesn't matter for vLLM
            messages=[
                {"role": "user", "content": "Say 'Hello from server!' and nothing else."}
            ],
            max_tokens=20,
            temperature=0.0
        )
        
        # Get response
        reply = response.choices[0].message.content
        
        print(f"✅ SUCCESS!")
        print(f"Response: {reply}")
        
    except Exception as e:
        print(f"❌ FAILED!")
        print(f"Error: {e}")
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("🎉 ALL SERVERS WORKING! Ready for τ-bench experiments!")
else:
    print("⚠️ SOME SERVERS FAILED! Check errors above.")
print("="*60)
