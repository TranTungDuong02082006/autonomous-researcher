import sys
import os
import yaml
import argparse

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.llm_server import LLMClient

def main():
    parser = argparse.ArgumentParser(description="Test LLM Server Client wrapper")
    parser.add_argument("prompt", type=str, nargs="?", default="Hello, system-2 researcher!", help="Prompt to send to the LLM")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    llm_config = config.get("llm", {})
    print(f"Initializing LLMClient with model: {llm_config.get('model')} on backend: {llm_config.get('backend')}...")
    
    try:
        client = LLMClient(
            model_name=llm_config.get("model"),
            backend=llm_config.get("backend"),
            quantization=llm_config.get("quantization"),
            api_base=llm_config.get("api_base"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048)
        )
        
        messages = [{"role": "user", "content": args.prompt}]
        print(f"\nSending message: {args.prompt}")
        print("Waiting for response (ensure your LLM server/backend is running if using vllm/openai)...")
        
        response = client.generate(messages)
        print("\n=== LLM RESPONSE ===")
        print(response)
        print("====================")
    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")
        print("Tip: If you don't have a vLLM server running, modify 'backend' in configs/base.yaml to 'openai' or 'transformers'.")

if __name__ == "__main__":
    main()
