import os
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent / "src"))

from config import load_config
from model_provider import build_chat_model
from langchain_core.messages import HumanMessage

def test_api():
    print("Loading config...")
    config = load_config()
    print(f"Provider: {config.model.provider}")
    print(f"Model: {config.model.model_name}")
    print(f"Base URL: {config.model.base_url}")
    print(f"API Key loaded: {'Yes' if config.model.api_key else 'No'}")
    
    print("\nBuilding model...")
    try:
        model = build_chat_model(config.model)
        
        print("Sending test message...")
        response = model.invoke([HumanMessage(content="Hello, can you hear me? Please reply with a short greeting.")])
        print("\nAPI Response SUCCESS!")
        print("-" * 20)
        print(response.content)
        print("-" * 20)
    except Exception as e:
        print("\nAPI Response FAILED!")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()
