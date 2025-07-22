#!/usr/bin/env python
import sys
sys.path.append('.')
from api.ai_agent_chatbot import chat_with_bot
import json

def test_chat_function():
    print("=== Testing Chat Function for Family Car Search ===")
    
    # Test the actual chat function that the user would call
    user_input = "i am looking for a car for my family"
    
    print(f"User input: {user_input}")
    print("Calling chat_with_bot function...")
    
    try:
        result = chat_with_bot(user_input)
        print(f"Raw result: {result}")
        
        # Parse the JSON response
        response_data = json.loads(result)
        print(f"\nParsed response:")
        print(f"Message: {response_data.get('message', 'No message')}")
        print(f"Car IDs: {response_data.get('car_ids', [])}")
        print(f"Number of car IDs: {len(response_data.get('car_ids', []))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chat_function() 