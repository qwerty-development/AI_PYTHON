#!/usr/bin/env python
import sys
sys.path.append('.')
from api.ai_agent_chatbot import app, AgentState
from langchain_core.messages import HumanMessage, AIMessage
import json

def debug_ai_response():
    print("=== Debugging AI Response ===")
    
    # Create the input state
    messages = [HumanMessage(content="i am looking for a car for my family")]
    current_input = {"messages": messages}
    
    print("Input:", current_input)
    print("\nRunning agent...")
    
    try:
        # Run the agent and capture the full result
        result = app.invoke(current_input)
        
        print(f"\nFull result keys: {result.keys()}")
        print(f"Number of messages: {len(result['messages'])}")
        
        # Print all messages
        for i, msg in enumerate(result["messages"]):
            print(f"\nMessage {i}:")
            print(f"  Type: {type(msg).__name__}")
            if hasattr(msg, 'content'):
                print(f"  Content: {msg.content}")
            if hasattr(msg, 'tool_calls'):
                tool_calls = getattr(msg, 'tool_calls', []) or []
                print(f"  Tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"    - {tc.get('name', 'Unknown')} with args: {tc.get('args', {})}")
        
        # Get the last AI message specifically
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            print(f"\nLast AI message content: '{last_ai_message.content}'")
            print(f"Last AI message type: {type(last_ai_message.content)}")
            
            # Try to see if it's valid JSON
            if last_ai_message.content:
                try:
                    parsed = json.loads(last_ai_message.content)
                    print(f"Successfully parsed as JSON: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"NOT valid JSON: {e}")
            else:
                print("Content is empty or None")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ai_response() 