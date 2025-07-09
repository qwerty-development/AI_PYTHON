#!/usr/bin/env python3
"""
Example demonstrating the conventional message approach
SystemMessage + HumanMessage pattern for better AI agent behavior
"""

import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from ai_agent import create_ai_agent

async def example_conventional_messages():
    """
    Example of how to use the conventional message structure
    """
    print("🔄 Creating LangGraph agent with conventional message approach...")
    
    # Create the agent
    agent = create_ai_agent()
    
    # System message defines the agent's role and capabilities
    system_message = SystemMessage(content="""You are an AI car comparison specialist.

Your role:
- Analyze car cost data comprehensively
- Use tools to gather real-time information
- Provide structured financial comparisons
- Return detailed JSON responses

Your tools:
- car_comparison_tool: For comprehensive cost analysis
- web_search_tool: For additional real-time data

Always be thorough and use realistic industry data.""")
    
    # Human message contains the specific user request
    human_message = HumanMessage(content="""Please compare a 2022 Tesla Model Y vs 2022 BMW iX using the car_comparison_tool.

I need:
1. Annual cost estimates
2. 5-year depreciation analysis
3. Total ownership cost comparison
4. Which car is more economical overall

Please provide the complete JSON analysis.""")
    
    print("📤 Messages being sent to agent:")
    print(f"System: {system_message.content[:100]}...")
    print(f"Human: {human_message.content[:100]}...")
    
    try:
        # Invoke agent with conventional message structure
        result = await agent.ainvoke({
            "messages": [system_message, human_message]
        })
        
        print("\n✅ Agent execution completed!")
        print(f"📨 Total messages in conversation: {len(result['messages'])}")
        
        # Print message flow for educational purposes
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:150] + "..." if len(str(msg.content)) > 150 else str(msg.content)
            print(f"  {i+1}. {msg_type}: {content_preview}")
        
        # Extract final response
        final_response = result["messages"][-1].content
        print(f"\n🎯 Final response length: {len(final_response)} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during agent execution: {str(e)}")
        return None

async def main():
    """
    Main function to run the conventional message example
    """
    print("🚀 Conventional Message Approach Example")
    print("=" * 50)
    
    result = await example_conventional_messages()
    
    if result:
        print("\n🎉 Example completed successfully!")
        print("\nKey benefits of this approach:")
        print("✅ Clear separation of system instructions vs user requests")
        print("✅ Better agent understanding of its role and capabilities")
        print("✅ More predictable and reliable agent behavior")
        print("✅ Easier debugging and conversation flow tracking")
        print("✅ Standard LangGraph/LangChain best practices")
    else:
        print("\n❌ Example failed - check your environment setup")

if __name__ == "__main__":
    asyncio.run(main()) 