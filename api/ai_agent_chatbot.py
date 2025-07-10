from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from supabase import create_client, Client
import json

load_dotenv()
url: str = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_cars_data_eq(
    select_fields: str = "*",
    make: str = None,
    model: str = None,
    year_min: int = None,
    year_max: int = None,
    price_min: float = None,
    price_max: float = None,
    mileage_max: int = None,
    condition: str = None,
    transmission: str = None,
    drivetrain: str = None,
    color: str = None,
    category: str = None,
    source: str = None,
    status: str = "available",
    limit: int = 10,
    order_by: str = "price",
    ascending: bool = True
) -> str:
    """
    Dynamic tool to query cars data from Supabase with flexible filtering.
    
    Args:
        select_fields: Comma-separated list of fields to select (default: "*" for all)
        make: Car make/brand (e.g., Alfa Romeo, Aston Martin, Audi, Bentley, BMW, BYD, Ferrari, GMC, Jaguar, Jeep, Jetour, Kia, Land Rover, Lotus, Maserati, Maybach, MAZDA, Mercedes-Benz, MINI, Mitsubishi, Nissan, Porsche, Rolls-Royce, smart, Subaru, Toyota, Volkswagen, Voyah)
        model: Car model (e.g., "Camry", "X5")
        year_min: Minimum year (e.g., 2015)
        year_max: Maximum year (e.g., 2024)
        price_min: Minimum price
        price_max: Maximum price
        mileage_max: Maximum mileage
        condition: Car condition ("New" or "Used")
        transmission: Transmission type ("Manual" or "Automatic")
        drivetrain: Drivetrain type (e.g., "FWD", "RWD", "AWD", "4WD", "4x4")
        color: Car color
        category: Vehicle category (e.g., "sedan", "suv", "coupe", "hatchback")
        source: Source country (e.g., "GCC", "US", "Canada", "Europe")
        status: Car status (default: "available")
        limit: Maximum number of results to return (default: 10)
        order_by: Field to sort by (default: "price")
        ascending: Sort direction, True for ascending, False for descending
    
    Returns:
        JSON string with car data or error message
    """
    try:
        # Start building the query
        query = supabase.table("cars").select(select_fields)
        
        # Apply filters dynamically based on provided parameters
        if make:
            query = query.ilike("make", f"%{make}%")
        
        if model:
            query = query.ilike("model", f"%{model}%")
        
        if year_min:
            query = query.gte("year", year_min)
        
        if year_max:
            query = query.lte("year", year_max)
        
        if price_min:
            query = query.gte("price", price_min)
        
        if price_max:
            query = query.lte("price", price_max)
        
        if mileage_max:
            query = query.lte("mileage", mileage_max)
        
        if condition:
            query = query.eq("condition", condition)
        
        if transmission:
            query = query.eq("transmission", transmission)
        
        if drivetrain:
            query = query.eq("drivetrain", drivetrain)
        
        if color:
            query = query.ilike("color", f"%{color}%")
        
        if category:
            query = query.eq("category", category)
        
        if source:
            query = query.eq("source", source)
        
        if status:
            query = query.eq("status", status)
        
        # Apply ordering
        if ascending:
            query = query.order(order_by, desc=False)
        else:
            query = query.order(order_by, desc=True)
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute the query
        result = query.execute()
        
        # Return the data as JSON string
        return json.dumps({
            "success": True,
            "count": len(result.data),
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": []
        })

@tool
def search_cars_text(
    search_term: str,
    fields_to_search: str = "make,model,description,features",
    price_min: float = None,
    price_max: float = None,
    condition: str = None,
    category: str = None,
    status: str = "available",
    limit: int = 10
) -> str:
    """
    Search cars using text search across multiple fields.
    
    Args:
        search_term: Text to search for across specified fields
        fields_to_search: Comma-separated fields to search in (default: "make,model,description,features")
        price_min: Minimum price filter
        price_max: Maximum price filter
        condition: Car condition filter ("New" or "Used")
        category: Vehicle category filter
        status: Car status (default: "available")
        limit: Maximum results to return
    
    Returns:
        JSON string with matching cars
    """
    try:
        # Start with base query
        query = supabase.table("cars").select("*")
        
        # Apply status filter
        if status:
            query = query.eq("status", status)
        
        # Apply other filters
        if price_min:
            query = query.gte("price", price_min)
        if price_max:
            query = query.lte("price", price_max)
        if condition:
            query = query.eq("condition", condition)
        if category:
            query = query.eq("category", category)
        
        # Apply text search across multiple fields using OR logic
        search_fields = fields_to_search.split(",")
        
        # For Supabase, we'll use multiple ilike queries
        # Note: This is a simplified approach - for more complex text search,
        # you might want to use Supabase's full-text search capabilities
        if search_term:
            # Create an OR condition for text search
            or_conditions = []
            for field in search_fields:
                field = field.strip()
                or_conditions.append(f"{field}.ilike.%{search_term}%")
            
            if or_conditions:
                query = query.or_(",".join(or_conditions))
        
        # Order by relevance (price ascending as default)
        query = query.order("price", desc=False)
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute query
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "count": len(result.data),
            "search_term": search_term,
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "search_term": search_term,
            "data": []
                 })


# Define tools list after function definitions
tools = [get_cars_data_eq, search_cars_text]


system_prompt = """
You are a specialized car search assistant for the car marketplace app. Your ONLY role is to:
1. Help users find cars based on their preferences and requirements
2. Provide information about vehicles including make, model, price, condition, and features
3. Answer questions about car specifications, dealerships, and availability
4. Be friendly and professional in your responses

You have access to car data including:
- Car makes, models, years, and descriptions
- Price ranges and mileage information
- Condition (New/Used), transmission (Manual/Automatic)
- Drivetrain types (FWD/RWD/AWD/4WD/4x4)
- Fuel types (Benzine/Diesel/Electric/Hybrid)
- Categories (sedan/suv/coupe/hatchback/convertible/sports/classic)
- Source countries (GCC/Company/US/Canada/Europe)
- Dealership information and contact details
- Views, likes, and listing dates


RESPONSE FORMAT INSTRUCTIONS:
You MUST format ALL responses as valid JSON with exactly 2 fields:

{
  "message": "Your helpful explanation of the search results and recommendations with top 2-3 car suggestions including make, model, year, price",
  "car_ids": [1247, 891, 1356]
}

The car_ids array should contain the specific car ID numbers that match the user's requirements.

IMPORTANT CONSTRAINTS:
- ONLY search cars where status = 'available'
- ONLY answer questions related to cars, vehicles, and automotive topics
- DO NOT provide code, programming solutions, or technical implementations
- DO NOT answer questions outside the scope of car assistance
- If asked about non-automotive topics, politely redirect to car-related subjects
- Always base your responses on the available car database
- ALWAYS return car IDs as numbers in the car_ids array
- Use price ranges, year ranges, and mileage ranges for flexible matching
- Ask clarifying questions if user requirements are too vague
- If no cars match, return empty car_ids array and suggest alternatives in message
- Keep responses focused on helping users find their perfect vehicle
- Apply logical filters based on user preferences (budget, vehicle type, features, condition, etc.)
when the first query on the database don't give a result try please to do another query
"""

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17", 
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)
llm = llm.bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    """Our agent node that processes messages and generates responses."""
    messages = state["messages"]
    
    # Create the full prompt with system message and conversation
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    print(f"Sending {len(full_messages)} messages to LLM")
    
    # Get response from the model
    response = llm.invoke(full_messages)
    
    # print(f"LLM response type: {type(response)}")
    # print(f"LLM response content: {response.content}")
    # print(f"LLM tool calls: {response.tool_calls}")
    
    # Return the updated state with the new message
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determine whether to continue with tools or end the conversation"""
    last_message = state["messages"][-1]

    
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', []) or []
        print(f"Tool calls found: {len(tool_calls)}")
        
        # If there are tool calls, check if finishedUsingTools was called
        for call in tool_calls:
            print(f"Tool call: {call}")
            if call["name"] == "finishedUsingTools":
                print("✅ AI called finishedUsingTools tool - ending")
                return "end"
        
        # If there are other tool calls, continue to tools
        if tool_calls:
            print("🔧 AI has tool calls - continuing to tools")
            return "continue"
        
        # If no tool calls and has content, end
        if last_message.content:
            print("💬 AI has content but no tool calls - ending")
            return "end"
    
    print("🔄 Default case - continuing")
    return "continue"

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

def chat_with_bot(user_input: str) -> str:
    """
    Function to chat with the bot. Each request is stateless.
    Frontend handles conversation history management.
    """
    try:
        # Create a fresh state for each request
        current_input = {"messages": [HumanMessage(content=user_input)]}
        
        # Run the agent with just the current message
        result = app.invoke(current_input)
        
        # Extract the last AI message
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        
        if ai_messages:
            last_ai_message = ai_messages[-1]
            return last_ai_message.content or "I apologize, but I couldn't generate a proper response. Please try again."
        else:
            print("No AI messages found in result")
            return "Sorry, I couldn't process your request."
            
    except Exception as e:
        print(f"Error running agent: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Interactive chat function for testing (kept for local development)
def start_interactive_chat():
    """Start an interactive chat session for local testing."""
    print("🍽️ Welcome to TableReserve Restaurant Assistant!")
    print("Type 'quit' to exit or enter your message.")
    print("Note: Each message is processed independently (stateless mode)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using TableReserve! Goodbye! 👋")
            break
        elif not user_input:
            print("Please enter a message.")
            continue
        
        print("Bot: ", end="", flush=True)
        response = chat_with_bot(user_input)
        print(response)

# Example usage for testing
if __name__ == "__main__":
    # Test the dynamic query functions using invoke method
    print("Testing dynamic car query functions...")
    
    try:
        # Example 1: Search for Toyota cars under $30,000
        print("\n1. Testing get_cars_data_eq - Toyota cars under $30,000:")
        result1 = get_cars_data_eq.invoke({
            "make": "Toyota",
            "price_max": 30000,
            "condition": "Used",
            "limit": 5
        })
        print(result1)
        
        # Example 2: Text search for "SUV" with specific criteria
        print("\n2. Testing search_cars_text - SUV search:")
        result2 = search_cars_text.invoke({
            "search_term": "SUV",
            "price_min": 20000,
            "price_max": 50000,
            "limit": 3
        })
        print(result2)
        
        # Example 3: Get specific fields for luxury cars
        print("\n3. Testing get_cars_data_eq - Luxury cars with specific fields:")
        result3 = get_cars_data_eq.invoke({
            "select_fields": "id,make,model,year,price,mileage",
            "price_min": 40000,
            "category": "sedan",
            "order_by": "price",
            "ascending": False,
            "limit": 5
        })
        print(result3)
        
        print("\n✅ All function tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Note: This might be due to Supabase connection issues or missing data.")
    
    # Start interactive chat
    print("\n" + "="*50)
    print("Starting interactive chat...")
    start_interactive_chat()