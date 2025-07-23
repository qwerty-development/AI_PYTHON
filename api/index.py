#!/usr/bin/env python3
"""
FastAPI REST API wrapper for the AI Car Comparison Agent
This creates HTTP endpoints that can be consumed by React Native apps.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import uvicorn
import os
import platform
import json
from datetime import datetime

# Flexible imports to allow running as a module (python -m api.index)
# or as a script from inside the `api` folder (python index.py)
try:
    # When `api` is a package (e.g., run with `python -m api.index` or uvicorn from project root)
    from .ai_agent import compare_cars_costs, run_car_agent
    from .ai_agent_chatbot import chat_with_bot
except ImportError:  # pragma: no cover
    # Fallback for running directly inside the api folder
    from ai_agent import compare_cars_costs, run_car_agent
    from ai_agent_chatbot import chat_with_bot

# ====================================================================
# FASTAPI APP SETUP
# ====================================================================
app = FastAPI(
    title="AI Car Comparison API",
    description="REST API for comparing cars using AI-powered cost analysis",
    version="1.0.0"
)

# Enable CORS for React Native and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React Native app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================
# REQUEST/RESPONSE MODELS
# ====================================================================
class CarComparisonRequest(BaseModel):
    """Request model for car comparison"""
    car1: str
    car2: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "car1": "2018 BMW X5",
                "car2": "2020 Mercedes-Benz G500"
            }
        }

class ChatRequest(BaseModel):
    """Request model for AI chatbot"""
    message: str
    conversation_history: list = None  # Optional conversation history
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I'm looking for a BMW under $30,000",
                "conversation_history": [
                    {"role": "user", "content": "What are the cars available"},
                    {"role": "assistant", "content": "Found 160 cars..."}
                ]
            }
        }

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Dict[str, Any] = None
    error: str = None
    message: str = None

# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Car Comparison & Chatbot API is running!",
        "version": "1.0.0",
        "endpoints": {
            "compare": "/compare",
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {"status": "healthy", "service": "ai-car-comparison"}

@app.post("/compare", response_model=ApiResponse)
async def compare_cars(request: CarComparisonRequest):
    """
    Compare two cars and return comprehensive cost analysis
    
    Args:
        request: CarComparisonRequest with car1 and car2
        
    Returns:
        ApiResponse with comparison data or error
    """
    try:
        # Validate input
        if not request.car1.strip() or not request.car2.strip():
            raise HTTPException(
                status_code=400, 
                detail="Both car1 and car2 must be provided and non-empty"
            )
        
        # Run the car comparison
        result = await run_car_agent(request.car1.strip(), request.car2.strip())
        
        # Check if there was an error in the agent execution
        if "error" in result:
            return ApiResponse(
                success=False,
                error=result["error"],
                message="Car comparison failed"
            )
        
        # Return successful result
        return ApiResponse(
            success=True,
            data=result,
            message=f"Successfully compared {request.car1} vs {request.car2}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/compare/{car1}/{car2}")
async def compare_cars_get(car1: str, car2: str):
    """
    GET endpoint for car comparison (alternative to POST)
    
    Args:
        car1: First car to compare
        car2: Second car to compare
        
    Returns:
        Comparison results
    """
    request = CarComparisonRequest(car1=car1, car2=car2)
    return await compare_cars(request)

# ====================================================================
# AI CHATBOT ENDPOINT
# ====================================================================

@app.post("/chat", response_model=ApiResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Chat with the AI car assistant
    
    Args:
        request: ChatRequest with user message and optional conversation history
        
    Returns:
        ApiResponse with AI response containing message and car_ids
    """
    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(
                status_code=400, 
                detail="Message cannot be empty"
            )
        
        # Get response from AI chatbot (ignore conversation_history)
        ai_response = chat_with_bot(
            request.message.strip()
        )
        
        # Try to parse the AI response as JSON (since it should return structured data)
        try:
            import json
            parsed_response = json.loads(ai_response)
            
            return ApiResponse(
                success=True,
                data=parsed_response,
                message="AI response generated successfully"
            )
            
        except json.JSONDecodeError:
            # If response is not JSON, return as plain text
            return ApiResponse(
                success=True,
                data={"message": ai_response, "car_ids": []},
                message="AI response generated successfully"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ====================================================================
# EXAMPLE ENDPOINTS FOR TESTING
# ====================================================================

@app.get("/examples")
async def get_examples():
    """Get example car comparisons"""
    return {
        "examples": [
            {"car1": "2018 BMW X5", "car2": "2020 Mercedes-Benz G500"},
            {"car1": "2022 Tesla Model Y", "car2": "2022 Audi Q5"},
            {"car1": "2021 Toyota RAV4", "car2": "2021 Honda CR-V"},
            {"car1": "2020 Porsche Cayenne", "car2": "2020 BMW X6"}
        ]
    }

@app.post("/compare/example")
async def compare_example_cars():
    """Run a comparison with example cars"""
    return await compare_cars(
        CarComparisonRequest(car1="2018 BMW X5", car2="2020 Mercedes-Benz G500")
    )

@app.post("/chat/example")
async def chat_example():
    """Test the chatbot with an example message"""
    return await chat_with_ai(
        ChatRequest(message="I'm looking for a BMW under $50,000")
    )

# ====================================================================
# SERVER STARTUP
# ====================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = True):
    """Start the FastAPI server"""
    print(f"🚀 Starting AI Car Comparison & Chatbot API on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"💬 Chat Example: http://{host}:{port}/chat/example")
    
    uvicorn.run(
        "index:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

# ====================================================================
# VERCEL DEPLOYMENT HANDLER
# ====================================================================

# Export the FastAPI app for Vercel (ASGI)
# Vercel automatically detects the `app` variable for ASGI/WGSI frameworks.

# Add a simple test endpoint for debugging
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for Vercel deployment debugging"""
    return {
        "message": "API is working on Vercel!",
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "has_api_key": bool(os.getenv("GEMINI_API_KEY")),
        "has_supabase_url": bool(os.getenv("EXPO_PUBLIC_SUPABASE_URL")),
        "has_supabase_key": bool(os.getenv("EXPO_PUBLIC_SUPABASE_ANON_KEY")),
        "environment": "vercel"
    }

@app.get("/debug-db")
async def debug_db():
    """Debug database connection"""
    try:
        from .ai_agent_chatbot import search_cars
        
        # Test simple database query
        result = search_cars.invoke({"category": "SUV", "limit": 1})
        result_data = json.loads(result)
        
        return {
            "success": True,
            "database_connection": "working",
            "query_result": result_data.get("success", False),
            "total_count": result_data.get("total_count", 0)
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "database_connection": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ====================================================================
# REACT NATIVE INTEGRATION EXAMPLES
# ====================================================================

"""
REACT NATIVE INTEGRATION EXAMPLE:

// In your React Native app:

// 1. CAR COMPARISON API
const compareCarsCosts = async (car1, car2) => {
  try {
    const response = await fetch('https://your-vercel-app.vercel.app/compare', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        car1: car1,
        car2: car2
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      return result.data;
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('Car comparison failed:', error);
    throw error;
  }
};

// 2. AI CHATBOT API
const chatWithAI = async (message) => {
  try {
    const response = await fetch('https://your-vercel-app.vercel.app/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      return result.data; // Contains: { message: "AI response", car_ids: [1, 2, 3] }
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('Chat failed:', error);
    throw error;
  }
};

// Usage Examples:
const comparisonData = await compareCarsCosts("2018 BMW X5", "2020 Mercedes G500");
const chatResponse = await chatWithAI("I'm looking for a BMW under $50,000");
"""

# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 