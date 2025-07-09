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
from datetime import datetime

from .ai_agent import compare_cars_costs, run_car_agent

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
        "message": "AI Car Comparison API is running!",
        "version": "1.0.0",
        "endpoints": {
            "compare": "/compare",
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

# ====================================================================
# SERVER STARTUP
# ====================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = True):
    """Start the FastAPI server"""
    print(f"🚀 Starting AI Car Comparison API on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    
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

# Export the FastAPI app for Vercel
handler = app

# Add a simple test endpoint for debugging
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for Vercel deployment debugging"""
    return {
        "message": "API is working on Vercel!",
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "has_api_key": bool(os.getenv("GEMINI_API_KEY")),
        "environment": "vercel"
    }

# ====================================================================
# REACT NATIVE INTEGRATION EXAMPLES
# ====================================================================

"""
REACT NATIVE INTEGRATION EXAMPLE:

// In your React Native app:
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

// Usage:
const comparisonData = await compareCarsCosts("2018 BMW X5", "2020 Mercedes G500");
"""

# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 