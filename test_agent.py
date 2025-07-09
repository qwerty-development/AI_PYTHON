#!/usr/bin/env python3
"""
Test script for the AI Car Comparison Agent
Tests the conventional message approach with SystemMessage + HumanMessage
"""

import asyncio
import os
from ai_agent import compare_cars_costs, run_car_agent

async def test_basic_comparison():
    """Test basic car comparison functionality"""
    print("🧪 Testing AI Car Comparison Agent...")
    print("=" * 50)
    
    # Test with default cars
    print("Testing with default cars: 2018 BMW X5 vs 2020 Mercedes-Benz G500")
    
    try:
        result = await run_car_agent("2018 BMW X5", "2020 Mercedes-Benz G500")
        
        if "error" in result:
            print(f"❌ Test failed: {result['error']}")
            return False
        
        print("✅ Basic comparison test passed!")
        print(f"📊 Status: {result.get('metadata', {}).get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

async def test_custom_comparison():
    """Test with custom car models"""
    print("\n🧪 Testing with custom cars...")
    print("=" * 50)
    
    try:
        result = await compare_cars_costs("2022 Tesla Model Y", "2022 Audi Q5")
        
        if "error" in result:
            print(f"❌ Custom test failed: {result['error']}")
            return False
        
        print("✅ Custom comparison test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Custom test failed with exception: {str(e)}")
        return False

def test_environment():
    """Test environment setup"""
    print("🔧 Testing environment setup...")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("❌ GEMINI_API_KEY not set properly")
        print("📝 Please create a .env file with your Gemini API key")
        return False
    
    print("✅ Environment setup looks good!")
    return True

async def main():
    """Run all tests"""
    print("🚀 Starting AI Car Agent Tests")
    print("=" * 60)
    
    # Test environment first
    if not test_environment():
        print("\n❌ Environment test failed. Please fix the setup first.")
        return
    
    # Run basic tests
    basic_test = await test_basic_comparison()
    
    # Run custom tests if basic test passed
    if basic_test:
        custom_test = await test_custom_comparison()
        
        if basic_test and custom_test:
            print("\n🎉 All tests passed! The AI agent is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the logs above.")
    else:
        print("\n❌ Basic test failed. Skipping additional tests.")

if __name__ == "__main__":
    asyncio.run(main()) 