#!/usr/bin/env python3
"""
====================================================================
AI CAR COMPARISON AGENT - Improved Version
====================================================================
This agent compares two cars by iteratively searching for specific
information using a focused web search tool. The AI determines what
to search for and formats the final JSON response. hello world
====================================================================
"""

import os
import json
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import quote_plus

# LangGraph and LangChain imports
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ====================================================================
# ENHANCED WEB SEARCH TOOL
# ====================================================================
@tool
async def web_search_tool(query: str) -> str:
    """
    Search the web for specific car information using multiple search engines and sources.
    
    Args:
        query: Specific search query (e.g., "2018 BMW X5 current market price")
        
    Returns:
        Formatted search results with source attribution
    """
    try:
        print(f"🔍 Searching: {query}")
        
        # ====================================================================
        # MULTIPLE SEARCH STRATEGIES
        # ====================================================================
        search_variations = [
            query,
            f"{query} 2024 current",
            f"{query} edmunds kelley blue book",
            f"{query} consumer reports autotrader",
            f"{query} dealer pricing market value"
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for search_query in search_variations:
                try:
                    # DuckDuckGo search
                    url = f"https://api.duckduckgo.com/?q={quote_plus(search_query)}&format=json&no_html=1&skip_disambig=1"
                    
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            result_text = await _extract_search_data(data, search_query)
                            
                            if result_text:
                                results.append(result_text)
                    
                    # Rate limiting
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    print(f"Search failed for: {search_query}")
                    continue
        
        # ====================================================================
        # COMPILE RESULTS
        # ====================================================================
        if results:
            compiled_results = f"Search Results for: {query}\n\n"
            for i, result in enumerate(results, 1):
                compiled_results += f"Result {i}:\n{result}\n\n"
            
            compiled_results += f"Total sources found: {len(results)}\n"
            compiled_results += f"Search performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return compiled_results
        
        # ====================================================================
        # FALLBACK WITH SPECIFIC GUIDANCE
        # ====================================================================
        return await _provide_fallback_guidance(query)
        
    except Exception as error:
        return f"Search error for '{query}': {str(error)}\n\nRecommendation: Try rephrasing the search query or search for more specific terms."

async def _extract_search_data(data: Dict, query: str) -> str:
    """Extract relevant information from DuckDuckGo API response"""
    result_parts = []
    
    # Abstract information
    if data.get("Abstract"):
        result_parts.append(f"Summary: {data['Abstract']}")
        if data.get("AbstractSource"):
            result_parts.append(f"Source: {data['AbstractSource']}")
    
    # Direct answers (great for specific facts)
    if data.get("Answer"):
        result_parts.append(f"Direct Answer: {data['Answer']}")
    
    # Infobox data (structured information)
    if data.get("Infobox") and data["Infobox"].get("content"):
        info_items = []
        for item in data["Infobox"]["content"][:5]:  # Limit to top 5
            if item.get("label") and item.get("value"):
                info_items.append(f"{item['label']}: {item['value']}")
        
        if info_items:
            result_parts.append("Specifications:\n" + "\n".join(info_items))
    
    # Related topics (additional context)
    if data.get("RelatedTopics"):
        topic_items = []
        for topic in data["RelatedTopics"][:3]:  # Limit to top 3
            if topic.get("Text"):
                topic_items.append(f"- {topic['Text']}")
                if topic.get("FirstURL"):
                    topic_items.append(f"  Source: {topic['FirstURL']}")
        
        if topic_items:
            result_parts.append("Related Information:\n" + "\n".join(topic_items))
    
    return "\n\n".join(result_parts) if result_parts else ""

async def _provide_fallback_guidance(query: str) -> str:
    """Provide specific guidance when search results are limited"""
    query_lower = query.lower()
    
    # Specific guidance based on query type
    if any(term in query_lower for term in ["price", "cost", "value", "msrp"]):
        return f"""Limited search results for: {query}

For accurate pricing information, consider:
- Kelley Blue Book (KBB.com) - Industry standard for vehicle values
- Edmunds.com - Comprehensive pricing and reviews
- AutoTrader.com - Current market listings
- Cars.com - Real-time dealer pricing
- Manufacturer websites - Official MSRP and incentives

Typical price ranges for luxury vehicles:
- Used luxury SUVs: $25,000-$80,000 depending on year/model
- New luxury SUVs: $50,000-$150,000+
- Depreciation: 15-25% first year, 45-60% after 5 years"""

    elif any(term in query_lower for term in ["insurance", "coverage"]):
        return f"""Limited search results for: {query}

Insurance cost factors:
- Vehicle value and theft rates
- Safety ratings and crash test scores
- Repair costs and parts availability
- Driver age, location, and driving record

Typical annual insurance costs:
- Luxury SUVs: $2,000-$4,000/year
- Sports cars: $3,000-$6,000/year
- Geographic variation: ±40% based on location
- Recommend getting quotes from multiple insurers"""

    elif any(term in query_lower for term in ["maintenance", "repair", "service"]):
        return f"""Limited search results for: {query}

Maintenance considerations:
- German luxury brands: Higher maintenance costs
- Typical annual costs: $1,200-$2,500/year
- Major services: Every 40,000-60,000 miles
- Warranty coverage: Varies by manufacturer

Factors affecting costs:
- Vehicle age and mileage
- Service location (dealer vs independent)
- Preventive vs reactive maintenance
- Parts availability and labor rates"""

    elif any(term in query_lower for term in ["fuel", "mpg", "gas"]):
        return f"""Limited search results for: {query}

Fuel economy considerations:
- Large SUVs: 15-25 MPG combined
- Luxury vehicles often require premium fuel
- Annual fuel costs: $2,000-$4,000 (12,000 miles/year)
- Hybrid options: 25-35 MPG combined

Cost calculation:
- Miles per year ÷ MPG × fuel price per gallon
- Premium fuel adds 10-15% to costs
- Driving style significantly impacts actual MPG"""

    else:
        return f"""Limited search results for: {query}

General recommendations:
- Try more specific search terms
- Include model year and trim level
- Search multiple automotive websites
- Consider consulting local dealers
- Check manufacturer official websites

For comprehensive car comparisons:
- Consumer Reports
- Motor Trend
- Car and Driver
- J.D. Power ratings"""

# ====================================================================
# AI AGENT CREATION
# ====================================================================
def create_ai_agent():
    """Initialize the AI model and create the LangGraph agent"""
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,  # Slightly higher for better reasoning
        max_retries=3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    tools = [web_search_tool]
    agent = create_react_agent(model, tools)
    
    return agent

# ====================================================================
# MAIN EXECUTION FUNCTION
# ====================================================================
async def run_car_agent(car1: str = "2018 BMW X5", car2: str = "2020 Mercedes-Benz G500") -> Dict[str, Any]:
    """
    Run the car comparison agent with iterative search approach
    """
    try:
        # ====================================================================
        # API KEY VALIDATION
        # ====================================================================
        if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "YOUR_GEMINI_API_KEY":
            print('❌ Missing GEMINI_API_KEY!')
            print('📝 To fix this:')
            print('1. Create a .env file in your project root')
            print('2. Add: GEMINI_API_KEY=your_actual_api_key')
            print('3. Get your API key from: https://aistudio.google.com/app/apikey')
            return {
                "error": "Missing GEMINI_API_KEY",
                "metadata": {
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            }

        print(f'🚗 Starting AI Car Comparison: {car1} vs {car2}\n')
        
        # Create the agent
        agent = create_ai_agent()
        
        # ====================================================================
        # SYSTEM MESSAGE WITH JSON STRUCTURE
        # ====================================================================
        system_message = SystemMessage(content=f"""You are an expert automotive analyst specializing in comprehensive vehicle cost comparisons. 

Your task is to compare {car1} vs {car2} by systematically gathering information using the web_search_tool, then provide a detailed analysis in the specified JSON format.

SEARCH STRATEGY:
1. Search for current market pricing for both vehicles
2. Research annual insurance costs for both models
3. Find maintenance/repair cost data for both vehicles
4. Look up fuel economy and annual fuel costs
5. Research depreciation rates and resale values
6. Find registration/tax information
7. Search for any additional ownership costs

Use the web_search_tool iteratively - make multiple targeted searches to gather comprehensive data. Be specific in your search queries (e.g., "2018 BMW X5 current market price insurance cost" rather than just "BMW X5").

FINAL RESPONSE FORMAT:
You must return your analysis in this exact JSON structure:

{{
  "comparison_date": "{datetime.now().isoformat()}",
  "cars_compared": ["{car1}", "{car2}"],
  
  "first_car": {{
    "name": "{car1}",
    "current_market_value": 0,
    "annual_costs": {{
      "insurance": 0,
      "maintenance": 0,
      "fuel": 0,
      "registration": 0,
      "total_annual": 0
    }},
    "depreciation": {{
      "annual_rate": 0.00,
      "five_year_loss": 0
    }},
    "five_year_ownership": {{
      "depreciation": 0,
      "insurance_total": 0,
      "maintenance_total": 0,
      "fuel_total": 0,
      "registration_total": 0,
      "total_cost": 0
    }}
  }},
  
  "second_car": {{
    "name": "{car2}",
    "current_market_value": 0,
    "annual_costs": {{
      "insurance": 0,
      "maintenance": 0,
      "fuel": 0,
      "registration": 0,
      "total_annual": 0
    }},
    "depreciation": {{
      "annual_rate": 0.00,
      "five_year_loss": 0
    }},
    "five_year_ownership": {{
      "depreciation": 0,
      "insurance_total": 0,
      "maintenance_total": 0,
      "fuel_total": 0,
      "registration_total": 0,
      "total_cost": 0
    }}
  }},
  
  "comparison_summary": {{
    "more_economical_car": "",
    "cost_difference": 0,
    "percentage_savings": 0.00,
    "key_factors": []
  }},
  
  "data_sources": [],
  "analysis_notes": []
}}

IMPORTANT:
- All monetary values should be in USD (numbers only, no $ symbols)
- Percentages should be decimals (e.g., 0.15 for 15%)
- Fill in ALL fields with realistic estimates based on your research
- If exact data isn't found, use industry-standard estimates but note this
- The "key_factors" should list the main reasons for cost differences
- Include all search sources in "data_sources"
- Add relevant notes about assumptions or limitations in "analysis_notes"

Begin by searching for comprehensive information about both vehicles.""")

        # ====================================================================
        # HUMAN MESSAGE
        # ====================================================================
        human_message = HumanMessage(content=f"""Please perform a comprehensive cost comparison between {car1} and {car2}.

Use the web_search_tool to gather current information about:
1. Market pricing/values
2. Insurance costs
3. Maintenance expenses
4. Fuel economy and costs
5. Depreciation rates
6. Registration fees
7. Any other ownership costs

After gathering this information, provide your analysis in the exact JSON format specified in the system message. Make sure all numerical fields are populated with realistic estimates based on your research.""")

        # ====================================================================
        # AGENT EXECUTION
        # ====================================================================
        print("🤖 AI Agent starting analysis...")
        
        result = await agent.ainvoke({
            "messages": [system_message, human_message]
        })
        
        # ====================================================================
        # EXTRACT AND PARSE RESPONSE
        # ====================================================================
        messages = result["messages"]
        final_message = messages[-1].content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', final_message, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
                
                # Add metadata
                parsed_response["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "agent_version": "2.0",
                    "search_method": "iterative_web_search"
                }
                
                print('\n✅ Car Comparison Analysis Complete!')
                return parsed_response
            else:
                # If no JSON found, return the raw response with metadata
                return {
                    "error": "JSON parsing failed",
                    "ai_response": final_message,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "status": "partial_completion",
                        "note": "AI provided analysis but not in expected JSON format"
                    }
                }
                
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "ai_response": final_message,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "parsing_error"
                }
            }
            
    except Exception as error:
        print(f'❌ Error: {str(error)}')
        return {
            "error": f"Agent execution failed: {str(error)}",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "status": "execution_error",
                "cars_compared": [car1, car2]
            }
        }

# ====================================================================
# PUBLIC API FUNCTIONS
# ====================================================================
async def compare_cars_costs(car1: str, car2: str) -> Dict[str, Any]:
    """
    Public API function for car comparison
    
    Args:
        car1: First car (e.g., "2018 BMW X5")
        car2: Second car (e.g., "2020 Mercedes-Benz G500")
        
    Returns:
        Dictionary with comprehensive comparison data
    """
    return await run_car_agent(car1, car2)

async def quick_car_search(query: str) -> str:
    """
    Quick search function for specific car information
    
    Args:
        query: Search query (e.g., "2018 BMW X5 insurance cost")
        
    Returns:
        Search results string
    """
    return await web_search_tool(query)

# ====================================================================
# COMMAND LINE INTERFACE
# ====================================================================
def main():
    """Command line interface for the car comparison agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Car Comparison Agent')
    parser.add_argument('--car1', default='2018 BMW X5', help='First car to compare')
    parser.add_argument('--car2', default='2020 Mercedes-Benz G500', help='Second car to compare')
    parser.add_argument('--search', help='Perform a quick search instead of full comparison')
    
    args = parser.parse_args()
    
    if args.search:
        # Quick search mode
        result = asyncio.run(quick_car_search(args.search))
        print(result)
    else:
        # Full comparison mode
        result = asyncio.run(run_car_agent(args.car1, args.car2))
        print(json.dumps(result, indent=2))

# ====================================================================
# DIRECT EXECUTION
# ====================================================================
if __name__ == "__main__":
    main()