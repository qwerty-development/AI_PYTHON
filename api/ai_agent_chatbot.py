from typing import TypedDict, Annotated, Sequence, List, Dict, Any
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
from datetime import datetime, timedelta
import requests
from urllib.parse import quote

load_dotenv()
url: str = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Centralized category mapping to avoid duplication
CATEGORY_SYNONYMS = {
    "sports car": "Sports",
    "sport car": "Sports", 
    "sports": "Sports",
    "sport": "Sports",
    "sedan": "Sedan",
    "suv": "SUV",
    "crossover": "SUV", 
    "family car": "SUV",
    "family": "SUV",
    "large family": "SUV",
    "big family": "SUV",
    "7 seater": "SUV",
    "8 seater": "SUV",
    "hatchback": "Hatchback",
    "coupe": "Coupe",
    "convertible": "Convertible"
}

def normalize_category(category: str) -> str:
    """Normalize category using centralized mapping"""
    if not category:
        return category
    return CATEGORY_SYNONYMS.get(category.lower(), category)

def build_search_query(filters: Dict[str, Any], search_term: str = None) -> str:
    """
    UNIFIED search function that handles all search scenarios.
    Consolidates get_cars_data_eq, search_cars_text, and search_cars_advanced.
    """
    try:
        # Build the query
        select_fields = filters.get("select_fields", "*")
        query = supabase.table("cars").select(select_fields)
        
        # Always filter by status
        status = filters.get("status", "available")
        if status:
            query = query.eq("status", status)
        
        # Apply structured filters
        if filters.get("make"):
            make = filters["make"]
            make_terms = make.split()
            if len(make_terms) > 1:
                make_conditions = [f"make.ilike.%{term}%" for term in make_terms]
                query = query.or_(",".join(make_conditions))
            else:
                query = query.ilike("make", f"%{make}%")
        
        if filters.get("model"):
            model = filters["model"]
            model_terms = model.split()
            if len(model_terms) > 1:
                model_conditions = [f"model.ilike.%{term}%" for term in model_terms]
                query = query.or_(",".join(model_conditions))
            else:
                query = query.ilike("model", f"%{model}%")
        
        # Price filters
        if filters.get("price_min") is not None:
            query = query.gte("price", int(filters["price_min"]))
        if filters.get("price_max") is not None:
            query = query.lte("price", int(filters["price_max"]))
        
        # Year filters
        if filters.get("year_min"):
            query = query.gte("year", filters["year_min"])
        if filters.get("year_max"):
            query = query.lte("year", filters["year_max"])
        
        # Other filters
        if filters.get("mileage_max"):
            query = query.lte("mileage", filters["mileage_max"])
        if filters.get("condition"):
            query = query.eq("condition", filters["condition"])
        if filters.get("transmission"):
            query = query.eq("transmission", filters["transmission"])
        if filters.get("drivetrain"):
            query = query.eq("drivetrain", filters["drivetrain"])
        if filters.get("source"):
            query = query.eq("source", filters["source"])
        if filters.get("color"):
            query = query.ilike("color", f"%{filters['color']}%")
        
        # Category filter with normalization
        if filters.get("category"):
            normalized_category = normalize_category(filters["category"])
            query = query.eq("category", normalized_category)
        
        # Text search across multiple fields (combines search_cars_text functionality)
        if search_term:
            search_fields = filters.get("search_fields", ["make", "model", "description", "features"])
            search_keywords = search_term.split()
            or_conditions = []
            
            for keyword in search_keywords:
                for field in search_fields:
                    or_conditions.append(f"{field}.ilike.%{keyword}%")
            
            if or_conditions:
                query = query.or_(",".join(or_conditions))
        
        # Features filter (array search)
        if filters.get("features"):
            features = filters["features"]
            if isinstance(features, list):
                for feature in features:
                    query = query.filter("features", "cs", f'["{feature}"]')
            else:
                query = query.filter("features", "cs", f'["{features}"]')
        
        # Apply sorting
        order_by = filters.get("order_by", "price")
        ascending = filters.get("ascending", True)
        query = query.order(order_by, desc=not ascending)
        
        # Execute the query to get ALL matching results
        result = query.execute()
        
        # Remove duplicates and extract all IDs
        unique_data = []
        all_car_ids = []
        seen_ids = set()
        for car in result.data:
            car_id = car.get("id")
            if car_id is None or car_id in seen_ids:
                continue
            seen_ids.add(car_id)
            unique_data.append(car)
            all_car_ids.append(car_id)
        
        # For AI context: only provide details of first 5 cars (performance optimization)
        AI_CONTEXT_LIMIT = 5
        cars_for_ai_context = unique_data[:AI_CONTEXT_LIMIT]
        
        return json.dumps({
            "success": True,
            "total_count": len(unique_data),
            "data": cars_for_ai_context,  # Only first 5 for AI to process
            "all_car_ids": all_car_ids,   # ALL matching IDs for frontend
            "filters_applied": filters,
            "search_term": search_term,
            "ai_context_limit": AI_CONTEXT_LIMIT,
            "showing_details_for": len(cars_for_ai_context)
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
            "data": [],
            "all_car_ids": []
        })

@tool
def search_cars(
    # Basic filters
    make: str = None,
    model: str = None,
    category: str = None,
    
    # Price filters
    price_min: float = None,
    price_max: float = None,
    budget_max: float = None,  # Alias for price_max for better UX
    
    # Year filters
    year_min: int = None,
    year_max: int = None,
    
    # Other common filters
    condition: str = None,
    mileage_max: int = None,
    transmission: str = None,
    color: str = None,
    
    # Text search
    search_term: str = None,
    
    # Advanced options
    sort_by: str = "price",
    ascending: bool = True,
    features: str = None,
    status: str = "available"
) -> str:
    """
    UNIFIED car search tool that handles all search scenarios.
    Replaces get_cars_data_eq, search_cars_text, and search_cars_advanced.
    
    Args:
        # Basic filters
        make: Car make/brand (fuzzy matching)
        model: Car model (fuzzy matching) 
        category: Vehicle category (SUV, Sedan, Sports, etc.) - handles synonyms
        
        # Price filters
        price_min: Minimum price
        price_max: Maximum price
        budget_max: Alternative to price_max for budget searches
        
        # Year filters
        year_min: Minimum year
        year_max: Maximum year
        
        # Other filters
        condition: "New" or "Used"
        mileage_max: Maximum mileage
        transmission: "Manual" or "Automatic"
        color: Car color (fuzzy matching)
        
        # Text search
        search_term: Free-text search across make, model, description, features
        
        # Advanced
        sort_by: Field to sort by (price, year, mileage, views)
        ascending: Sort direction
        features: Comma-separated list of required features
        status: Car status (default: "available")
    
    Returns:
        JSON string with ALL matching cars
    """
    # Handle budget_max alias
    if budget_max and not price_max:
        price_max = budget_max
    
    # Build filters dictionary
    filters = {
        "make": make,
        "model": model,
        "category": category,
        "price_min": price_min,
        "price_max": price_max,
        "year_min": year_min,
        "year_max": year_max,
        "condition": condition,
        "mileage_max": mileage_max,
        "transmission": transmission,
        "color": color,
        "status": status,
        "order_by": sort_by,
        "ascending": ascending
    }
    
    # Handle features
    if features:
        filters["features"] = [f.strip() for f in features.split(",")]
    
    try:
        return build_search_query(filters, search_term)
    except Exception as e:
        # Debug: Log the error for troubleshooting
        print(f"Error in search_cars: {e}")
        print(f"Filters: {filters}")
        print(f"Search term: {search_term}")
        return json.dumps({
            "success": False,
            "error": f"Search error: {str(e)}",
            "total_count": 0,
            "data": [],
            "all_car_ids": []
        })

# search_cars_text functionality is now integrated into search_cars tool

@tool
def get_similar_cars(
    car_id: int,
    similarity_factors: str = "make,category,price_range,year_range",
    price_tolerance: float = 0.3,
    year_tolerance: int = 5
) -> str:
    """
    Find ALL cars similar to a specific car based on various factors.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        car_id: ID of the reference car
        similarity_factors: Comma-separated factors to consider
        price_tolerance: Price tolerance as percentage (0.3 = 30%)
        year_tolerance: Year tolerance in years (5 = ±5 years)
    
    Returns:
        JSON string with ALL similar cars
    """
    try:
        # First get the reference car
        ref_result = supabase.table("cars").select("*").eq("id", car_id).execute()
        
        if not ref_result.data:
            return json.dumps({
                "success": False,
                "error": "Reference car not found",
                "data": []
            })
        
        ref_car = ref_result.data[0]
        factors = [f.strip() for f in similarity_factors.split(",")]
        
        # Start building query for similar cars
        query = supabase.table("cars").select("*").neq("id", car_id).eq("status", "available")
        
        # Apply similarity filters based on factors
        conditions = []
        
        if "make" in factors:
            query = query.eq("make", ref_car["make"])
        
        if "model" in factors:
            query = query.eq("model", ref_car["model"])
        
        if "category" in factors:
            query = query.eq("category", ref_car["category"])
        
        if "source" in factors:
            query = query.eq("source", ref_car["source"])
        
        if "price_range" in factors and ref_car.get("price"):
            price = float(ref_car["price"])
            price_min = int(price * (1 - price_tolerance))
            price_max = int(price * (1 + price_tolerance))
            query = query.gte("price", price_min).lte("price", price_max)
        
        if "year_range" in factors and ref_car.get("year"):
            year = int(ref_car["year"])
            year_min = year - year_tolerance
            year_max = year + year_tolerance
            query = query.gte("year", year_min).lte("year", year_max)
        
        # Order by price
        query = query.order("price", desc=False)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "reference_car": ref_car,
            "similarity_factors": factors,
            "total_count": len(result.data),
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
            "data": []
        })

@tool
def get_cars_by_budget_range(
    budget_min: float,
    budget_max: float,
    category: str = None,
    condition: str = None,
    mileage_max: int = None,
    year_min: int = None,
    transmission: str = None,
    make: str = None,
    sort_by: str = "price"
) -> str:
    """
    SIMPLIFIED: Find cars within budget range using the unified search.
    This is now just a convenience wrapper around search_cars.
    """
    # Use the unified search_cars tool
    filters = {
        "price_min": budget_min,
        "price_max": budget_max,
        "category": category,
        "condition": condition,
        "mileage_max": mileage_max,
        "year_min": year_min,
        "transmission": transmission,
        "make": make,
        "order_by": sort_by,
        "ascending": True,
        "status": "available"
    }
    
    result = build_search_query(filters)
    
    # Add budget_range info to result
    try:
        result_data = json.loads(result)
        if result_data.get("success"):
            result_data["budget_range"] = f"${budget_min:,.0f} - ${budget_max:,.0f}"
            # Ensure all_car_ids is properly set for budget searches
            if not result_data.get("all_car_ids") and result_data.get("data"):
                # Fallback: extract IDs from data if all_car_ids is missing
                result_data["all_car_ids"] = [car.get("id") for car in result_data["data"] if car.get("id")]
        return json.dumps(result_data)
    except:
        return result

@tool
def get_market_insights(
    make: str = None,
    category: str = None,
    condition: str = None,
    year_min: int = None,
    year_max: int = None
) -> str:
    """
    Get comprehensive market insights and statistics for cars with optional filters.
    Analyzes ALL available cars matching the criteria.
    
    Args:
        make: Car make filter
        category: Vehicle category filter
        condition: Car condition filter
        year_min: Minimum year filter
        year_max: Maximum year filter
    
    Returns:
        JSON string with detailed market insights
    """
    try:
        # Build base query
        query = supabase.table("cars").select("*").eq("status", "available")
        
        if make:
            query = query.ilike("make", f"%{make}%")
        if category:
            query = query.eq("category", category)
        if condition:
            query = query.eq("condition", condition)
        if year_min:
            query = query.gte("year", year_min)
        if year_max:
            query = query.lte("year", year_max)
        
        # Get ALL matching cars
        result = query.execute()
        
        if not result.data:
            return json.dumps({
                "success": False,
                "error": "No data found for the specified filters",
                "data": {}
            })
        
        # Calculate comprehensive statistics
        cars = result.data
        prices = [float(car["price"]) for car in cars if car.get("price")]
        mileages = [int(car["mileage"]) for car in cars if car.get("mileage")]
        years = [int(car["year"]) for car in cars if car.get("year")]
        
        # Additional analytics
        makes_count = {}
        categories_count = {}
        conditions_count = {}
        
        for car in cars:
            # Count makes
            car_make = car.get("make", "Unknown")
            makes_count[car_make] = makes_count.get(car_make, 0) + 1
            
            # Count categories
            car_category = car.get("category", "Unknown")
            categories_count[car_category] = categories_count.get(car_category, 0) + 1
            
            # Count conditions
            car_condition = car.get("condition", "Unknown")
            conditions_count[car_condition] = conditions_count.get(car_condition, 0) + 1
        
        # Sort by count for top insights
        top_makes = sorted(makes_count.items(), key=lambda x: x[1], reverse=True)[:10]
        top_categories = sorted(categories_count.items(), key=lambda x: x[1], reverse=True)
        
        insights = {
            "total_cars": len(cars),
            "price_stats": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0,
                "median": sorted(prices)[len(prices)//2] if prices else 0
            },
            "mileage_stats": {
                "min": min(mileages) if mileages else 0,
                "max": max(mileages) if mileages else 0,
                "avg": sum(mileages) / len(mileages) if mileages else 0,
                "median": sorted(mileages)[len(mileages)//2] if mileages else 0
            },
            "year_stats": {
                "min": min(years) if years else 0,
                "max": max(years) if years else 0,
                "avg": sum(years) / len(years) if years else 0
            },
            "top_makes": top_makes,
            "categories_breakdown": top_categories,
            "conditions_breakdown": list(conditions_count.items())
        }
        
        return json.dumps({
            "success": True,
            "insights": insights,
            "filters_applied": {
                "make": make,
                "category": category,
                "condition": condition,
                "year_range": f"{year_min}-{year_max}" if year_min and year_max else None
            }
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": {}
        })

@tool
def get_cars_by_features(
    required_features: str,
    preferred_features: str = None,
    price_max: float = None,
    category: str = None,
    condition: str = None,
    make: str = None
) -> str:
    """
    Find ALL cars based on required and preferred features.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        required_features: Comma-separated list of required features
        preferred_features: Comma-separated list of preferred features
        price_max: Maximum price filter
        category: Vehicle category filter
        condition: Car condition filter
        make: Car make filter
    
    Returns:
        JSON string with ALL cars matching features
    """
    try:
        query = supabase.table("cars").select("*").eq("status", "available")
        
        # Apply basic filters
        if price_max is not None:
            query = query.lte("price", int(price_max))
        if category:
            normalized_category = normalize_category(category)
            query = query.eq("category", normalized_category)
        if condition:
            query = query.eq("condition", condition)
        if make:
            query = query.ilike("make", f"%{make}%")
        
        # Apply feature filters - simplified approach to avoid PostgreSQL array operator issues
        required_list = [f.strip().lower() for f in required_features.split(",") if f.strip()]
        
        # Simple text search in features column (avoiding array operators entirely)
        if required_list:
            # Use simple contains search - much safer
            for feature in required_list:
                if feature:
                    # Simple text containment search
                    query = query.filter("features", "cs", f'["{feature}"]')
        
        # Order by price
        query = query.order("price", desc=False)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        # Score results based on preferred features
        scored_results = result.data
        if preferred_features:
            preferred_list = [f.strip().lower() for f in preferred_features.split(",") if f.strip()]
            
            for car in scored_results:
                score = 0
                car_features = car.get("features", "").lower()
                
                for pref_feature in preferred_list:
                    if pref_feature in car_features:
                        score += 1
                
                car["preference_score"] = score
            
            # Sort by preference score (descending) then by price (ascending)
            scored_results.sort(key=lambda x: (-x.get("preference_score", 0), x.get("price", 0)))
        
        return json.dumps({
            "success": True,
            "required_features": required_list,
            "preferred_features": preferred_features.split(",") if preferred_features else [],
            "total_count": len(scored_results),
            "data": scored_results
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
            "data": []
        })

@tool
def get_recently_added_cars(
    days_back: int = 7,
    category: str = None,
    price_max: float = None,
    condition: str = None,
    make: str = None
) -> str:
    """
    Get ALL recently added cars within specified time period.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        days_back: Number of days to look back (default: 7)
        category: Vehicle category filter
        price_max: Maximum price filter
        condition: Car condition filter
        make: Car make filter
    
    Returns:
        JSON string with ALL recently added cars
    """
    try:
        # Calculate the date threshold
        date_threshold = datetime.now() - timedelta(days=days_back)
        date_str = date_threshold.strftime('%Y-%m-%d')
        
        query = supabase.table("cars").select("*").eq("status", "available")
        # Use listed_at instead of created_at (based on database schema)
        query = query.gte("listed_at", date_str)
        
        if category:
            normalized_category = normalize_category(category)
            query = query.eq("category", normalized_category)
        if price_max is not None:
            query = query.lte("price", int(price_max))
        if condition:
            query = query.eq("condition", condition)
        if make:
            query = query.ilike("make", f"%{make}%")
        
        # Order by listing date (newest first)
        query = query.order("listed_at", desc=True)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "days_back": days_back,
            "date_threshold": date_str,
            "total_count": len(result.data),
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
            "data": []
        })

@tool
def get_popular_cars(
    sort_by: str = "views",
    category: str = None,
    price_range: str = None,
    condition: str = None,
    make: str = None
) -> str:
    """
    Get ALL popular cars based on views, likes, or other popularity metrics.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        sort_by: Sort by 'views', 'likes', or 'created_at'
        category: Vehicle category filter
        price_range: Price range filter (e.g., "20000-40000")
        condition: Car condition filter
        make: Car make filter
    
    Returns:
        JSON string with ALL popular cars
    """
    try:
        query = supabase.table("cars").select("*").eq("status", "available")
        
        if category:
            normalized_category = normalize_category(category)
            query = query.eq("category", normalized_category)
        if condition:
            query = query.eq("condition", condition)
        if make:
            query = query.ilike("make", f"%{make}%")
        
        # Apply price range filter
        if price_range:
            try:
                price_min, price_max = map(float, price_range.split("-"))
                query = query.gte("price", int(price_min)).lte("price", int(price_max))
            except ValueError:
                pass  # Invalid price range format, ignore
        
        # Apply sorting
        if sort_by in ["views", "likes", "created_at"]:
            query = query.order(sort_by, desc=True)
        else:
            query = query.order("views", desc=True)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "sorted_by": sort_by,
            "total_count": len(result.data),
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
            "data": []
        })

# search_cars_advanced functionality is now integrated into search_cars tool

@tool
def search_web_for_car_info(
    query: str,
    max_results: int = 3
) -> str:
    """
    Search the internet for car-related information to better understand user needs.
    Use this when users ask about general car concepts like 'family car', 'sports car',
    'fuel efficient car', etc. to understand what specific car types/categories to search for.
    
    Args:
        query: The search query about cars (e.g., "best family cars", "fuel efficient vehicles")
        max_results: Maximum number of search results to return (default: 5)
    
    Returns:
        JSON string with search results and extracted car insights
    """
    try:
        # Use DuckDuckGo Instant Answer API for car-related searches
        search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        
        # Make the search request
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information (simplified to reduce token usage)
        results = []
        
        # Get abstract if available (truncated)
        if data.get('Abstract'):
            abstract = data['Abstract'][:300] + "..." if len(data['Abstract']) > 300 else data['Abstract']
            results.append({
                "type": "abstract",
                "content": abstract,
                "source": data.get('AbstractSource', 'DuckDuckGo')
            })
        
        # Get related topics (limit and truncate)
        if data.get('RelatedTopics'):
            for i, topic in enumerate(data['RelatedTopics'][:min(3, max_results)]):  # Limit to 3
                if isinstance(topic, dict) and topic.get('Text'):
                    text = topic['Text'][:200] + "..." if len(topic['Text']) > 200 else topic['Text']
                    results.append({
                        "type": "related_topic",
                        "content": text,
                        "source": "DuckDuckGo"  # Simplified
                    })
        
        # ALWAYS use predefined car category mappings FIRST (even if DuckDuckGo has results)
        # This ensures we prioritize expert knowledge for family cars
        car_type_mappings = {
            "family car": ["SUV", "Sedan"],
            "family": ["SUV", "Sedan"],  # Also handle just "family"
            "family of 5": ["SUV"],  # 5+ people need larger vehicles
            "large family": ["SUV"],  # Large families need SUVs
            "big family": ["SUV"],
            "7 seater": ["SUV"],
            "8 seater": ["SUV"],
            "college student": ["Hatchback", "Sedan"],  # Budget-friendly, efficient
            "student": ["Hatchback", "Sedan"],
            "first car": ["Hatchback", "Sedan"],  # Safe, affordable
            "new driver": ["Sedan", "Hatchback"],  # Safe, easy to handle
            "commute": ["Sedan", "Hatchback"],  # Fuel efficient
            "long commute": ["Sedan", "Hatchback"],
            "city driving": ["Hatchback", "Sedan"],  # Compact, maneuverable
            "weekend": ["Convertible", "Sports"],  # Fun driving
            "business": ["Sedan"],  # Professional appearance
            "elderly": ["Sedan", "SUV"],  # Easy entry/exit
            "mobility": ["SUV", "Sedan"],  # Higher seating, easier access
            "sport": ["Sports"],
            "sports car": ["Sports"],
            "sport car": ["Sports"],
            "sporty": ["Sports"],
            "performance": ["Sports"],
            "fast car": ["Sports"],
            "luxury car": ["Sedan"],
            "fuel efficient": ["Sedan", "Hatchback"],
            "off road": ["SUV"],
            "reliable car": ["Sedan", "SUV"],
            "economical car": ["Hatchback", "Sedan"],
            "safe car": ["SUV", "Sedan"],
            "spacious car": ["SUV"]
        }
        
        # Check if query matches any known patterns (more flexible matching)
        query_lower = query.lower()
        suggested_categories = []
        
        for car_type, categories in car_type_mappings.items():
            # More flexible matching - check if any words from car_type are in query
            car_type_words = car_type.split()
            if any(word in query_lower for word in car_type_words):
                suggested_categories.extend(categories)
        
        # Remove duplicates and ensure we have some categories
        if suggested_categories:
            unique_categories = list(set(suggested_categories))
            results.append({
                "type": "category_suggestion",
                "content": f"Based on your search for '{query}', I recommend looking at these vehicle categories: {', '.join(unique_categories)}",
                "categories": unique_categories
            })
            
        # If no predefined mapping and no DuckDuckGo results, use fallback
        if not suggested_categories and not results:
            # Add a generic fallback message
            results.append({
                "type": "fallback",
                "content": f"No specific recommendations found for '{query}'. Try searching for general categories like SUV, Sedan, or Sports.",
                "categories": []
            })
        
        # Extract car categories and types from the results
        extracted_insights = {
            "suggested_categories": [],
            "suggested_makes": [],
            "key_features": [],
            "price_insights": []
        }
        
        # First check if we have direct category suggestions from fallback
        for result in results:
            if result.get("type") == "category_suggestion" and result.get("categories"):
                extracted_insights["suggested_categories"].extend(result["categories"])
        
        # Also analyze text content for car-related keywords
        all_text = " ".join([r.get("content", "") for r in results]).lower()
        
        # Map common car categories to database categories (exact match with database)
        category_mapping = {
            "suv": "SUV",
            "sedan": "Sedan", 
            "hatchback": "Hatchback",
            "coupe": "Coupe",
            "convertible": "Convertible",
            "sports": "Sports",
            "sport": "Sports",  # Map both sport and sports to Sports
            "sporty": "Sports",  # Also map sporty
            "performance": "Sports",  # Map performance cars
            "fast": "Sports",  # Map fast cars
            "minivan": "SUV",  # Map minivan to SUV as closest match
            "crossover": "SUV",  # Map crossover to SUV as closest match
            "luxury": "Sedan",  # Map luxury to Sedan as default
            "wagon": "Hatchback",  # Map wagon to Hatchback as closest match
            "classic": "Classic"  # Add classic category
        }
        
        found_categories = set()
        for search_term, db_category in category_mapping.items():
            if search_term in all_text:
                found_categories.add(db_category)
        
        # Combine direct suggestions and text analysis
        extracted_insights["suggested_categories"].extend(list(found_categories))
        
        # Remove duplicates while preserving order and prioritise SUV if present
        original_cats = extracted_insights["suggested_categories"]
        extracted_insights["suggested_categories"] = list(dict.fromkeys(original_cats))  # dedupe keep order
        if "SUV" in extracted_insights["suggested_categories"]:
            extracted_insights["suggested_categories"].remove("SUV")
            extracted_insights["suggested_categories"].insert(0, "SUV")
        
        # (deduplication already handled above while preserving order)
        
        # Common car makes to look for
        makes = ["toyota", "honda", "ford", "bmw", "mercedes", "audi", "nissan", "hyundai", "kia", "mazda", "subaru", "volkswagen"]
        for make in makes:
            if make in all_text:
                extracted_insights["suggested_makes"].append(make)
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "extracted_insights": extracted_insights,
            "total_results": len(results)
        })
        
    except requests.RequestException as e:
        # Fallback response with common car knowledge (using correct database categories)
        fallback_insights = {
            "family car": {
                "categories": ["SUV", "Sedan"],  # Use database category names
                "features": ["safety", "space", "reliability"],
                "description": "Family cars prioritize safety, space, and reliability"
            },
            "sport": {  # Handle both "sport" and "sports" variations
                "categories": ["Sports", "Coupe", "Convertible"],  # Use database category names
                "features": ["performance", "speed", "handling"],
                "description": "Sports cars focus on performance and driving experience"
            },
            "sports car": {
                "categories": ["Sports", "Coupe", "Convertible"],  # Use database category names
                "features": ["performance", "speed", "handling"],
                "description": "Sports cars focus on performance and driving experience"
            },
            "luxury car": {
                "categories": ["Sedan", "Coupe", "SUV"],  # Use database category names
                "features": ["premium", "comfort", "technology"],
                "description": "Luxury cars offer premium features and comfort"
            },
            "fuel efficient": {
                "categories": ["Sedan", "Hatchback"],  # Use database category names
                "features": ["mpg", "hybrid", "economy"],
                "description": "Fuel efficient cars prioritize low fuel consumption"
            }
        }
        
        # Try to match query with fallback knowledge
        query_lower = query.lower()
        matched_insight = None
        for key, insight in fallback_insights.items():
            if key in query_lower:
                matched_insight = insight
                break
        
        if matched_insight:
            return json.dumps({
                "success": True,
                "query": query,
                "fallback_used": True,
                "extracted_insights": {
                    "suggested_categories": matched_insight["categories"],
                    "key_features": matched_insight["features"],
                    "description": matched_insight["description"]
                },
                "error": f"Web search failed: {str(e)}, used fallback knowledge"
            })
        else:
                    return json.dumps({
            "success": False,
            "error": f"Web search failed: {str(e)}",
            "query": query
        })

@tool
def finishedUsingTools() -> str:
    """
    Signal that the AI has finished using tools and is ready to provide a final response.
    Call this tool when you have all the information needed to answer the user's question.
    """
    return "AI has finished using tools and will now provide the final response."

# Define tools list after function definitions - STREAMLINED
tools = [
    search_cars,  # UNIFIED tool that replaces get_cars_data_eq, search_cars_text, search_cars_advanced
    get_similar_cars, 
    get_cars_by_budget_range,  # Simplified wrapper
    get_market_insights,
    get_cars_by_features,
    get_recently_added_cars,
    get_popular_cars,
    search_web_for_car_info,
    finishedUsingTools  # Signal completion of tool usage
]

system_prompt = """
You are CarFinder.ai – a professional assistant for our online car marketplace.

PRIMARY GOALS
1. Find relevant cars in our Supabase inventory that match the user's needs.
2. Provide factual information about cars (specifications, prices, availability) strictly derived from the tool outputs.

ABSOLUTE RULE – USE TOOLS FIRST
• Before answering you MUST execute at least one of the provided tools.
• Never fabricate car data or IDs.

AVAILABLE TOOLS (STREAMLINED)
• search_cars – MAIN TOOL: handles all search scenarios (filters, keywords, budget, etc.)
• get_similar_cars – "cars similar to this ID"
• get_cars_by_budget_range – budget-specific search (wrapper around search_cars)
• get_recently_added_cars / get_popular_cars – recency & popularity
• get_market_insights – statistics & trends  
• get_cars_by_features – feature-based search
• search_web_for_car_info – turns vague lifestyle requests into concrete filters
• finishedUsingTools – call when done with tools and ready to respond

SIMPLIFIED DECISION GUIDE
1. Most searches → search_cars (handles make, model, price, year, category, keywords, etc.)
2. "Similar to car ID" → get_similar_cars  
3. Lifestyle/vague requests ("family car", "student car") → try search_cars FIRST with category mapping (family→SUV, sport→Sports), only use search_web_for_car_info if search_cars fails
4. Recent/popular cars → get_recently_added_cars / get_popular_cars
5. Market analysis → get_market_insights
6. Feature-specific → get_cars_by_features

IMPORTANT: For performance and reliability, prefer direct database searches over web searches. Use category synonyms: family car→SUV, sports car→Sports, luxury→Sedan.

KEY IMPROVEMENTS
• search_cars handles: structured filters, text search, budget ranges, category synonyms
• No more confusion between similar tools - ONE main search tool
• Automatic category mapping (family car → SUV, sports car → Sports, etc.)
• Better performance with unified queries

WORKFLOW
a) Choose the appropriate tool (usually search_cars for most queries)
b) Run the tool and inspect results
c) Call finishedUsingTools when you have all needed information
d) Reply with EXACTLY ONE JSON object - nothing else

CRITICAL RESPONSE FORMAT - MANDATORY
Your final response MUST be EXACTLY this JSON format with NO additional text before or after:
{
  "message": "Found X cars. {brief description of top 5 cars}. 1. Make Model Year – $Price (ID:###) – Condition ...",
  "car_ids": [list_of_all_matching_car_ids]
}

STRICT FORMATTING RULES:
• NEVER include any text before or after the JSON object
• NO markdown, code fences, or explanatory text
• NO conversational preamble like "Here are the results:" 
• ONLY the raw JSON object as shown above
• Include details of up to 5 cars in "message"; use ALL ids from "all_car_ids" field in "car_ids"
• Always mention total_count in the message
• If no cars match, apologize and suggest adjusting the search parameters
• The "data" field contains only 5 cars for performance, but "all_car_ids" contains ALL matching IDs

VIOLATION WARNING: Any response that is not pure JSON format will be rejected. Do not include ANY other text.
"""

# Initialize the model with the stable GA version
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
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

def validate_and_fix_response(response_content: str) -> str:
    """
    Validate and fix the AI response to ensure consistent JSON format.
    Handles cases where AI provides both descriptive text and JSON.
    """
    try:
        # Clean up markdown formatting if present
        cleaned_content = response_content.strip()
        
        # Look for JSON block in the content
        json_start = -1
        json_end = -1
        
        # Try to find JSON markdown block first
        if '```json' in cleaned_content:
            json_start = cleaned_content.find('```json') + 7
            json_end = cleaned_content.find('```', json_start)
            if json_end != -1:
                cleaned_content = cleaned_content[json_start:json_end].strip()
        # Try to find plain ``` block
        elif '```' in cleaned_content:
            first_triple = cleaned_content.find('```')
            json_start = first_triple + 3
            json_end = cleaned_content.find('```', json_start)
            if json_end != -1:
                cleaned_content = cleaned_content[json_start:json_end].strip()
        # Try to find JSON object by looking for opening brace
        elif '{' in cleaned_content:
            # Try to find the LAST JSON object (in case there are multiple)
            # This handles cases where AI provides text followed by JSON
            potential_json_parts = []
            
            # Find all potential JSON objects
            start_pos = 0
            while True:
                brace_start = cleaned_content.find('{', start_pos)
                if brace_start == -1:
                    break
                    
                # Find the matching closing brace
                brace_count = 0
                brace_end = -1
                for i in range(brace_start, len(cleaned_content)):
                    if cleaned_content[i] == '{':
                        brace_count += 1
                    elif cleaned_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i + 1
                            break
                
                if brace_end != -1:
                    potential_json = cleaned_content[brace_start:brace_end].strip()
                    potential_json_parts.append(potential_json)
                    start_pos = brace_end
                else:
                    start_pos = brace_start + 1
            
            # Try to parse each potential JSON object, preferring the last valid one
            for potential_json in reversed(potential_json_parts):
                try:
                    test_data = json.loads(potential_json)
                    if isinstance(test_data, dict) and "message" in test_data and "car_ids" in test_data:
                        cleaned_content = potential_json
                        break
                except json.JSONDecodeError:
                    continue
            else:
                # If no valid JSON found, use the first potential JSON part
                if potential_json_parts:
                    cleaned_content = potential_json_parts[-1]
        
        # Try to parse as JSON
        data = json.loads(cleaned_content)
        
        # Validate required fields
        if not isinstance(data, dict) or "message" not in data or "car_ids" not in data:
            raise ValueError("Missing required fields")
        
        # Ensure car_ids is array of numbers
        if not isinstance(data["car_ids"], list):
            data["car_ids"] = []
        else:
            # Convert string IDs to numbers if needed
            fixed_ids = []
            for car_id in data["car_ids"]:
                try:
                    fixed_ids.append(int(car_id))
                except (ValueError, TypeError):
                    continue
            data["car_ids"] = fixed_ids
        
        # Ensure message is string
        if not isinstance(data["message"], str):
            data["message"] = str(data["message"])
        
        return json.dumps(data)
        
    except (json.JSONDecodeError, ValueError) as e:
        # If JSON parsing fails, try to extract information from the text
        print(f"JSON parsing failed: {e}")
        print(f"Content: {response_content[:200]}...")
        
        # Extract car IDs from text patterns
        car_ids = []
        
        # Look for ID patterns like "(ID:123)" or "ID: 123"
        import re
        id_patterns = [
            r'\(ID:(\d+)\)',  # (ID:123)
            r'ID:\s*(\d+)',   # ID: 123 or ID:123
            r'id:\s*(\d+)',   # id: 123 or id:123 (lowercase)
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, response_content)
            for match in matches:
                try:
                    car_ids.append(int(match))
                except ValueError:
                    continue
        
        # Also look for explicit car ID lists like "[290, 280, 287, ...]"
        # Pattern for finding arrays of numbers
        array_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        array_matches = re.findall(array_pattern, response_content)
        for match in array_matches:
            # Split by comma and convert to integers
            try:
                ids_in_array = [int(x.strip()) for x in match.split(',')]
                # If this array is longer than our current car_ids, it's probably the main list
                if len(ids_in_array) > len(car_ids):
                    car_ids = ids_in_array
            except ValueError:
                continue
        
        # Remove duplicates while preserving order
        unique_car_ids = []
        seen = set()
        for car_id in car_ids:
            if car_id not in seen:
                unique_car_ids.append(car_id)
                seen.add(car_id)
        
        # Fallback: try to extract car information from the text
        if "Found" in response_content and "cars" in response_content:
            # Use the entire response as the message, but clean it up
            message = response_content.strip()
            
            # Remove any markdown code block markers
            message = re.sub(r'```json?\s*', '', message)
            message = re.sub(r'```\s*$', '', message)
            
            return json.dumps({
                "message": message,
                "car_ids": unique_car_ids
            })
        
        # Ultimate fallback
        return json.dumps({
            "message": "I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
            "car_ids": []
        })
    except Exception as e:
        print(f"Unexpected error in validate_and_fix_response: {e}")
        return json.dumps({
            "message": "I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
            "car_ids": []
        })

def chat_with_bot(user_input: str, conversation_history: list = None) -> str:
    """
    Function to chat with the bot. Supports conversation history for context.
    
    Args:
        user_input: Current user message
        conversation_history: Optional list of previous messages in format:
                            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        JSON string with AI response
    """
    import time
    import traceback
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"🤖 chat_with_bot attempt {attempt + 1}: {user_input[:50]}...")
            
            # Build message history
            messages = []
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        # Parse assistant message to extract just the text content
                        assistant_content = msg.get("content", "")
                        try:
                            # If it's JSON, extract the message field
                            import json
                            parsed = json.loads(assistant_content)
                            if isinstance(parsed, dict) and "message" in parsed:
                                assistant_content = parsed["message"]
                        except:
                            # If not JSON, use as-is
                            pass
                        messages.append(AIMessage(content=assistant_content))
            
            # Add current user message
            messages.append(HumanMessage(content=user_input))
            print(f"📝 Built message history with {len(messages)} messages")
            
            # Create state with message history
            current_input = {"messages": messages}
            
            # Run the agent
            print("🚀 Invoking AI agent...")
            result = app.invoke(current_input)
            print("✅ AI agent completed successfully")
            
            # Extract the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            
            if ai_messages:
                last_ai_message = ai_messages[-1]
                raw_response = last_ai_message.content or "I apologize, but I couldn't generate a proper response. Please try again."
                print(f"📤 Raw AI response received: {len(raw_response)} characters")
                
                # Validate and fix the response format
                validated_response = validate_and_fix_response(raw_response)
                print(f"✅ Response validated and returning")
                return validated_response
            else:
                print("❌ No AI messages found in result")
                return json.dumps({
                    "message": "Sorry, I couldn't process your request.",
                    "car_ids": []
                })
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error running agent (attempt {attempt + 1}): {error_msg}")
            print(f"📍 Traceback: {traceback.format_exc()}")
            
            # Check if it's a Gemini API error that we should retry
            if "500" in error_msg or "InternalServerError" in error_msg or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's a persistent API error, try a simplified response
                    print("🔄 Max retries reached, falling back to simplified search")
                    return fallback_car_search(user_input)
            else:
                # For other errors, don't retry
                print(f"💥 Non-retryable error, returning error message")
                return json.dumps({
                    "message": f"Sorry, I encountered an error: {error_msg}",
                    "car_ids": []
                })
    
    print("⚠️ All retries exhausted")
    return json.dumps({
        "message": "Sorry, I'm experiencing technical difficulties. Please try again later.",
        "car_ids": []
    })

def fallback_car_search(user_input: str) -> str:
    """
    Enhanced fallback function that directly searches the database without using the LLM.
    Used when the AI agent is experiencing issues.
    """
    try:
        print(f"🚨 Fallback search activated for: {user_input}")
        
        # Simple keyword-based search for common car-related terms
        user_lower = user_input.lower()
        
        # Try to extract basic search criteria
        search_params = {}
        
        # Enhanced keyword detection for family cars
        if any(word in user_lower for word in ["family", "families", "family car", "family vehicle"]):
            search_params["category"] = "SUV"
            print("🔍 Detected family car request, searching SUVs")
        elif any(word in user_lower for word in ["sport", "sports", "sporty", "fast", "performance"]):
            search_params["category"] = "Sports"
            print("🔍 Detected sports car request")
        elif any(word in user_lower for word in ["luxury", "premium", "elegant"]):
            search_params["category"] = "Sedan"
            print("🔍 Detected luxury car request, searching Sedans")
        elif any(word in user_lower for word in ["suv", "crossover", "4x4", "awd"]):
            search_params["category"] = "SUV"
            print("🔍 Detected SUV request")
        elif any(word in user_lower for word in ["sedan"]):
            search_params["category"] = "Sedan"
            print("🔍 Detected sedan request")
        else:
            # Default to showing all cars if no specific category is detected
            print("🔍 No specific category detected, showing all cars")
        
        # Check for budget mentions
        if "budget" in user_lower or "cheap" in user_lower or "affordable" in user_lower:
            search_params["price_max"] = 25000
            print("💰 Budget constraint detected: under $25,000")
        elif "expensive" in user_lower or "high-end" in user_lower:
            search_params["price_min"] = 50000
            print("💰 High-end constraint detected: over $50,000")
        
        # Sort by price ascending to show most affordable first
        search_params["sort_by"] = "price"
        search_params["ascending"] = True
        
        print(f"🔍 Search parameters: {search_params}")
        
        # Perform a basic database search using the unified search_cars function
        result = search_cars.invoke(search_params)
        result_data = json.loads(result)
        
        if result_data.get("success") and result_data.get("data"):
            cars_for_display = result_data["data"]  # Limited cars for AI context
            all_car_ids = result_data.get("all_car_ids", [])  # All matching IDs
            total_count = result_data.get("total_count", 0)
            
            # Fallback: if all_car_ids is missing, extract from data
            if not all_car_ids:
                all_car_ids = [car.get("id") for car in cars_for_display if car.get("id")]
            
            # Create a more user-friendly message based on the search type
            if "family" in user_lower:
                message_prefix = f"Found {total_count} family-friendly SUVs. Here are the top {len(cars_for_display)} options:"
            elif search_params.get("category") == "Sports":
                message_prefix = f"Found {total_count} sports cars. Here are the top {len(cars_for_display)} options:"
            else:
                message_prefix = f"Found {total_count} cars that might interest you. Here are the top {len(cars_for_display)}:"
            
            response = {
                "message": message_prefix,
                "car_ids": all_car_ids  # Use ALL matching IDs for frontend
            }
            
            # Format top cars for display in message
            car_list = []
            for i, car in enumerate(cars_for_display, 1):
                price = car.get('price', 'N/A')
                if isinstance(price, (int, float)):
                    price_str = f"${price:,.0f}"
                else:
                    price_str = f"${price}"
                
                car_line = f" {i}. {car.get('make', 'Unknown')} {car.get('model', 'Unknown')} {car.get('year', 'N/A')} – {price_str} (ID:{car.get('id', 'N/A')}) – {car.get('condition', 'Used')}, {car.get('color', 'Unknown')}"
                car_list.append(car_line)
            
            response["message"] += " " + " ".join(car_list)
            
            print(f"✅ Fallback search successful: {total_count} cars found")
            return json.dumps(response)
        else:
            print("❌ Fallback search failed: no results from database")
            return json.dumps({
                "message": "I couldn't find any cars matching your criteria at the moment. Please try with different search terms or contact support.",
                "car_ids": []
            })
            
    except Exception as e:
        print(f"❌ Fallback search error: {e}")
        import traceback
        print(f"📍 Traceback: {traceback.format_exc()}")
        return json.dumps({
            "message": "I'm currently experiencing technical difficulties. Please try again later or contact support if the problem persists.",
            "car_ids": []
        })

# Interactive chat function for testing (kept for local development)
def start_interactive_chat():
    """Start an interactive chat session for local testing."""
    print("🚗 Welcome to Enhanced Car Search Assistant!")
    print("Type 'quit' to exit or enter your message.")
    print("Note: Each message is processed independently (stateless mode)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using the Car Search Assistant! Goodbye! 👋")
            break
        elif not user_input:
            print("Please enter a message.")
            continue
        
        print("Bot: ", end="", flush=True)
        response = chat_with_bot(user_input)
        print(response)

# Example usage for testing
if __name__ == "__main__":
    # Test the enhanced search functions
    print("Testing enhanced car search functions...")
    
    try:
        # Example 1: Similar cars search
        print("\n1. Testing get_similar_cars:")
        result1 = get_similar_cars.invoke({
            "car_id": 1,
            "similarity_factors": "make,category,price_range",
            "limit": 3
        })
        print(result1)
        
        # Example 2: Budget range search
        print("\n2. Testing get_cars_by_budget_range:")
        result2 = get_cars_by_budget_range.invoke({
            "budget_min": 20000,
            "budget_max": 40000,
            "category": "sedan",
            "condition": "Used"
        })
        print(result2)
        
        # Example 3: Market insights
        print("\n3. Testing get_market_insights:")
        result3 = get_market_insights.invoke({
            "make": "Toyota",
            "condition": "Used"
        })
        print(result3)
        
        # Example 4: Feature-based search
        print("\n4. Testing get_cars_by_features:")
        result4 = get_cars_by_features.invoke({
            "required_features": "bluetooth,leather",
            "preferred_features": "sunroof,navigation",
            "price_max": 35000
        })
        print(result4)
        
        # Example 5: Recently added cars
        print("\n5. Testing get_recently_added_cars:")
        result5 = get_recently_added_cars.invoke({
            "days_back": 14,
            "category": "suv"
        })
        print(result5)
        
        # Example 6: Popular cars
        print("\n6. Testing get_popular_cars:")
        result6 = get_popular_cars.invoke({
            "sort_by": "views",
            "category": "sedan",
            "limit": 5
        })
        print(result6)
        
        # Example 7: Advanced search
        print("\n7. Testing search_cars_advanced:")
        result7 = search_cars_advanced.invoke({
            "filters": '{"make": "BMW", "year_min": 2018, "price_max": 50000, "condition": "Used"}',
            "search_term": "luxury",
            "sort_options": "price:asc"
        })
        print(result7)
        
        # Example 8: Web search
        print("\n8. Testing search_web_for_car_info:")
        result8 = search_web_for_car_info.invoke({
            "query": "best family cars"
        })
        print(result8)
        
        # Example 9: Demonstrating web search + database search workflow
        print("\n9. Testing combined web search + database search workflow:")
        
        # First, search web for family cars
        web_result = search_web_for_car_info.invoke({
            "query": "family cars with good safety ratings"
        })
        web_data = json.loads(web_result)
        
        if web_data.get("success") and web_data.get("extracted_insights"):
            suggested_categories = web_data["extracted_insights"].get("suggested_categories", [])
            print(f"Web search suggested categories: {suggested_categories}")
            
            # Then search database using the suggested categories
            if suggested_categories:
                for category in suggested_categories[:2]:  # Test first 2 categories
                    print(f"\nSearching database for {category} cars:")
                    db_result = get_cars_data_eq.invoke({
                        "category": category,
                        "price_max": 40000,
                        "condition": "Used"
                    })
                    db_data = json.loads(db_result)
                    if db_data.get("success"):
                        print(f"Found {db_data.get('total_count', 0)} {category} cars")
        
        print("\n✅ All enhanced function tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Note: This might be due to Supabase connection issues or missing data.")
    
    # Start interactive chat
    print("\n" + "="*50)
    print("Starting interactive chat with enhanced features...")
    start_interactive_chat()