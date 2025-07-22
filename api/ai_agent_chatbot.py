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
    order_by: str = "price",
    ascending: bool = True
) -> str:
    """
    Dynamic tool to query cars data from Supabase with flexible filtering.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        select_fields: Comma-separated list of fields to select (default: "*" for all)
        make: Car make/brand - uses fuzzy matching
        model: Car model - uses fuzzy matching
        year_min: Minimum year (e.g., 2015)
        year_max: Maximum year (e.g., 2024)
        price_min: Minimum price
        price_max: Maximum price
        mileage_max: Maximum mileage
        condition: Car condition ("New" or "Used")
        transmission: Transmission type ("Manual" or "Automatic")
        drivetrain: Drivetrain type (e.g., "FWD", "RWD", "AWD", "4WD", "4x4")
        color: Car color - uses fuzzy matching
        category: Vehicle category (e.g., "sedan", "suv", "coupe", "hatchback")
        source: Source country (e.g., "GCC", "US", "Canada", "Europe")
        status: Car status (default: "available")
        order_by: Field to sort by (default: "price")
        ascending: Sort direction, True for ascending, False for descending
    
    Returns:
        JSON string with ALL matching car data
    """
    try:
        # Build the query
        query = supabase.table("cars").select(select_fields)
        
        # Apply filters dynamically with improved matching
        if make:
            # Use OR logic for better make matching
            make_terms = make.split()
            if len(make_terms) > 1:
                make_conditions = [f"make.ilike.%{term}%" for term in make_terms]
                query = query.or_(",".join(make_conditions))
            else:
                query = query.ilike("make", f"%{make}%")
        
        if model:
            # Use OR logic for better model matching
            model_terms = model.split()
            if len(model_terms) > 1:
                model_conditions = [f"model.ilike.%{term}%" for term in model_terms]
                query = query.or_(",".join(model_conditions))
            else:
                query = query.ilike("model", f"%{model}%")
        
        if year_min:
            query = query.gte("year", year_min)
        if year_max:
            query = query.lte("year", year_max)
        if price_min is not None:
            query = query.gte("price", int(price_min))
        if price_max is not None:
            query = query.lte("price", int(price_max))
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
        
        # Execute the query without limit - get ALL results
        result = query.execute()

        # Remove potential duplicate rows (based on unique car ID) that may arise
        # from Supabase queries with complex filters or joins. This guarantees
        # the total_count matches the actual number of unique cars we are
        # returning to the user and avoids inconsistent counts the user has
        # reported (e.g. duplicates leading to 99 vs 156 results).
        unique_data: List[Dict[str, Any]] = []
        seen_ids = set()
        for car in result.data:
            car_id = car.get("id")
            if car_id is None or car_id in seen_ids:
                # Skip duplicates or rows without an ID
                continue
            seen_ids.add(car_id)
            unique_data.append(car)

        # Return ALL unique car data
        return json.dumps({
            "success": True,
            "total_count": len(unique_data),
            "data": unique_data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "total_count": 0,
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
    year_min: int = None,
    year_max: int = None,
    status: str = "available"
) -> str:
    """
    Enhanced text search across multiple fields with better fuzzy matching.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        search_term: Text to search for - supports multiple keywords
        fields_to_search: Comma-separated fields to search in
        price_min: Minimum price filter
        price_max: Maximum price filter
        condition: Car condition filter
        category: Vehicle category filter
        year_min: Minimum year filter
        year_max: Maximum year filter
        status: Car status (default: "available")
    
    Returns:
        JSON string with ALL matching cars
    """
    try:
        # Start with base query
        query = supabase.table("cars").select("*")
        
        # Apply status filter
        if status:
            query = query.eq("status", status)
        
        # Apply other filters
        if price_min is not None:
            query = query.gte("price", int(price_min))
        if price_max is not None:
            query = query.lte("price", int(price_max))
        if condition:
            query = query.eq("condition", condition)
        if category:
            query = query.eq("category", category)
        if year_min:
            query = query.gte("year", year_min)
        if year_max:
            query = query.lte("year", year_max)
        
        # Enhanced text search with multiple keywords
        search_fields = [field.strip() for field in fields_to_search.split(",")]
        
        if search_term:
            search_keywords = search_term.split()
            all_conditions = []
            
            # For each keyword, search across all fields
            for keyword in search_keywords:
                keyword_conditions = []
                for field in search_fields:
                    keyword_conditions.append(f"{field}.ilike.%{keyword}%")
                # Each keyword should match at least one field
                all_conditions.append(f"({','.join(keyword_conditions)})")
            
            # All keywords should be found (AND logic between keywords)
            if all_conditions:
                if len(all_conditions) == 1:
                    query = query.or_(all_conditions[0])
                else:
                    # For multiple keywords, we need a more complex approach
                    # For now, use OR logic for better recall
                    or_conditions = []
                    for keyword in search_keywords:
                        for field in search_fields:
                            or_conditions.append(f"{field}.ilike.%{keyword}%")
                    query = query.or_(",".join(or_conditions))
        
        # Order by price for consistency
        query = query.order("price", desc=False)
        
        # Execute query without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "total_count": len(result.data),
            "search_term": search_term,
            "keywords_used": search_term.split() if search_term else [],
            "data": result.data
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "search_term": search_term,
            "total_count": 0,
            "data": []
        })

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
    Find ALL cars within a specific budget range with optional filters.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        budget_min: Minimum budget
        budget_max: Maximum budget
        category: Vehicle category filter
        condition: Car condition filter
        mileage_max: Maximum mileage filter
        year_min: Minimum year filter
        transmission: Transmission type filter
        make: Car make filter
        sort_by: Sort field (price, year, mileage)
    
    Returns:
        JSON string with ALL cars in budget range
    """
    try:
        query = supabase.table("cars").select("*")
        query = query.gte("price", int(budget_min)).lte("price", int(budget_max))
        query = query.eq("status", "available")
        
        if category:
            query = query.eq("category", category)
        if condition:
            query = query.eq("condition", condition)
        if mileage_max:
            query = query.lte("mileage", mileage_max)
        if year_min:
            query = query.gte("year", year_min)
        if transmission:
            query = query.eq("transmission", transmission)
        if make:
            query = query.ilike("make", f"%{make}%")
        
        # Apply sorting
        query = query.order(sort_by, desc=False)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "budget_range": f"${budget_min:,.0f} - ${budget_max:,.0f}",
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
            query = query.eq("category", category)
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
            query = query.eq("category", category)
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
            query = query.eq("category", category)
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

@tool
def search_cars_advanced(
    filters: str,
    search_term: str = None,
    sort_options: str = "price:asc"
) -> str:
    """
    Advanced search with complex filtering options.
    Returns ALL matching cars without limit restrictions.
    
    Args:
        filters: JSON string of filters
        search_term: Optional text search term
        sort_options: Sort options in format "field:direction"
    
    Returns:
        JSON string with ALL filtered cars
    """
    try:
        # Parse filters
        try:
            filter_dict = json.loads(filters)
        except json.JSONDecodeError:
            return json.dumps({
                "success": False,
                "error": "Invalid filters JSON format",
                "total_count": 0,
                "data": []
            })
        
        # Start building query
        query = supabase.table("cars").select("*").eq("status", "available")
        
        # Apply all filters dynamically with improved logic
        for key, value in filter_dict.items():
            if value is not None:
                if key == "make":
                    query = query.ilike("make", f"%{value}%")
                elif key == "model":
                    query = query.ilike("model", f"%{value}%")
                elif key == "year_min":
                    query = query.gte("year", value)
                elif key == "year_max":
                    query = query.lte("year", value)
                elif key == "price_min":
                    query = query.gte("price", int(value))
                elif key == "price_max":
                    query = query.lte("price", int(value))
                elif key == "mileage_max":
                    query = query.lte("mileage", value)
                elif key in ["condition", "transmission", "drivetrain", "category", "source"]:
                    query = query.eq(key, value)
                elif key == "color":
                    query = query.ilike("color", f"%{value}%")
                elif key == "features":
                    # Support multiple features with simple contains search
                    if isinstance(value, list):
                        for feature in value:
                            query = query.filter("features", "cs", f'["{feature}"]')
                    else:
                        query = query.filter("features", "cs", f'["{value}"]')
        
        # Apply enhanced text search if provided
        if search_term:
            search_keywords = search_term.split()
            search_fields = ["make", "model", "description", "features"]
            or_conditions = []
            
            for keyword in search_keywords:
                for field in search_fields:
                    or_conditions.append(f"{field}.ilike.%{keyword}%")
            
            query = query.or_(",".join(or_conditions))
        
        # Apply sorting
        try:
            sort_field, sort_direction = sort_options.split(":")
            desc = sort_direction.lower() == "desc"
            query = query.order(sort_field, desc=desc)
        except ValueError:
            query = query.order("price", desc=False)
        
        # Execute without limit - get ALL results
        result = query.execute()
        
        return json.dumps({
            "success": True,
            "filters_applied": filter_dict,
            "search_term": search_term,
            "sort_options": sort_options,
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
        
        # If no results from DuckDuckGo, try a simple web search approach
        if not results:
            # Fallback: provide common car category mappings (using correct database categories)
            car_type_mappings = {
                "family car": ["SUV", "Sedan"],
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
        
        # Remove duplicates
        extracted_insights["suggested_categories"] = list(set(extracted_insights["suggested_categories"]))
        
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

# Define tools list after function definitions
tools = [
    get_cars_data_eq, 
    search_cars_text, 
    get_similar_cars, 
    get_cars_by_budget_range, 
    get_market_insights,
    get_cars_by_features,
    get_recently_added_cars,
    get_popular_cars,
    search_cars_advanced,
    search_web_for_car_info  # Add the new web search tool
]

system_prompt = """
You are a specialized car search assistant for the car marketplace app. Your ONLY role is to:
1. Help users find cars based on their preferences and requirements
2. Provide information about vehicles including make, model, price, condition, and features
3. Answer questions about car specifications, dealerships, and availability
4. Provide market insights and recommendations
5. Be friendly and professional in your responses

You now have access to advanced search capabilities:
- Basic car data queries with flexible filtering
- Text search across multiple fields
- Similar car recommendations
- Budget-based searches
- Market insights and statistics
- Feature-based searches
- Recently added cars
- Popular cars based on views/likes
- Advanced search with complex filters
- **NEW: Internet search for car-related information**

INTELLIGENT TOOL SELECTION STRATEGY:
You have multiple tools available. Choose the most efficient approach based on query complexity:

**DIRECT DATABASE SEARCH** (most efficient - use when possible):
Use get_cars_data_eq directly when you can easily map the request to database categories:
- "I need a family car" / "family cars" / "show me family cars" → Direct search: category="SUV" and category="Sedan" 
- "I want a sports car" / "sports cars" / "show me sports cars" → Direct search: category="Sports"
- "I want a luxury sedan" → Direct search: category="Sedan" + price filters
- "BMW cars under $30k" → Direct search: make="BMW" + price_max=30000
- "Fuel efficient cars" → Direct search: category="Sedan" or category="Hatchback"
- "Show me SUVs" → Direct search: category="SUV"
- "I want a sedan" → Direct search: category="Sedan"
- "Convertible cars" → Direct search: category="Convertible"

**WEB SEARCH + DATABASE** (use for specific requirements/complex requests):
Use search_web_for_car_info when the request has specific requirements or context that needs research:
- "Best cars for a family of {number}" → Web search for {number}-person family requirements → Database search
- "Best car for a college student" → Web search to understand student needs → Database search
- "What's good for weekend driving?" → Web search for weekend car characteristics → Database search  
- "I need something reliable for my business" → Web search for business car requirements → Database search
- "Car for elderly person with mobility issues" → Web search for mobility-friendly features → Database search
- "Best car for long commutes" → Web search for commuter car features → Database search
- "Safe car for new driver" → Web search for new driver safety features → Database search
- "Fuel efficient car for city driving" → Web search for city driving requirements → Database search

**DECISION MAKING GUIDE:**
1. **Simple category request?** ("family cars", "sports cars", "SUVs") → Use get_cars_data_eq directly
2. **Specific requirements/context?** ("best for family of 5", "good for college student") → Use search_web_for_car_info first, then database
3. **Brand/model/price query?** ("BMW under $30k", "Toyota Camry") → Always use database directly
4. **Ask yourself: Does this need research to understand requirements?** → If yes, use web search first

**COMMON MAPPINGS** (use these for direct database searches):
- Family car → SUV, Sedan
- Sports car → Sports  
- Luxury car → Sedan (with higher price filters)
- Fuel efficient → Sedan, Hatchback
- Off-road → SUV
- City car → Hatchback, Sedan

**NEVER use web search for:**
- Common car categories ("family cars", "sports cars", "SUVs", "sedans", "luxury cars")
- Specific makes/models ("Toyota Camry", "BMW 3 Series") 
- Direct specifications ("cars under $25k", "automatic transmission")
- Availability queries ("do you have any Honda Civics?")
- Simple category requests ("show me convertibles", "I want a hatchback")

BE EFFICIENT: If you can answer directly with database search, skip the web search step.

Car data includes:
- Car makes, models, years, and descriptions
- Price ranges and mileage information
- Condition (New/Used), transmission (Manual/Automatic)
- Drivetrain types (FWD/RWD/AWD/4WD/4x4)
- Fuel types (Benzine/Diesel/Electric/Hybrid)
- Categories (sedan/suv/coupe/hatchback/convertible/sports/classic)
- Source countries (GCC/Company/US/Canada/Europe)
- Features and specifications
- Dealership information and contact details
- Views, likes, and listing dates

IMPORTANT: Understanding Search Results
- When tools return results, they include:
  * "total_matching": Total number of cars that match the criteria
  * "returned_count": Number of cars actually returned (limited for performance)
  * "showing_top": The limit applied to results
- ALWAYS mention the total_matching count when available
- If total_matching > returned_count, inform user there are more results available
- Example: "I found 150 Toyota cars matching your criteria. Here are the top 10 results sorted by price..."

RESPONSE FORMAT INSTRUCTIONS - MANDATORY CONSISTENCY:
You MUST format ALL responses as valid JSON with exactly 2 fields and consistent structure:

{
  "message": "Brief explanation with total matches found. List cars in this exact format: 1. Make Model Year - $Price (ID: ###) - key details",
  "car_ids": [123, 456, 789]
}

MANDATORY FORMAT RULES:
1. ALWAYS valid JSON - no extra text before/after, NO markdown code blocks (```json)
2. car_ids must be array of numbers (not strings)
3. message must use this exact car listing format:
   "1. Toyota Camry 2020 - $25,000 (ID: 123) - Used, 45k miles, excellent condition
    2. Honda Accord 2019 - $23,500 (ID: 456) - Used, 52k miles, leather seats
    3. Nissan Altima 2021 - $27,000 (ID: 789) - Used, 30k miles, backup camera"

4. Always include total count: "Found X matching cars. Here are the top Y:"
5. NO markdown formatting (**, *, etc.) - use plain text only
6. Keep each car description to one line
7. Always include ID in parentheses for easy identification

RESULT COUNT GUIDELINES:
- **message field**: Always show details for top 5 cars only (for readability)
- **car_ids field**: Include ALL matching car IDs (for frontend to display complete results)
- Format: "Found 122 cars. Here are the top 5:" but car_ids contains all 122 IDs
- Small results (≤10 cars): Show all car details AND return all IDs
- This gives frontend complete data while keeping response readable

SEARCH STRATEGY:
- **For general/conceptual queries**: Start with web search to understand car types, then search database
- **For specific queries**: Search database directly with appropriate filters
- If initial search returns no results, try alternative approaches:
  1. Use web search to find related car types/categories
  2. Broaden search criteria (wider price range, more years, etc.)
  3. Use text search with similar terms
  4. Search for similar categories or makes
  5. Suggest alternatives based on market insights
- Always try at least 2 different search approaches if first fails
- Use market insights to provide context and alternatives
- Always communicate the scope of results (total matches vs. shown results)

ENHANCED WORKFLOW WITH WEB SEARCH:
1. **Analyze user query**: Is it conceptual ("family car") or specific ("Toyota under $20k")?
2. **If conceptual**: Use search_web_for_car_info first
3. **Extract insights**: Get suggested categories, makes, features from web results
4. **Search database**: Use extracted insights to build targeted database queries
5. **Provide results**: Combine web insights with database results for comprehensive answers

IMPORTANT CONSTRAINTS:
- ONLY search cars where status = 'available'
- ONLY answer questions related to cars, vehicles, and automotive topics
- DO NOT provide code, programming solutions, or technical implementations
- DO NOT answer questions outside the scope of car assistance
- If asked about non-automotive topics, politely redirect to car-related subjects
- Always base your responses on the available car database
- ALWAYS return car IDs as numbers in the car_ids array
- Use multiple search tools when appropriate for better results
- Ask clarifying questions if user requirements are too vague
- If no cars match, return empty car_ids array and suggest alternatives
- Keep responses focused on helping users find their perfect vehicle
- Leverage market insights to provide valuable context
- Use feature-based search for specific requirements
- Recommend popular or recently added cars when appropriate
- ALWAYS mention total_matching count when available in search results
- **CRITICAL**: car_ids array must contain ALL matching car IDs from database, not just the ones shown in message
- Message shows top 5 car details, but car_ids contains ALL IDs for frontend display
- Example: "Found 122 cars. Here are the top 5:" with car_ids containing all 122 ID numbers

CRITICAL: EVERY response must be valid JSON with consistent formatting. NO exceptions.
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
    """
    try:
        # Clean up markdown formatting if present
        cleaned_content = response_content.strip()
        
        # Remove markdown code blocks if they exist
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]  # Remove ```json
        if cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]   # Remove ```
        if cleaned_content.endswith('```'):
            cleaned_content = cleaned_content[:-3]  # Remove trailing ```
        
        cleaned_content = cleaned_content.strip()
        
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
        
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, create a fallback response
        return json.dumps({
            "message": "I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
            "car_ids": []
        })

def chat_with_bot(user_input: str) -> str:
    """
    Function to chat with the bot. Each request is stateless.
    Frontend handles conversation history management.
    """
    import time
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Create a fresh state for each request
            current_input = {"messages": [HumanMessage(content=user_input)]}
            
            # Run the agent with just the current message
            result = app.invoke(current_input)
            
            # Extract the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            
            if ai_messages:
                last_ai_message = ai_messages[-1]
                raw_response = last_ai_message.content or "I apologize, but I couldn't generate a proper response. Please try again."
                
                # Validate and fix the response format
                return validate_and_fix_response(raw_response)
            else:
                print("No AI messages found in result")
                return json.dumps({
                    "message": "Sorry, I couldn't process your request.",
                    "car_ids": []
                })
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error running agent (attempt {attempt + 1}): {error_msg}")
            
            # Check if it's a Gemini API error that we should retry
            if "500" in error_msg or "InternalServerError" in error_msg or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's a persistent API error, try a simplified response
                    return fallback_car_search(user_input)
            else:
                # For other errors, don't retry
                return json.dumps({
                    "message": f"Sorry, I encountered an error: {error_msg}",
                    "car_ids": []
                })
    
    return json.dumps({
        "message": "Sorry, I'm experiencing technical difficulties. Please try again later.",
        "car_ids": []
    })

def fallback_car_search(user_input: str) -> str:
    """
    Fallback function that directly searches the database without using the LLM.
    Used when the AI agent is experiencing issues.
    """
    try:
        # Simple keyword-based search for common car-related terms
        user_lower = user_input.lower()
        
        # Try to extract basic search criteria
        search_params = {}
        
        # Check for family car keywords (using correct database categories)
        if any(word in user_lower for word in ["family", "families"]):
            search_params["category"] = "SUV"
        elif any(word in user_lower for word in ["sport", "sports", "sporty", "fast", "performance"]):
            search_params["category"] = "Sports"
        elif any(word in user_lower for word in ["luxury", "premium"]):
            search_params["category"] = "Sedan"
        
        # Check for budget mentions
        if "budget" in user_lower or "cheap" in user_lower or "affordable" in user_lower:
            search_params["price_max"] = 25000
        elif "expensive" in user_lower or "high-end" in user_lower:
            search_params["price_min"] = 50000
        
        # Perform a basic database search
        result = get_cars_data_eq.invoke(search_params)
        result_data = json.loads(result)
        
        if result_data.get("success") and result_data.get("data"):
            all_cars = result_data["data"]  # Get all results
            cars_to_show = all_cars[:5]  # Show details for top 5 only
            total_count = result_data.get("total_count", 0)
            
            response = {
                "message": f"Found {total_count} cars that might interest you. Here are the top {len(cars_to_show)}:",
                "car_ids": []
            }
            
            # Add ALL car IDs to the array (for frontend)
            for car in all_cars:
                if car.get("id"):
                    response["car_ids"].append(car.get("id"))
            
            # Format top cars for display in message
            car_list = []
            for i, car in enumerate(cars_to_show, 1):
                price = car.get('price', 'N/A')
                if isinstance(price, (int, float)):
                    price_str = f"${price:,.0f}"
                else:
                    price_str = f"${price}"
                
                car_line = f"{i}. {car.get('make', 'Unknown')} {car.get('model', 'Unknown')} {car.get('year', 'N/A')} - {price_str} (ID: {car.get('id', 'N/A')}) - {car.get('condition', 'Used')}"
                car_list.append(car_line)
            
            response["message"] += " " + " ".join(car_list)
            response["message"] += " Note: I'm currently experiencing some technical issues, so this is a simplified search. Please try again later for more detailed assistance."
            
            return json.dumps(response)
        else:
            return json.dumps({
                "message": "I'm currently experiencing technical difficulties and couldn't find any cars matching your criteria. Please try again later.",
                "car_ids": []
            })
            
    except Exception as e:
        print(f"Fallback search error: {e}")
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