from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, NotRequired
from urllib.parse import quote

import json
import os
import re
import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from supabase import Client, create_client
from pathlib import Path


# Load .env from the ai_python directory (parent of api/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# Support both naming conventions: SUPABASE_* and EXPO_PUBLIC_SUPABASE_*
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("EXPO_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("EXPO_PUBLIC_SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        f"Missing Supabase credentials; set SUPABASE_URL and SUPABASE_KEY in your environment. "
        f"Looked for .env at: {_env_path}"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


DEFAULT_RESULT_LIMIT = 200
DEFAULT_CAR_COLUMNS = (
    "id,make,model,year,price,mileage,condition,category,type,color,drivetrain,"
    "transmission,source,status,features"
)
DEFAULT_SEARCH_FIELDS = "make,model,description,features"

KNOWN_MAKES = [
    "acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "dodge",
    "ford", "gmc", "honda", "hyundai", "infiniti", "jeep", "kia",
    "land rover", "range rover", "lexus", "lincoln", "mazda", "mercedes",
    "mercedes-benz", "mercedes benz", "mini", "mitsubishi", "nissan",
    "porsche", "ram", "subaru", "tesla", "toyota", "volkswagen", "volvo"
]

MAKE_NORMALIZATION = {
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "merc": "Mercedes-Benz",
    "vw": "Volkswagen",
    "land rover": "Land Rover",
    "range rover": "Land Rover",
}

CATEGORY_KEYWORDS = {
    "suv": "SUV",
    "crossover": "SUV",
    "sedan": "Sedan",
    "hatchback": "Hatchback",
    "wagon": "Hatchback",
    "convertible": "Convertible",
    "coupe": "Coupe",
    "sports car": "Sports",
    "sport car": "Sports",
    "sporty": "Sports",
    "performance": "Sports",
    "pickup": "Truck",
    "truck": "Truck",
    "van": "Van",
    "family": "SUV",
}

TYPE_KEYWORDS = {
    "electric": "Electric",
    "ev": "Electric",
    "hybrid": "Hybrid",
    "plug-in": "Hybrid",
    "diesel": "Diesel",
    "gas": "Gasoline",
    "gasoline": "Gasoline",
}

LIFESTYLE_KEYWORDS = [
    "student", "college", "family", "commuter", "road trip", "off-road",
    "luxury", "business", "weekend", "track", "adventure", "snow", "mountain"
]

COLOR_KEYWORDS = [
    "black", "white", "silver", "gray", "grey", "blue", "red", "green",
    "yellow", "orange", "brown", "beige", "gold", "purple"
]

CONDITION_KEYWORDS = {
    "new": "New",
    "brand new": "New",
    "used": "Used",
    "certified": "Certified",
    "cpo": "Certified",
}

TRANSMISSION_KEYWORDS = {
    "automatic": "Automatic",
    "auto": "Automatic",
    "manual": "Manual",
    "stick": "Manual",
}

DRIVETRAIN_KEYWORDS = {
    "awd": "AWD",
    "4x4": "AWD",
    "4wd": "AWD",
    "fwd": "FWD",
    "front wheel": "FWD",
    "rwd": "RWD",
    "rear wheel": "RWD",
}


PRICE_RANGE_RE = re.compile(r"\$?\s*(\d[\d,\.]*\s*[km]?)\s*(?:-|to|–|—)\s*\$?\s*(\d[\d,\.]*\s*[km]?)", re.I)
PRICE_MAX_RE = re.compile(r"(?:under|below|less than|up to|upto|max)\s*\$?\s*(\d[\d,\.]*\s*[km]?)", re.I)
PRICE_MIN_RE = re.compile(r"(?:over|above|more than|at least|min)\s*\$?\s*(\d[\d,\.]*\s*[km]?)", re.I)
PRICE_AROUND_RE = re.compile(r"(?:around|about|close to|near|approximately|~)\s*\$?\s*(\d[\d,\.]*\s*[km]?)", re.I)
PRICE_SINGLE_RE = re.compile(r"\$(\d[\d,\.]*\s*[km]?)", re.I)
PRICE_BUDGET_RE = re.compile(r"budget\s+(?:is\s+)?(?:of\s+)?\$?\s*(\d[\d,\.]*\s*[km]?)", re.I)

YEAR_RANGE_RE = re.compile(r"(19|20)\d{2}\s*(?:-|to|through|–|—)\s*(19|20)\d{2}")
YEAR_SINGLE_RE = re.compile(r"\b(19|20)\d{2}\b")

MILEAGE_RE = re.compile(r"(?:under|below|less than)\s*(\d[\d,\.]*\s*[km]?)\s*miles", re.I)


def _parse_amount(raw_value: str) -> Optional[int]:
    if not raw_value:
        return None
    cleaned = raw_value.strip().lower().replace(",", "")
    multiplier = 1
    if cleaned.endswith("k"):
        multiplier = 1000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("m"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return int(value * multiplier)


def _extract_price_filters(text_lower: str) -> Dict[str, int]:
    filters: Dict[str, int] = {}
    range_match = PRICE_RANGE_RE.search(text_lower)
    if range_match:
        price_min = _parse_amount(range_match.group(1))
        price_max = _parse_amount(range_match.group(2))
        if price_min:
            filters["price_min"] = price_min
        if price_max:
            filters["price_max"] = price_max
        return filters

    # Check for "around X" / "close to X" patterns - set a range of ±20%
    around_match = PRICE_AROUND_RE.search(text_lower)
    if around_match:
        target = _parse_amount(around_match.group(1))
        if target:
            # Create a range around the target price (±20%)
            filters["price_min"] = int(target * 0.8)
            filters["price_max"] = int(target * 1.2)
            return filters

    # Check for "budget X" pattern - treat as max price
    budget_match = PRICE_BUDGET_RE.search(text_lower)
    if budget_match:
        budget = _parse_amount(budget_match.group(1))
        if budget:
            # If they say "budget around 20k", use around logic
            if "around" in text_lower or "close" in text_lower or "about" in text_lower:
                filters["price_min"] = int(budget * 0.8)
                filters["price_max"] = int(budget * 1.2)
            else:
                filters["price_max"] = budget
            return filters

    max_match = PRICE_MAX_RE.search(text_lower)
    if max_match:
        price_max = _parse_amount(max_match.group(1))
        if price_max:
            filters["price_max"] = price_max

    min_match = PRICE_MIN_RE.search(text_lower)
    if min_match:
        price_min = _parse_amount(min_match.group(1))
        if price_min:
            filters.setdefault("price_min", price_min)

    if not filters:
        single_match = PRICE_SINGLE_RE.search(text_lower)
        if single_match:
            amount = _parse_amount(single_match.group(1))
            if amount:
                filters["price_max"] = amount
    return filters


def _extract_year_filters(text_lower: str) -> Dict[str, int]:
    filters: Dict[str, int] = {}
    range_match = YEAR_RANGE_RE.search(text_lower)
    if range_match:
        years = [int(s) for s in re.findall(r"(19|20)\d{2}", range_match.group(0))]
        if len(years) >= 2:
            filters["year_min"] = min(years)
            filters["year_max"] = max(years)
            return filters

    years = [int(y) for y in YEAR_SINGLE_RE.findall(text_lower)]
    if years:
        filters["year_min"] = min(years)
        filters["year_max"] = max(years)
    return filters


def _extract_mileage_filter(text_lower: str) -> Optional[int]:
    match = MILEAGE_RE.search(text_lower)
    if match:
        amount = _parse_amount(match.group(1))
        return amount
    return None


def extract_explicit_filters(user_text: str) -> Dict[str, Any]:
    if not user_text:
        return {}
    text_lower = user_text.lower()
    filters: Dict[str, Any] = {}

    filters.update(_extract_price_filters(text_lower))
    filters.update(_extract_year_filters(text_lower))
    mileage_cap = _extract_mileage_filter(text_lower)
    if mileage_cap:
        filters["mileage_max"] = mileage_cap

    for color in COLOR_KEYWORDS:
        if f" {color} " in f" {text_lower} ":
            filters["color"] = color.title()
            break

    for keyword, condition in CONDITION_KEYWORDS.items():
        if keyword in text_lower:
            filters["condition"] = condition
            break

    for keyword, transmission in TRANSMISSION_KEYWORDS.items():
        if keyword in text_lower:
            filters["transmission"] = transmission
            break

    for keyword, drivetrain in DRIVETRAIN_KEYWORDS.items():
        if keyword in text_lower:
            filters["drivetrain"] = drivetrain
            break

    for make in KNOWN_MAKES:
        if make in text_lower:
            normalized = MAKE_NORMALIZATION.get(make, make.title())
            filters["make"] = normalized
            break

    tokens = re.findall(r"[a-z0-9\-]+", text_lower)
    for idx, token in enumerate(tokens[:-1]):
        if token in KNOWN_MAKES:
            candidate = tokens[idx + 1]
            if candidate not in KNOWN_MAKES and 1 < len(candidate) <= 6:
                if any(ch.isdigit() for ch in candidate):
                    filters["model"] = candidate.replace("-", " ").upper()
                else:
                    filters["model"] = candidate.title()
                break

    if "model" not in filters:
        model_match = re.search(r"\b([a-z]{1,2}\d{1,3})\b", text_lower)
        if model_match and model_match.group(1) not in KNOWN_MAKES:
            filters["model"] = model_match.group(1).upper()

    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in text_lower:
            filters["category"] = category
            break

    for keyword, fuel_type in TYPE_KEYWORDS.items():
        if keyword in text_lower:
            filters["type"] = fuel_type
            break

    # Detect exclusion patterns for fuel types
    exclude_patterns = [
        (r"(?:no|not|don'?t like|don'?t want|hate|avoid|exclude|without)\s+electric", "Electric"),
        (r"(?:no|not|don'?t like|don'?t want|hate|avoid|exclude|without)\s+hybrid", "Hybrid"),
        (r"(?:no|not|don'?t like|don'?t want|hate|avoid|exclude|without)\s+diesel", "Diesel"),
        (r"non[- ]?electric", "Electric"),
        (r"non[- ]?hybrid", "Hybrid"),
        (r"(?:he|she|i|we)\s+(?:doesn'?t|don'?t)\s+like\s+electric", "Electric"),
        (r"(?:he|she|i|we)\s+(?:doesn'?t|don'?t)\s+like\s+hybrid", "Hybrid"),
    ]
    for pattern, exclude_type in exclude_patterns:
        if re.search(pattern, text_lower):
            filters["exclude_type"] = exclude_type
            # If we're excluding a type, don't also filter BY that type
            if filters.get("type") == exclude_type:
                del filters["type"]
            break

    return filters


def detect_lifestyle_keywords(user_text: str) -> List[str]:
    text_lower = user_text.lower() if user_text else ""
    return [kw for kw in LIFESTYLE_KEYWORDS if kw in text_lower]


def should_enrich_with_web(user_text: str, explicit_filters: Dict[str, Any]) -> bool:
    return bool(detect_lifestyle_keywords(user_text))


def merge_filters(explicit_filters: Dict[str, Any], web_enrichment: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(explicit_filters or {})
    if not web_enrichment:
        return merged

    insights = web_enrichment.get("extracted_insights") or {}
    categories = insights.get("suggested_categories") or []
    fuel_types = insights.get("suggested_types") or []

    if categories and "category" not in merged:
        merged["category"] = categories[0]
    if fuel_types and "type" not in merged:
        merged["type"] = fuel_types[0]

    return merged


@tool
def get_cars_data_eq(
    select_fields: Optional[str] = DEFAULT_CAR_COLUMNS,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    condition: Optional[str] = None,
    transmission: Optional[str] = None,
    drivetrain: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    type: Optional[str] = None,
    exclude_type: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = "available",
    search_term: Optional[str] = None,
    fields_to_search: str = DEFAULT_SEARCH_FIELDS,
    order_by: str = "price",
    ascending: bool = True,
    limit: Optional[int] = DEFAULT_RESULT_LIMIT,
) -> str:
    """Primary Supabase inventory search with deterministic filtering.
    
    Args:
        exclude_type: Fuel type to exclude (e.g., 'Electric' to exclude electric cars)
    """

    try:
        columns = select_fields or DEFAULT_CAR_COLUMNS
        query = supabase.table("cars").select(columns)

        if make:
            query = query.ilike("make", f"%{make}%")
        if model:
            query = query.ilike("model", f"%{model}%")
        if year_min:
            query = query.gte("year", int(year_min))
        if year_max:
            query = query.lte("year", int(year_max))
        if price_min is not None:
            query = query.gte("price", int(price_min))
        if price_max is not None:
            query = query.lte("price", int(price_max))
        if mileage_max is not None:
            query = query.lte("mileage", int(mileage_max))
        if condition:
            query = query.eq("condition", condition)
        if transmission:
            query = query.eq("transmission", transmission)
        if drivetrain:
            query = query.eq("drivetrain", drivetrain)
        if color:
            query = query.ilike("color", f"%{color}%")
        if category:
            cat_synonyms = {
                "sports car": "Sports",
                "sport car": "Sports",
                "sport": "Sports",
                "crossover": "SUV",
            }
            category_mapped = cat_synonyms.get(category.lower(), category)
            query = query.eq("category", category_mapped)
        if type:
            type_synonyms = {
                "electric": "Electric",
                "hybrid": "Hybrid",
                "gas": "Benzine",
                "gasoline": "Benzine",
                "petrol": "Benzine",
                "benzine": "Benzine",
                "diesel": "Diesel",
            }
            type_mapped = type_synonyms.get(type.lower(), type)
            query = query.eq("type", type_mapped)
        if exclude_type:
            exclude_synonyms = {
                "electric": "Electric",
                "hybrid": "Hybrid",
                "gas": "Benzine",
                "gasoline": "Benzine",
                "petrol": "Benzine",
                "benzine": "Benzine",
                "diesel": "Diesel",
            }
            exclude_mapped = exclude_synonyms.get(exclude_type.lower(), exclude_type)
            query = query.neq("type", exclude_mapped)
        if source:
            query = query.eq("source", source)
        if status:
            query = query.eq("status", status)

        if search_term:
            search_fields = [field.strip() for field in fields_to_search.split(",") if field.strip()]
            if " OR " in search_term.upper():
                or_terms = [term.strip() for term in search_term.upper().split(" OR ") if term.strip()]
                or_conditions = [
                    f"{field}.ilike.%{term}%"
                    for term in or_terms
                    for field in search_fields
                ]
                if or_conditions:
                    query = query.or_(",".join(or_conditions))
            else:
                keywords = [kw for kw in search_term.split() if kw.upper() != "OR"]
                or_conditions = [
                    f"{field}.ilike.%{keyword}%"
                    for keyword in keywords if keyword
                    for field in search_fields
                ]
                if or_conditions:
                    query = query.or_(",".join(or_conditions))

        safe_limit = min(int(limit) if limit else DEFAULT_RESULT_LIMIT, DEFAULT_RESULT_LIMIT)
        query = query.order(order_by or "price", desc=not ascending).limit(safe_limit)
        result = query.execute()

        return json.dumps({
            "success": True,
            "total_count": len(result.data),
            "search_term": search_term,
            "keywords_used": search_term.split() if search_term else [],
            "data": result.data,
        })

    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": str(exc),
            "search_term": search_term,
            "total_count": 0,
            "data": [],
        })


def summarize_inventory(cars: List[Dict[str, Any]], limit: int = 5) -> str:
    if not cars:
        return "No inventory matches were preloaded."
    snippets = []
    for car in cars[:limit]:
        make = car.get("make", "Unknown")
        model = car.get("model", "")
        year = car.get("year", "N/A")
        price = car.get("price")
        price_str = f"${price:,.0f}" if isinstance(price, (int, float)) else str(price)
        snippets.append(f"ID {car.get('id', 'N/A')}: {year} {make} {model} at {price_str}")
    return "Prefetched cars: " + "; ".join(snippets)


def prefetch_inventory(merged_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not merged_filters:
        return None

    allowed_keys = {
        "select_fields",
        "make",
        "model",
        "year_min",
        "year_max",
        "price_min",
        "price_max",
        "mileage_max",
        "condition",
        "transmission",
        "drivetrain",
        "color",
        "category",
        "type",
        "exclude_type",
        "search_term",
        "fields_to_search",
        "order_by",
        "ascending",
    }
    payload = {k: v for k, v in merged_filters.items() if k in allowed_keys and v is not None}
    if not payload:
        return None

    payload.setdefault("status", "available")
    payload.setdefault("order_by", "price")
    payload.setdefault("ascending", True)

    try:
        raw = get_cars_data_eq.invoke(payload)
        result = json.loads(raw)
        if result.get("success"):
            return result
    except Exception as exc:
        print(f"Prefetch inventory failed: {exc}")
    return None


def build_prefetch_context(prefetch_result: Optional[Dict[str, Any]]) -> str:
    if not prefetch_result:
        return "Prefetch status: skipped (insufficient filters)."
    cars = prefetch_result.get("data") or []
    summary = summarize_inventory(cars)
    return json.dumps({
        "total_count": prefetch_result.get("total_count", len(cars)),
        "summary": summary,
        "car_ids": [car.get("id") for car in cars if car.get("id")],
    })


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    explicit_filters: NotRequired[Dict[str, Any]]
    web_enrichment: NotRequired[Dict[str, Any]]
    merged_filters: NotRequired[Dict[str, Any]]
    prefetched_inventory: NotRequired[Dict[str, Any]]


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
        
        # Start building query for similar cars (limited columns)
        query = supabase.table("cars").select(DEFAULT_CAR_COLUMNS).neq("id", car_id).eq("status", "available")
        
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
        
        # Execute with result cap
        result = query.limit(DEFAULT_RESULT_LIMIT).execute()
        
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
        query = supabase.table("cars").select(DEFAULT_CAR_COLUMNS).eq("status", "available")
        
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
        
        # Order and limit
        query = query.order("price", desc=False).limit(DEFAULT_RESULT_LIMIT)
        
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
        
        # Start building query with limited payload
        query = supabase.table("cars").select(DEFAULT_CAR_COLUMNS).eq("status", "available")
        
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
        
        query = query.limit(DEFAULT_RESULT_LIMIT)
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
                "spacious car": ["SUV"],
                # Electric/hybrid cars should use type field, not category
            }
            
            # Separate mapping for type field (electric/hybrid)
            car_type_fuel_mappings = {
                "electric": "Electric",
                "electric car": "Electric",
                "electric vehicle": "Electric",
                "ev": "Electric",
                "hybrid": "Hybrid",
                "hybrid car": "Hybrid",
                "eco friendly": "Electric",
                "environmentally friendly": "Electric",
                "zero emission": "Electric",
                "tesla": "Electric",  # Brand often associated with electric
                "gas": "Gasoline",
                "gasoline": "Gasoline",
                "petrol": "Gasoline",
                "diesel": "Diesel",
            }
            
            # Check if query matches any known patterns (more flexible matching)
            query_lower = query.lower()
            suggested_categories = []
            suggested_type = None
            
            # Check for category matches
            for car_type, categories in car_type_mappings.items():
                # More flexible matching - check if any words from car_type are in query
                car_type_words = car_type.split()
                if any(word in query_lower for word in car_type_words):
                    suggested_categories.extend(categories)
            
            # Check for fuel type matches
            for fuel_term, fuel_type in car_type_fuel_mappings.items():
                fuel_words = fuel_term.split()
                if any(word in query_lower for word in fuel_words):
                    suggested_type = fuel_type
                    break
            
            # Remove duplicates and ensure we have some suggestions
            if suggested_categories or suggested_type:
                unique_categories = list(set(suggested_categories)) if suggested_categories else []
                result_content = f"Based on your search for '{query}', I recommend looking at"
                
                if suggested_type:
                    result_content += f" {suggested_type} vehicles"
                    if unique_categories:
                        result_content += f" in these categories: {', '.join(unique_categories)}"
                else:
                    result_content += f" these vehicle categories: {', '.join(unique_categories)}"
                
                results.append({
                    "type": "category_suggestion",
                    "content": result_content,
                    "categories": unique_categories,
                    "fuel_type": suggested_type
                })
        
        # Extract car categories and types from the results
        extracted_insights = {
            "suggested_categories": [],
            "suggested_types": [],
            "suggested_makes": [],
            "key_features": [],
            "price_insights": []
        }
        
        # First check if we have direct category suggestions from fallback
        for result in results:
            if result.get("type") == "category_suggestion":
                if result.get("categories"):
                    extracted_insights["suggested_categories"].extend(result["categories"])
                if result.get("fuel_type"):
                    extracted_insights["suggested_types"].append(result["fuel_type"])
        
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
            "classic": "Classic",  # Add classic category
        }
        
        # Separate mapping for type field (fuel types)
        type_mapping = {
            "electric": "Electric",  # Add electric vehicles
            "electric car": "Electric",
            "electric vehicle": "Electric",
            "ev": "Electric",
            "hybrid": "Hybrid",  # Add hybrid vehicles
            "hybrid car": "Hybrid",
            "tesla": "Electric",  # Tesla is typically electric
            "eco": "Electric",  # Eco-friendly usually refers to electric
            "green": "Electric",  # Green cars usually electric
            "gas": "Gasoline",
            "gasoline": "Gasoline",
            "petrol": "Gasoline",
            "diesel": "Diesel"
        }
        
        found_categories = set()
        found_types = set()
        for search_term, db_category in category_mapping.items():
            if search_term in all_text:
                found_categories.add(db_category)
        
        for search_term, db_type in type_mapping.items():
            if search_term in all_text:
                found_types.add(db_type)
        
        # Combine direct suggestions and text analysis
        extracted_insights["suggested_categories"].extend(list(found_categories))
        extracted_insights["suggested_types"].extend(list(found_types))
        
        # Remove duplicates while preserving order
        original_cats = extracted_insights["suggested_categories"]
        extracted_insights["suggested_categories"] = list(dict.fromkeys(original_cats))  # dedupe keep order
        original_types = extracted_insights["suggested_types"]
        extracted_insights["suggested_types"] = list(dict.fromkeys(original_types))  # dedupe keep order
        
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
            },
            "electric": {
                "categories": [],  # Electric is a type, not category
                "types": ["Electric"],
                "features": ["zero emissions", "instant torque", "quiet operation"],
                "description": "Electric cars offer environmental benefits and modern technology"
            },
            "electric car": {
                "categories": [],  # Electric is a type, not category
                "types": ["Electric"],
                "features": ["zero emissions", "instant torque", "quiet operation"],
                "description": "Electric cars offer environmental benefits and modern technology"
            },
            "hybrid": {
                "categories": [],  # Hybrid is a type, not category
                "types": ["Hybrid"],
                "features": ["fuel efficiency", "low emissions", "dual power"],
                "description": "Hybrid cars combine electric and gas power for efficiency"
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
                    "suggested_categories": matched_insight.get("categories", []),
                    "suggested_types": matched_insight.get("types", []),
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
    get_similar_cars,
    get_market_insights,
    search_web_for_car_info
]

system_prompt = """You are CarFinder.ai, a helpful car shopping assistant for a Supabase-powered marketplace.

CONTEXT YOU RECEIVE
The orchestrator may inject system notes with: (a) explicit filters from the user, (b) web enrichment results, (c) prefetched inventory. Treat explicit filters as hard constraints.

AVAILABLE TOOLS
1. get_cars_data_eq - primary inventory search with filters (make, model, year, price, category, type, exclude_type, etc.)
   - Use `type` to filter BY a fuel type (e.g., type="Electric")
   - Use `exclude_type` to EXCLUDE a fuel type (e.g., exclude_type="Electric" for non-electric cars)
2. get_similar_cars - find cars similar to a specific car ID
3. get_market_insights - get statistics and trends
4. search_web_for_car_info - research lifestyle queries (student car, family car, etc.). Always follow with get_cars_data_eq.

RULES
- Never invent car IDs or specs. Only reference cars from tool results or prefetched data.
- Use tools to search before answering car-related questions.
- For lifestyle queries (son, student, family, commute), use search_web_for_car_info first, then get_cars_data_eq.
- When user says "no electric", "not electric", "non-electric", use exclude_type="Electric" in get_cars_data_eq.
- When user says "no hybrid", use exclude_type="Hybrid".

RESPONSE FORMAT
When you have finished using tools and are ready to respond to the user, you MUST output valid JSON in exactly this format:
{"message": "Your helpful response text here", "car_ids": [1, 2, 3]}

- "message": A friendly, helpful narrative (string)
- "car_ids": Array of up to 15 car IDs from your search results (integers)
- If no cars found, use an empty array: "car_ids": []
- Do NOT wrap in markdown code blocks
- Do NOT add any text before or after the JSON

STYLE
- Friendly and professional
- Mention top 3-5 cars in prose
- Ask follow-up questions if budget, size, or features are unclear
"""

# Initialize the model with the stable GA version
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)
llm = llm.bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    """Our agent node that processes messages and generates responses."""
    messages = state["messages"]
    context_chunks: List[str] = []

    explicit_filters = state.get("explicit_filters")
    if explicit_filters:
        context_chunks.append(f"Explicit filters extracted from user: {json.dumps(explicit_filters)}")

    web_enrichment = state.get("web_enrichment")
    if web_enrichment:
        insights = web_enrichment.get("extracted_insights") or {}
        context_chunks.append(
            "Web enrichment insights: " + json.dumps({
                "query": web_enrichment.get("query"),
                "suggested_categories": insights.get("suggested_categories"),
                "suggested_types": insights.get("suggested_types"),
                "key_features": insights.get("key_features")
            })
        )

    merged_filters = state.get("merged_filters")
    if merged_filters:
        context_chunks.append(f"Merged filters to respect: {json.dumps(merged_filters)}")
    
    prefetched_inventory = state.get("prefetched_inventory")
    if prefetched_inventory:
        context_chunks.append("Inventory snapshot: " + build_prefetch_context(prefetched_inventory))
    
    # Create the full prompt with system message and conversation
    if context_chunks:
        context_message = SystemMessage(content="\n".join(context_chunks))
        full_messages = [SystemMessage(content=system_prompt), context_message] + messages
    else:
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

def validate_and_fix_response(response_content: str, tool_results: list = None) -> str:
    """
    Validate and fix the AI response to ensure consistent JSON format.
    Handles both string and list response_content.
    Falls back to extracting car data from tool_results if AI response is invalid.
    """
    import json
    import re
    
    try:
        # If response_content is a list, join its elements into a string
        if isinstance(response_content, list):
            cleaned_content = " ".join(str(x) for x in response_content).strip()
        else:
            cleaned_content = str(response_content).strip()
        
        # Remove markdown code blocks if they exist (handle various formats)
        cleaned_content = re.sub(r'^```(?:json)?\s*', '', cleaned_content)
        cleaned_content = re.sub(r'\s*```$', '', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        # Fix literal newlines inside JSON string values (LLMs often return these)
        # Regex: Find content between "message": " and the closing " before car_ids
        def fix_newlines_in_json_string(match):
            prefix = match.group(1)
            content = match.group(2)
            suffix = match.group(3)
            # Replace actual newlines with escaped \n
            fixed_content = content.replace('\n', '\\n').replace('\r', '')
            return f'{prefix}{fixed_content}{suffix}'
        
        cleaned_content = re.sub(
            r'("message"\s*:\s*")([^"]*?)("\s*,\s*"car_ids")', 
            fix_newlines_in_json_string, 
            cleaned_content, 
            flags=re.DOTALL
        )
        
        # If content is too short, it's likely truncated - try to extract from tool results
        if len(cleaned_content) < 20:
            print(f"⚠️ Content too short ({len(cleaned_content)} chars), checking tool results")
            if tool_results:
                return _extract_response_from_tools(tool_results, cleaned_content)
            raise ValueError("Response too short and no tool results available")
        
        # Try to find JSON object in the content
        json_match = re.search(r'\{[^{}]*"message"[^{}]*"car_ids"[^{}]*\}', cleaned_content, re.DOTALL)
        if json_match:
            cleaned_content = json_match.group(0)
        else:
            # Handle corrupted/truncated JSON by finding the first complete JSON object
            brace_count = 0
            json_start = -1
            json_end = -1
            for i, char in enumerate(cleaned_content):
                if char == '{':
                    if json_start == -1:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_start >= 0 and json_end > json_start:
                cleaned_content = cleaned_content[json_start:json_end]
        
        # Try to parse as JSON
        data = json.loads(cleaned_content)
        
        # Validate required fields
        if not isinstance(data, dict) or "message" not in data or "car_ids" not in data:
            raise ValueError("Missing required fields")
        
        # Ensure car_ids is array of numbers (deduplicated)
        if not isinstance(data["car_ids"], list):
            data["car_ids"] = []
        else:
            # Convert string IDs to numbers and remove duplicates
            fixed_ids = []
            seen_ids = set()
            for car_id in data["car_ids"]:
                try:
                    id_int = int(car_id)
                    if id_int not in seen_ids:
                        fixed_ids.append(id_int)
                        seen_ids.add(id_int)
                except (ValueError, TypeError):
                    continue
            data["car_ids"] = fixed_ids
        
        # Ensure message is string and not too long
        if not isinstance(data["message"], str):
            data["message"] = str(data["message"])
        
        # Truncate message if too long (prevent token overflow)
        if len(data["message"]) > 2000:
            data["message"] = data["message"][:1900] + "... Please let me know if you'd like more details about any specific cars."
        
        return json.dumps(data)
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON validation error: {e}")
        # Try to extract from tool results if available
        if tool_results:
            extracted = _extract_response_from_tools(tool_results, response_content)
            if extracted:
                return extracted
        
        # If the response looks like plain text (not JSON), wrap it
        if isinstance(response_content, str) and response_content.strip() and not response_content.strip().startswith('{'):
            # Clean up the text and use it as the message
            plain_text = response_content.strip()
            if len(plain_text) > 20:  # Seems like a real message
                return json.dumps({
                    "message": plain_text,
                    "car_ids": []
                })
        
        # If JSON parsing fails completely, create a fallback response
        return json.dumps({
            "message": "I'd be happy to help you find the right car. Could you share more details about what you're looking for? For example, your budget range, preferred size, or any specific features you need?",
            "car_ids": []
        })


def _extract_response_from_tools(messages: list, ai_text: str = "") -> str:
    """Extract car data from tool messages when AI response is invalid."""
    import json
    
    car_ids = []
    car_count = 0
    
    # Search through messages for tool results with car data
    for msg in reversed(messages):
        content = getattr(msg, 'content', None)
        if not content:
            continue
            
        try:
            content_str = str(content)
            if '"success": true' in content_str.lower() or '"success":true' in content_str.lower():
                data = json.loads(content_str)
                if data.get("success") and data.get("data"):
                    cars = data["data"]
                    car_count = data.get("total_count", len(cars))
                    car_ids = [car.get("id") for car in cars[:15] if car.get("id")]
                    break
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    if car_ids:
        # Create a helpful message based on the results
        if car_count == 1:
            message = "I found 1 car that matches your criteria."
        elif car_count <= 5:
            message = f"I found {car_count} cars that match your criteria. Here are your options."
        else:
            message = f"I found {car_count} cars matching your search. Here are the top options to consider."
        
        return json.dumps({
            "message": message,
            "car_ids": car_ids
        })
    
    # No car data found, return generic response
    return json.dumps({
        "message": "I'd be happy to help you find the right car. Could you share more details about what you're looking for? For example, your budget range, preferred size, or any specific features you need?",
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
    import json  # Ensure json is always imported for error handling
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"🤖 chat_with_bot attempt {attempt + 1}: {user_input[:50]}...")

            # Pre-process request to extract deterministic filters
            explicit_filters = extract_explicit_filters(user_input)
            web_enrichment: Dict[str, Any] = {}
            if should_enrich_with_web(user_input, explicit_filters):
                try:
                    web_raw = search_web_for_car_info.invoke({"query": user_input})
                    parsed_web = json.loads(web_raw)
                    if parsed_web.get("success"):
                        web_enrichment = parsed_web
                except Exception as enrich_error:
                    print(f"⚠️ Web enrichment skipped: {enrich_error}")

            merged_filters = merge_filters(explicit_filters, web_enrichment)
            prefetched_inventory = prefetch_inventory(merged_filters)
            
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
            current_input: AgentState = {"messages": messages}
            if explicit_filters:
                current_input["explicit_filters"] = explicit_filters
            if web_enrichment:
                current_input["web_enrichment"] = web_enrichment
            if merged_filters:
                current_input["merged_filters"] = merged_filters
            if prefetched_inventory:
                current_input["prefetched_inventory"] = prefetched_inventory
            
            # Run the agent
            print("🚀 Invoking AI agent...")
            result = app.invoke(current_input)
            print("✅ AI agent completed successfully")
            
            # Extract the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            all_messages = result["messages"]  # Keep all messages for tool result extraction
            
            if ai_messages:
                last_ai_message = ai_messages[-1]
                
                # Debug: print full message details
                print(f"🔍 Last AI message type: {type(last_ai_message)}")
                print(f"🔍 Content type: {type(last_ai_message.content)}")
                print(f"🔍 Content repr: {repr(last_ai_message.content)[:200]}")
                
                # Handle case where content might be a list of parts (Gemini format)
                raw_content = last_ai_message.content
                if isinstance(raw_content, list):
                    # Extract text from Gemini parts format: [{'type': 'text', 'text': '...'}]
                    text_parts = []
                    for part in raw_content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif isinstance(part, str):
                            text_parts.append(part)
                        else:
                            text_parts.append(str(part))
                    raw_response = "".join(text_parts)
                elif raw_content:
                    raw_response = str(raw_content)
                else:
                    raw_response = ""
                
                print(f"📤 Raw AI response received: {len(raw_response)} characters")
                print(f"📤 Response preview: {raw_response[:300] if raw_response else '(empty)'}...")
                
                # Validate and fix the response format, passing all messages for fallback extraction
                validated_response = validate_and_fix_response(raw_response, all_messages)
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
        
        # Enhanced keyword detection
        if any(word in user_lower for word in ["sport", "sports", "sporty", "fast", "performance"]):
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
        elif any(word in user_lower for word in ["electric", "ev", "tesla", "electric car", "electric vehicle"]):
            search_params["type"] = "Electric"
            print("🔍 Detected electric car request")
        elif any(word in user_lower for word in ["hybrid", "hybrid car", "prius"]):
            search_params["type"] = "Hybrid"
            print("🔍 Detected hybrid car request")
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
        
        # Perform a basic database search
        result = get_cars_data_eq.invoke(search_params)
        result_data = json.loads(result)
        
        if result_data.get("success") and result_data.get("data"):
            all_cars = result_data["data"]  # Get all results
            cars_to_show = all_cars[:5]  # Show details for top 5 only
            total_count = result_data.get("total_count", 0)
            
            # Create list of all car IDs for frontend
            all_car_ids = [car.get("id") for car in all_cars if car.get("id")]
            
            response = {
                "message": f"Found {total_count} cars that might interest you. Here are the top {len(cars_to_show)}:",
                "car_ids": all_car_ids  # Use ALL matching IDs for frontend
            }
            
            # Format top cars for display in message
            car_list = []
            for i, car in enumerate(cars_to_show, 1):
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
    print("Type 'quit' to exit, 'clear' to start a new conversation.")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using the Car Search Assistant! Goodbye! 👋")
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("🔄 Conversation cleared. Starting fresh!")
            continue
        elif not user_input:
            print("Please enter a message.")
            continue
        
        print("Bot: ", end="", flush=True)
        response = chat_with_bot(user_input, conversation_history)
        print(response)
        
        # Update conversation history for next turn
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        print(f"\n📚 History: {len(conversation_history)} messages")

# Example usage for testing
if __name__ == "__main__":
    # Test the enhanced search functions
    print("Testing enhanced car search functions...")
    
    try:
        print("\n1. Testing get_similar_cars:")
        result1 = get_similar_cars.invoke({
            "car_id": 1,
            "similarity_factors": "make,category,price_range",
            "limit": 3
        })
        print(result1)

        print("\n2. Testing get_market_insights:")
        result2 = get_market_insights.invoke({
            "make": "Toyota",
            "condition": "Used"
        })
        print(result2)

        print("\n3. Testing search_web_for_car_info:")
        result3 = search_web_for_car_info.invoke({
            "query": "best family cars"
        })
        print(result3)

        print("\n4. Testing combined enrichment workflow:")
        web_result = search_web_for_car_info.invoke({
            "query": "college student commuter car"
        })
        web_data = json.loads(web_result)
        merged_filters = merge_filters({}, web_data)
        db_payload = {k: v for k, v in merged_filters.items() if k in {"category", "type", "price_max", "price_min"}}
        if db_payload:
            db_payload.setdefault("price_max", 30000)
            db_payload.setdefault("status", "available")
            inventory = get_cars_data_eq.invoke(db_payload)
            print(inventory)

        print("\n✅ Smoke tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Note: This might be due to Supabase connection issues or missing data.")
    
    # Start interactive chat
    print("\n" + "="*50)
    print("Starting interactive chat with enhanced features...")
    start_interactive_chat()