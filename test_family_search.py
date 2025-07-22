#!/usr/bin/env python
import sys
sys.path.append('.')
from api.ai_agent_chatbot import search_cars, search_web_for_car_info
import json

def test_family_car_search():
    print("=== Testing Family Car Search ===")
    
    # Test 1: Web search for family car
    print("\n1. Web search for 'family car':")
    web_result = search_web_for_car_info.invoke({'query': 'family car'})
    web_data = json.loads(web_result)
    print(f"Success: {web_data.get('success')}")
    if web_data.get('extracted_insights'):
        categories = web_data['extracted_insights'].get('suggested_categories', [])
        print(f"Suggested categories: {categories}")
    
    # Test 2: Search for SUVs (better for families)
    print("\n2. Database search for SUVs:")
    suv_result = search_cars.invoke({'category': 'SUV', 'sort_by': 'price', 'ascending': True})
    suv_data = json.loads(suv_result)
    print(f"SUV search success: {suv_data.get('success')}")
    print(f"Total SUVs found: {suv_data.get('total_count', 0)}")
    if suv_data.get('data') and len(suv_data['data']) > 0:
        first_suv = suv_data['data'][0]
        print(f"First SUV: {first_suv.get('make')} {first_suv.get('model')} {first_suv.get('year')} - ${first_suv.get('price')}")
    
    # Test 3: Search for Sedans
    print("\n3. Database search for Sedans:")
    sedan_result = search_cars.invoke({'category': 'Sedan', 'sort_by': 'price', 'ascending': True})
    sedan_data = json.loads(sedan_result)
    print(f"Sedan search success: {sedan_data.get('success')}")
    print(f"Total Sedans found: {sedan_data.get('total_count', 0)}")
    if sedan_data.get('data') and len(sedan_data['data']) > 0:
        first_sedan = sedan_data['data'][0]
        print(f"First Sedan: {first_sedan.get('make')} {first_sedan.get('model')} {first_sedan.get('year')} - ${first_sedan.get('price')}")
    
    # Test 4: General search without category
    print("\n4. General search (no category filter):")
    general_result = search_cars.invoke({'sort_by': 'price', 'ascending': True})
    general_data = json.loads(general_result)
    print(f"General search success: {general_data.get('success')}")
    print(f"Total cars found: {general_data.get('total_count', 0)}")
    if general_data.get('data') and len(general_data['data']) > 0:
        first_car = general_data['data'][0]
        print(f"First car: {first_car.get('make')} {first_car.get('model')} {first_car.get('year')} - ${first_car.get('price')}")

if __name__ == "__main__":
    test_family_car_search() 