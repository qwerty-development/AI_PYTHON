# AI Car Comparison Agent - Python Version (with LangGraph!)

This is the Python version of the AI Car Comparison Agent, designed to work as a backend service that can be easily integrated with React Native or any frontend application. **Now properly using LangGraph** for modern AI agent workflows!

## Why Python + LangGraph?

- **Better AI/ML ecosystem**: More libraries and tools available
- **React Native compatibility**: Avoid Node.js package compatibility issues
- **Backend-first approach**: Designed to be deployed as a web service
- **Modern LangGraph Architecture**: Uses the same advanced agent framework as the JavaScript version
- **Conventional Message Structure**: Proper SystemMessage + HumanMessage pattern for better agent behavior
- **Better State Management**: LangGraph provides superior workflow control and debugging
- **Scalability**: Easier to deploy and scale in production

## Features

- **Comprehensive Car Comparison**: Compares two cars across multiple cost categories
- **Real-time Web Search**: Uses DuckDuckGo API for current pricing and cost data
- **Detailed Cost Breakdown**: Annual costs, 5-year ownership costs, depreciation analysis
- **Async Performance**: Fast parallel web searches for efficient data gathering
- **Structured JSON Output**: Clean, standardized response format

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `ai_python` directory:

```bash
# Copy the template
cp env_template.txt .env

# Edit with your actual API key
GEMINI_API_KEY=your_actual_gemini_api_key
```

Get your Gemini API key from: https://aistudio.google.com/app/apikey

### 3. Run the Agent

```bash
python ai_agent.py
```

## Usage

### As a Module

```python
from ai_agent import compare_cars_costs

# Compare two cars
result = await compare_cars_costs("2018 BMW X5", "2020 Mercedes-Benz G500")
print(result)
```

### Direct Execution

The agent will run with default comparison (BMW X5 vs Mercedes G500) when executed directly:

```bash
python ai_agent.py
```

## API Response Format

The agent returns a structured JSON response with:

- **Annual Cost Estimates**: Maintenance, insurance, fuel, registration
- **Depreciation Analysis**: Current value, 5-year projection, depreciation rate
- **5-Year Cost Breakdown**: Total ownership costs over 5 years
- **Comparison Results**: Which car is more economical and by how much
- **Raw Search Data**: Original web search results for transparency

## Integration with React Native

This Python backend can be easily integrated with React Native by:

1. **Deploy as API**: Use FastAPI or Flask to create REST endpoints
2. **Docker Deployment**: Containerize for easy deployment
3. **Cloud Functions**: Deploy on AWS Lambda, Google Cloud Functions, etc.

## Cost Categories Analyzed

1. **Purchase Price**: MSRP and current market value
2. **Depreciation**: Value loss over 5 years
3. **Insurance**: Annual premium costs
4. **Maintenance**: Regular service and repair costs
5. **Fuel**: Annual fuel consumption costs
6. **Registration**: Government fees and vehicle taxes
7. **Total Cost of Ownership**: Complete 5-year financial picture

## Error Handling

- Graceful fallback to industry averages when web search fails
- Comprehensive error logging and reporting
- Retry mechanisms for network requests

## Development

The code is structured with clear separation of concerns using **LangGraph**:

- `web_search_tool`: Handles DuckDuckGo API integration with async operations
- `car_comparison_tool`: Orchestrates the comparison process with parallel searches
- `create_ai_agent`: Creates LangGraph ReAct agent (modern approach)
- `run_car_agent`: Main execution function with LangGraph AI analysis

### LangGraph Advantages:
- **Better debugging**: Full visibility into agent decision-making process
- **State persistence**: Maintains conversation context across tool calls
- **Graph-based workflows**: More flexible than linear agent chains
- **Modern architecture**: Same as the JavaScript version, ensuring consistency

## Next Steps

This Python agent is ready to be integrated into a web API for your React Native app! 