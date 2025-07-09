# Quick Start Guide - LangGraph AI Car Agent

## 🚀 Get Running in 3 Steps

### 1. Install Dependencies
```bash
cd ai_python
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Copy template and edit with your API key
cp env_template.txt .env

# Edit .env file:
# GEMINI_API_KEY=your_actual_gemini_api_key
```

Get your API key: https://aistudio.google.com/app/apikey

### 3. Run the Agent
```bash
# Test the agent
python test_agent.py

# Run direct comparison
python ai_agent.py

# Start REST API server
python api_server.py
```

## 📡 API Endpoints (when running api_server.py)

- **Health Check**: `GET http://localhost:8000/health`
- **Compare Cars**: `POST http://localhost:8000/compare`
- **API Docs**: `GET http://localhost:8000/docs`

### Example API Request
```bash
curl -X POST "http://localhost:8000/compare" \
     -H "Content-Type: application/json" \
     -d '{"car1": "2018 BMW X5", "car2": "2020 Mercedes-Benz G500"}'
```

## 🔧 Architecture Highlights

- ✅ **Modern LangGraph**: Uses the same advanced framework as your JavaScript version
- ✅ **Conventional Messages**: Proper SystemMessage + HumanMessage structure for better AI behavior
- ✅ **Async Operations**: Fast parallel web searches
- ✅ **REST API Ready**: Easy React Native integration
- ✅ **Comprehensive Testing**: Built-in test suite
- ✅ **Production Ready**: Error handling, logging, CORS support

## 🐛 Troubleshooting

**Missing API Key Error**:
- Make sure `.env` file exists in `ai_python/` directory
- Verify `GEMINI_API_KEY` is set correctly

**Import Errors**:
- Run `pip install -r requirements.txt` again
- Check Python version (requires 3.7+)

**Search Failures**:
- Check internet connection
- DuckDuckGo API might be rate-limited (built-in fallback data)

## 📱 React Native Integration

Once the API server is running:

```javascript
// In your React Native app
const compareCarsCosts = async (car1, car2) => {
  const response = await fetch('http://your-server:8000/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ car1, car2 })
  });
  return response.json();
};
```

You're all set! 🎉 