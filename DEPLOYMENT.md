# Vercel Deployment Guide

## Prerequisites
- GitHub repository connected to Vercel
- Gemini API key from Google AI Studio

## Deployment Steps

### 1. Environment Variables
In your Vercel dashboard, go to your project settings and add:
- `GEMINI_API_KEY`: Your Google Gemini API key

### 2. Deploy
The project is configured for automatic deployment when you push to the main branch.

### 3. Test Endpoints
After deployment, test these endpoints:
- `GET /` - Health check
- `GET /health` - Health status  
- `GET /test` - Vercel deployment test
- `POST /compare` - Car comparison (main functionality)
- `GET /docs` - FastAPI documentation

### 4. Example API Call
```bash
curl -X POST "https://your-app.vercel.app/compare" \
  -H "Content-Type: application/json" \
  -d '{"car1": "2018 BMW X5", "car2": "2020 Mercedes G500"}'
```

## Files Configured for Vercel
- `vercel.json` - Vercel configuration
- `index.py` - Main entry point
- `runtime.txt` - Python version specification
- `requirements.txt` - Dependencies
- `index.py` - FastAPI application

## Troubleshooting
1. Check Vercel function logs if deployment fails
2. Verify environment variables are set correctly
3. Ensure all dependencies in requirements.txt are compatible
4. Check that the GEMINI_API_KEY is properly configured

## Cold Start Performance
- First request may take 10-15 seconds due to cold start
- Subsequent requests will be much faster
- Consider using Vercel Pro for better cold start performance 