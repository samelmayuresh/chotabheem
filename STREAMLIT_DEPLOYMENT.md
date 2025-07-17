# 🚀 Deploy to Streamlit Cloud - Quick Guide

## Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

## Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

## Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy!"

## Step 3: Environment Variables (if needed)
If your app uses API keys:
1. In Streamlit Cloud dashboard
2. Go to "Settings" → "Secrets"
3. Add your environment variables

## Alternative: Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Troubleshooting
- If sentencepiece fails: The system dependencies we added should fix this
- If memory issues: Streamlit Cloud has 1GB RAM limit
- If build timeout: Consider removing heavy dependencies

Your app should deploy successfully with the fixes we made!