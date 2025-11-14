# Quick Deployment Guide

## Deploy to Render (Easiest Option)

### Step 1: Prepare Your Code

1. **Make sure your `artifacts/` folder is committed to Git** (the models need to be deployed):
   ```bash
   git add artifacts/
   git commit -m "Add model artifacts"
   ```

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

### Step 2: Deploy on Render

1. **Sign up/Login to Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub (free tier available)

2. **Create a Blueprint**:
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and create both services

3. **After Deployment**:
   - Note the API service URL (e.g., `https://4th-down-decision-api.onrender.com`)
   - Go to the Dash app service settings
   - Add environment variable:
     - **Key**: `DECISION_API_URL`
     - **Value**: `https://4th-down-decision-api.onrender.com` (use your actual API URL)

4. **Redeploy the Dash App**:
   - After adding the environment variable, trigger a manual deploy

### Step 3: Access Your App

- **API**: `https://4th-down-decision-api.onrender.com`
- **Dash App**: `https://4th-down-decision-app.onrender.com`

---

## Alternative: Railway (Also Easy)

1. **Install Railway CLI**:
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy Backend**:
   ```bash
   railway init
   railway up
   railway service
   railway variables set PORT=8000
   railway run uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
   Note the service URL

3. **Deploy Frontend**:
   - Create new service in Railway dashboard
   - Set: `DECISION_API_URL=https://your-api-url.railway.app`
   - Deploy: `python apppreview.py`

---

## Important Notes

⚠️ **Make sure `artifacts/` folder is in Git** - Your models need to be deployed!

⚠️ **Render Free Tier**: Services spin down after 15 min of inactivity (first request will be slow)

⚠️ **First Deployment**: May take 5-10 minutes to build and deploy

---

## Troubleshooting

- **API not connecting**: Check `DECISION_API_URL` environment variable in Dash app service
- **Models not loading**: Ensure `artifacts/` folder is committed to Git
- **Build fails**: Check that all dependencies are in `requirements.txt`

