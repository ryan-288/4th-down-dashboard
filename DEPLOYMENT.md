# Deployment Guide

This application consists of two services:
1. **FastAPI Backend** (`api.py`) - Provides the decision-making API
2. **Dash Frontend** (`apppreview.py`) - Provides the web interface

## Option 1: Render (Recommended)

### Prerequisites
- GitHub account
- Render account (free tier available)

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and deploy both services

3. **Verify Deployment**
   - The API will be available at: `https://4th-down-decision-api.onrender.com`
   - The Dash app will be available at: `https://4th-down-decision-app.onrender.com`
   - The Dash app will automatically connect to the API

### Important Notes
- Render free tier services spin down after 15 minutes of inactivity
- First deployment may take 5-10 minutes
- Make sure your `artifacts/` folder is committed to Git (or use a different storage solution)

---

## Option 2: Railway

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy Backend**
   ```bash
   railway init
   railway up
   railway service
   railway variables set PORT=8000
   railway run uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

3. **Deploy Frontend**
   - Create a new service in Railway dashboard
   - Set environment variable: `DECISION_API_URL=https://your-api-url.railway.app`
   - Deploy: `python apppreview.py`

---

## Option 3: Heroku

### Backend (FastAPI)

1. **Create `Procfile`**
   ```
   web: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name-api
   git push heroku main
   ```

### Frontend (Dash)

1. **Create `Procfile`**
   ```
   web: python apppreview.py
   ```

2. **Set Environment Variable**
   ```bash
   heroku create your-app-name-app
   heroku config:set DECISION_API_URL=https://your-api-name-api.herokuapp.com
   git push heroku main
   ```

---

## Option 4: Docker + Any Cloud Provider

### Create `Dockerfile` for API
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create `Dockerfile` for Dash App
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DECISION_API_URL=http://api:8000
CMD ["python", "apppreview.py"]
```

### Create `docker-compose.yml`
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
  
  app:
    build: .
    ports:
      - "8050:8050"
    environment:
      - DECISION_API_URL=http://api:8000
    depends_on:
      - api
```

Deploy to:
- AWS (ECS, Elastic Beanstalk)
- Google Cloud (Cloud Run, GKE)
- Azure (Container Instances, AKS)
- DigitalOcean (App Platform)

---

## Option 5: Self-Hosted VPS

1. **SSH into your server**
2. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3.10 python3-pip nginx
   ```

3. **Clone repository**
   ```bash
   git clone <your-repo>
   cd 4th-Down-Decision-Tool-main
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Run with systemd** (create service files)
5. **Configure Nginx** as reverse proxy

---

## Environment Variables

### Backend (API)
- `PORT` - Port to run on (default: 8000)

### Frontend (Dash)
- `PORT` - Port to run on (default: 8050)
- `DECISION_API_URL` - URL of the API service (default: http://localhost:8000)

---

## Troubleshooting

1. **API not connecting**: Check `DECISION_API_URL` environment variable
2. **Models not loading**: Ensure `artifacts/` folder is included in deployment
3. **Port issues**: Make sure services use `$PORT` environment variable (Render/Railway) or `0.0.0.0` host
4. **CORS errors**: API already has CORS enabled for all origins

---

## Quick Test Locally

```bash
# Terminal 1 - Start API
uvicorn api:app --reload

# Terminal 2 - Start Dash App
python apppreview.py
```

Visit: http://localhost:8050

