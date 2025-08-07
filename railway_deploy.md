# 🚀 Deploy on Railway (Easiest Option)

## Step 1: Create Railway Account
1. Go to https://railway.app/
2. Sign up with GitHub account
3. Connect your GitHub account

## Step 2: Deploy Your Branch
1. Click "New Project" → "Deploy from GitHub repo"
2. Select `DivamSanghvi/bajaj_finserv` repository
3. **IMPORTANT**: Change branch from `main` to `vedica-2`
4. Click "Deploy"

## Step 3: Add Environment Variables
In Railway dashboard, go to Variables tab and add:
```
OPENAI_API_KEY=your-actual-openai-key
POSTGRES_HOST=railway-postgres-host
POSTGRES_PORT=5432
POSTGRES_DB=railway_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=railway-password
```

## Step 4: Configure Deployment
Railway will auto-detect your Python app, but create a `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

## ✅ Your app will be live at: `https://your-app-name.railway.app`

**Free Tier**: $5 credit monthly, should be enough for testing!
