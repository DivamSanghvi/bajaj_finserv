# 🌐 Deploy on Render (Alternative Option)

## Step 1: Create Render Account  
1. Go to https://render.com/
2. Sign up with GitHub account

## Step 2: Deploy Web Service
1. Click "New +" → "Web Service"
2. Connect GitHub → Select `DivamSanghvi/bajaj_finserv`
3. **Branch**: Change to `vedica-2` 
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Step 3: Environment Variables
Add in Render dashboard:
```
OPENAI_API_KEY=your-actual-openai-key
POSTGRES_HOST=render-postgres-host  
POSTGRES_PORT=5432
POSTGRES_DB=render_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=render-password
```

## Step 4: Database (Optional)
1. Create PostgreSQL database in Render
2. Use the connection details in environment variables

## ✅ Your app will be live at: `https://your-app-name.onrender.com`

**Free Tier**: 750 hours/month, spins down after inactivity
