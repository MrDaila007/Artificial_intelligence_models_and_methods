# 🚀 Deployment Guide

This guide explains how to deploy the Streamlit web application to Streamlit Cloud (free hosting).

## 📋 Prerequisites

- GitHub account
- Repository pushed to GitHub
- Streamlit Cloud account (free)

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

Make sure your code is pushed to GitHub:

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Sign up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### Step 3: Configure Deployment

1. **Select repository**: Choose your `Artificial_intelligence_models_and_methods` repository
2. **Select branch**: `main` (or your default branch)
3. **Main file path**: 
   - For unified app: `app.py`
   - For Lab1 only: `Lab1/app.py`
   - For Lab2 only: `Lab2/app.py`
4. **Python version**: 3.8+ (auto-detected)
5. **Advanced settings** (optional):
   - Secrets: Not needed for this app
   - App URL: Auto-generated (e.g., `your-app-name.streamlit.app`)

### Step 4: Deploy

Click "Deploy" and wait for the build to complete (usually 1-2 minutes).

## 🔗 Access Your App

After deployment, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

Share this URL with others to access your application!

## 🔄 Automatic Updates

Streamlit Cloud automatically redeploys your app when you push changes to the connected branch.

## 📝 Configuration Files

The app uses `.streamlit/config.toml` for configuration:
- Theme customization
- Server settings
- Browser settings

## 🛠️ Troubleshooting

### Build Fails

1. Check that `requirements.txt` includes all dependencies
2. Verify Python version compatibility
3. Check build logs in Streamlit Cloud dashboard

### Import Errors

1. Ensure all imports use relative paths correctly
2. Check that `solution.py` files are in the correct directories
3. Verify `sys.path` modifications in `app.py`

### Module Not Found

If you see import errors, ensure:
- All dependencies are in `requirements.txt`
- File paths are correct
- No circular imports

## 🌍 Alternative Deployment Options

### Heroku

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Railway

1. Connect GitHub repository
2. Set start command: `streamlit run app.py --server.port=$PORT`
3. Deploy automatically

### Render

1. Create new Web Service
2. Connect GitHub repository
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 📊 Monitoring

Streamlit Cloud provides:
- Usage statistics
- Error logs
- Performance metrics

Access these in your Streamlit Cloud dashboard.

## 🔒 Security Notes

- Streamlit Cloud apps are public by default
- Don't commit sensitive data (API keys, passwords)
- Use Streamlit secrets for environment variables if needed

## 📚 Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Deployment Best Practices](https://docs.streamlit.io/knowledge-base/tutorials/deploy)

---

**Note**: GitHub Pages only supports static sites. For Python web apps like Streamlit, use Streamlit Cloud or other PaaS providers.

