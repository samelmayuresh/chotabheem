# 🚀 Google Cloud Run Deployment Guide

Complete guide to deploy your Emotion AI app to Google Cloud Run.

## 📋 Prerequisites

### 1. Install Required Tools

**Google Cloud CLI:**
- Download from: https://cloud.google.com/sdk/docs/install
- Run: `gcloud init` to authenticate

**Docker Desktop:**
- Download from: https://docs.docker.com/get-docker/
- Make sure Docker is running

### 2. Google Cloud Setup

1. **Create a Google Cloud Project:**
   - Go to: https://console.cloud.google.com/
   - Create a new project or select existing one
   - Note your PROJECT_ID

2. **Enable Billing:**
   - Cloud Run requires a billing account
   - Go to Billing section and add payment method

3. **Authenticate gcloud:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## 🚀 Deployment Methods

### Method 1: Automated Deployment (Recommended)

1. **Update Project ID:**
   - Edit `deploy.bat` (Windows) or `deploy.sh` (Linux/Mac)
   - Replace `your-project-id` with your actual Google Cloud Project ID

2. **Run Deployment Script:**
   ```bash
   # Windows
   deploy.bat
   
   # Linux/Mac
   ./deploy.sh
   ```

3. **Wait for Deployment:**
   - Script will build, push, and deploy automatically
   - Takes 5-10 minutes for first deployment

### Method 2: Manual Deployment

1. **Set Project:**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable APIs:**
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

3. **Build and Push Image:**
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/emotion-ai .
   docker push gcr.io/YOUR_PROJECT_ID/emotion-ai
   ```

4. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy emotion-ai \
     --image gcr.io/YOUR_PROJECT_ID/emotion-ai \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 900 \
     --max-instances 10
   ```

### Method 3: Using Cloud Build (CI/CD)

1. **Connect GitHub Repository:**
   - Go to Cloud Build > Triggers
   - Connect your GitHub repository
   - Create trigger using `cloudbuild.yaml`

2. **Automatic Deployment:**
   - Every push to main branch will trigger deployment
   - Fully automated CI/CD pipeline

## ⚙️ Configuration Options

### Resource Allocation
- **Memory:** 2Gi (can be adjusted based on usage)
- **CPU:** 2 vCPUs (can be scaled up/down)
- **Timeout:** 900 seconds (15 minutes)
- **Max Instances:** 10 (auto-scaling)

### Environment Variables
Add environment variables during deployment:
```bash
--set-env-vars OPENROUTER_KEY=your_key,SUPABASE_URL=your_url
```

### Custom Domain
1. Go to Cloud Run console
2. Select your service
3. Click "Manage Custom Domains"
4. Add your domain and verify

## 🔧 Troubleshooting

### Common Issues

**1. Build Fails:**
- Check Dockerfile syntax
- Ensure all dependencies in requirements.txt
- Verify Docker is running

**2. Memory Issues:**
- Increase memory allocation: `--memory 4Gi`
- Optimize your code for memory usage

**3. Timeout Issues:**
- Increase timeout: `--timeout 1800`
- Optimize app startup time

**4. Permission Errors:**
- Check IAM permissions
- Ensure billing is enabled
- Verify APIs are enabled

### Debugging Commands

**View Logs:**
```bash
gcloud run services logs read emotion-ai --region us-central1
```

**Service Details:**
```bash
gcloud run services describe emotion-ai --region us-central1
```

**List Services:**
```bash
gcloud run services list
```

## 💰 Cost Estimation

**Cloud Run Pricing (us-central1):**
- **CPU:** $0.00002400 per vCPU-second
- **Memory:** $0.00000250 per GiB-second
- **Requests:** $0.40 per million requests

**Estimated Monthly Cost (1000 users, 10 requests/day):**
- ~$20-50/month depending on usage
- First 2 million requests free per month

## 🔒 Security Best Practices

1. **Environment Variables:**
   - Store API keys as environment variables
   - Never commit secrets to repository

2. **Authentication:**
   - Enable authentication if needed: `--no-allow-unauthenticated`
   - Use IAM for access control

3. **HTTPS:**
   - Cloud Run provides HTTPS by default
   - Custom domains get free SSL certificates

## 📊 Monitoring & Scaling

### Monitoring
- **Cloud Console:** Monitor requests, latency, errors
- **Cloud Logging:** View application logs
- **Cloud Monitoring:** Set up alerts

### Auto-scaling
- **Min Instances:** 0 (scales to zero when not used)
- **Max Instances:** 10 (adjust based on expected traffic)
- **Concurrency:** 80 requests per instance (default)

## 🎯 Post-Deployment

### 1. Test Your App
- Visit the provided URL
- Test all features (voice, text, music, etc.)
- Check analytics and database connectivity

### 2. Update DNS (Optional)
- Point your custom domain to Cloud Run URL
- Update any external integrations

### 3. Monitor Performance
- Check Cloud Run metrics
- Monitor costs and usage
- Set up alerts for errors

## 🔄 Updates & Maintenance

### Update Deployment
```bash
# Rebuild and redeploy
docker build -t gcr.io/YOUR_PROJECT_ID/emotion-ai .
docker push gcr.io/YOUR_PROJECT_ID/emotion-ai
gcloud run deploy emotion-ai --image gcr.io/YOUR_PROJECT_ID/emotion-ai
```

### Rollback
```bash
# List revisions
gcloud run revisions list --service emotion-ai

# Rollback to previous revision
gcloud run services update-traffic emotion-ai --to-revisions REVISION_NAME=100
```

## 🎉 Success!

Your Emotion AI app is now deployed on Google Cloud Run with:
- ✅ Auto-scaling infrastructure
- ✅ HTTPS by default
- ✅ Global CDN
- ✅ 99.95% uptime SLA
- ✅ Pay-per-use pricing

**Your app URL:** https://emotion-ai-[hash]-uc.a.run.app

Enjoy your production-ready Emotion AI platform! 🧠✨