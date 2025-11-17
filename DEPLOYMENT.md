# Deployment Guide: GitHub to RunPod

This guide walks you through deploying the Emotion Detection API from GitHub to RunPod.

## Prerequisites

- GitHub account
- Docker Hub account (free tier works)
- RunPod account

## Step-by-Step Deployment

### 1. Create Docker Hub Account

1. Go to [Docker Hub](https://hub.docker.com/)
2. Sign up for a free account
3. Create an access token:
   - Go to Account Settings → Security → New Access Token
   - Name it "GitHub Actions"
   - Copy the token (you won't see it again!)

### 2. Push Code to GitHub

```bash
# Initialize git repository
cd /home/sameed/Documents/code/omniscent/container
git init
git add .
git commit -m "Initial commit: Emotion detection API"

# Create repository on GitHub first, then:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/emotion-detection-api.git
git push -u origin main
```

### 3. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these two secrets:

   **Secret 1:**
   - Name: `DOCKERHUB_USERNAME`
   - Value: Your Docker Hub username (e.g., `johnsmith`)

   **Secret 2:**
   - Name: `DOCKERHUB_TOKEN`
   - Value: The access token you created in Docker Hub

### 4. Trigger GitHub Action

The GitHub Action will automatically run when you push to the `main` branch.

To manually trigger it:
1. Go to **Actions** tab in your GitHub repository
2. Click on "Build and Push Docker Image" workflow
3. Click **Run workflow** → **Run workflow**

The workflow will:
- Build your Docker image
- Push it to Docker Hub as `your-username/emotion-detection-api:latest`
- This takes about 5-10 minutes

### 5. Deploy on RunPod

1. **Go to RunPod**:
   - Visit [RunPod.io](https://www.runpod.io/)
   - Sign in or create an account

2. **Deploy Container**:
   - Click **Deploy** in the top menu
   - Select **GPU Pod** or **CPU Pod** (GPU recommended)
   
3. **Configure Pod**:
   - **Container Image**: `your-dockerhub-username/emotion-detection-api:latest`
   - **Container Disk**: 10 GB minimum
   - **Expose HTTP Ports**: `8000`
   - **Expose TCP Ports**: Leave empty
   - **Environment Variables**: None needed (optional)
   
4. **Select GPU** (if using GPU pod):
   - RTX 3090 (good balance of price/performance)
   - RTX 4090 (faster, more expensive)
   - A100 (best performance, highest cost)
   - Or use **CPU** for lower traffic/cost

5. **Deploy**:
   - Click **Deploy On-Demand** or **Deploy Spot** (cheaper but can be interrupted)
   - Wait for pod to start (1-2 minutes)

### 6. Access Your API

Once deployed, RunPod provides:
- **Pod URL**: Something like `https://abcd1234-8000.proxy.runpod.net`

Test your endpoints:
- Health check: `https://YOUR-POD-URL/health`
- Interactive docs: `https://YOUR-POD-URL/docs`
- Single prediction: `POST https://YOUR-POD-URL/predict`
- Batch prediction: `POST https://YOUR-POD-URL/predict/batch`

### 7. Testing Your Deployed API

Using curl:
```bash
# Health check
curl https://YOUR-POD-URL/health

# Single prediction
curl -X POST https://YOUR-POD-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'

# Batch prediction
curl -X POST https://YOUR-POD-URL/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am happy!", "This is frustrating."]}'
```

Or use the interactive docs at `https://YOUR-POD-URL/docs`

## Updating Your Deployment

When you make changes to your code:

```bash
# Make your changes, then:
git add .
git commit -m "Description of changes"
git push
```

GitHub Actions will automatically:
1. Build the new Docker image
2. Push it to Docker Hub with the `:latest` tag

To use the updated version on RunPod:
1. Stop your current pod
2. Start a new pod with the same configuration
3. RunPod will pull the latest image automatically

## Cost Estimation

### Docker Hub
- **Free tier**: Unlimited public repositories ✅

### GitHub Actions
- **Free tier**: 2,000 minutes/month ✅
- Each build takes ~5-10 minutes

### RunPod Pricing (approximate)
- **CPU Pod**: $0.2-0.4/hour
- **RTX 3090**: $0.4-0.6/hour
- **RTX 4090**: $0.6-0.8/hour
- **A100**: $1.5-2.5/hour

**Spot instances**: Save 50-70% but can be interrupted

## Troubleshooting

### GitHub Action Fails
- Check that secrets are set correctly
- Verify Docker Hub credentials
- Check the Actions logs for specific errors

### RunPod Pod Won't Start
- Verify the Docker image exists on Docker Hub
- Check that port 8000 is exposed
- Ensure container disk is at least 10GB
- Check RunPod logs for errors

### API Not Responding
- Wait 2-3 minutes after pod starts (model loads on first request)
- Check health endpoint first: `/health`
- Verify the URL includes port: `https://xxx-8000.proxy.runpod.net`

### Out of Memory
- Use GPU pod instead of CPU
- Reduce batch size in requests
- Select pod with more VRAM (A100 has 40GB)

## Advanced: Continuous Deployment

To automatically restart RunPod when GitHub is updated:
1. Use RunPod API (requires RunPod API key)
2. Add a step to your GitHub Action to trigger pod restart
3. See [RunPod API docs](https://docs.runpod.io/reference/api-reference)

## Security Notes

- Never commit secrets to GitHub
- Use GitHub Secrets for sensitive data
- Docker Hub access tokens can be rotated
- Consider making Docker image private if needed

## Support

- GitHub Actions: Check the Actions tab for build logs
- RunPod: Check pod logs in RunPod dashboard
- API Issues: Check `/health` endpoint and pod logs
