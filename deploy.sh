#!/bin/bash

# Google Cloud Run Deployment Script for Emotion AI
# Make sure you have gcloud CLI installed and authenticated

set -e

echo "🚀 Starting Google Cloud Run Deployment..."

# Configuration
PROJECT_ID="your-project-id"  # Replace with your actual project ID
SERVICE_NAME="emotion-ai"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first:"
    echo "https://docs.docker.com/get-docker/"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"
echo "🏷️  Service Name: $SERVICE_NAME"
echo "🌍 Region: $REGION"

# Prompt for project ID if not set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "⚠️  Please update PROJECT_ID in this script with your actual Google Cloud Project ID"
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
fi

# Set the project
echo "🔧 Setting Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -t $IMAGE_NAME .

# Push to Google Container Registry
echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 10 \
    --set-env-vars PYTHONUNBUFFERED=1

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo "🎉 Deployment completed successfully!"
echo "🌐 Your Emotion AI app is now live at:"
echo "   $SERVICE_URL"
echo ""
echo "📊 Service details:"
echo "   Project: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"
echo "   Region: $REGION"
echo "   Image: $IMAGE_NAME"
echo ""
echo "🔧 To update the deployment, run this script again."
echo "🗑️  To delete the service: gcloud run services delete $SERVICE_NAME --region $REGION"