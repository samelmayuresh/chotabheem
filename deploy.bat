@echo off
REM Google Cloud Run Deployment Script for Emotion AI (Windows)
REM Make sure you have gcloud CLI installed and authenticated

echo 🚀 Starting Google Cloud Run Deployment...

REM Configuration
set PROJECT_ID=your-project-id
set SERVICE_NAME=emotion-ai
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo ❌ gcloud CLI is not installed. Please install it first:
    echo https://cloud.google.com/sdk/docs/install
    pause
    exit /b 1
)

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install it first:
    echo https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

echo 📋 Project ID: %PROJECT_ID%
echo 🏷️  Service Name: %SERVICE_NAME%
echo 🌍 Region: %REGION%

REM Prompt for project ID if not set
if "%PROJECT_ID%"=="your-project-id" (
    echo ⚠️  Please update PROJECT_ID in this script with your actual Google Cloud Project ID
    set /p PROJECT_ID=Enter your Google Cloud Project ID: 
    set IMAGE_NAME=gcr.io/!PROJECT_ID!/!SERVICE_NAME!
)

REM Set the project
echo 🔧 Setting Google Cloud project...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo 🔌 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Build the Docker image
echo 🏗️  Building Docker image...
docker build -t %IMAGE_NAME% .

REM Push to Google Container Registry
echo 📤 Pushing image to Container Registry...
docker push %IMAGE_NAME%

REM Deploy to Cloud Run
echo 🚀 Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_NAME% ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 2Gi ^
    --cpu 2 ^
    --timeout 900 ^
    --max-instances 10 ^
    --set-env-vars PYTHONUNBUFFERED=1

REM Get the service URL
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo.
echo 🎉 Deployment completed successfully!
echo 🌐 Your Emotion AI app is now live at:
echo    %SERVICE_URL%
echo.
echo 📊 Service details:
echo    Project: %PROJECT_ID%
echo    Service: %SERVICE_NAME%
echo    Region: %REGION%
echo    Image: %IMAGE_NAME%
echo.
echo 🔧 To update the deployment, run this script again.
echo 🗑️  To delete the service: gcloud run services delete %SERVICE_NAME% --region %REGION%

pause