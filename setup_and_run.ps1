# SmallVille Resource Scramble - Quick Start Guide

Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "SmallVille: Resource Scramble Experiment Setup"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""

# Check Python
Write-Host "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.8+." -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host "Checking pip..."
try {
    $pipVersion = pip --version 2>&1
    Write-Host "  ✓ pip installed" -ForegroundColor Green
} catch {
    Write-Host "  ✗ pip not found!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check API key
Write-Host ""
Write-Host "Checking DeepSeek API key..."
if ($env:DEEPSEEK_API_KEY) {
    Write-Host "  ✓ API key is set" -ForegroundColor Green
} else {
    Write-Host "  ✗ API key not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please set your DeepSeek API key:"
    Write-Host "  `$env:DEEPSEEK_API_KEY = 'your-api-key-here'" -ForegroundColor Cyan
    Write-Host ""
    $response = Read-Host "Do you want to enter your API key now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        $apiKey = Read-Host "Enter your DeepSeek API key"
        $env:DEEPSEEK_API_KEY = $apiKey
        Write-Host "  ✓ API key set for this session" -ForegroundColor Green
    } else {
        Write-Host "  Please set the API key before running the experiment." -ForegroundColor Yellow
        exit 0
    }
}

# Ready to run
Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "Setup Complete! Ready to run experiment." -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""
Write-Host "To start the experiment, run:"
Write-Host "  python run_experiment.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:"
Write-Host "  - Vocabulary: 20 abstract tokens (tok1-tok20)"
Write-Host "  - Max message length: 5 tokens"
Write-Host "  - Episodes: 50 (configurable in run_experiment.py)"
Write-Host "  - Curriculum: 3 phases"
Write-Host ""

$response = Read-Host "Start experiment now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    python run_experiment.py
}
