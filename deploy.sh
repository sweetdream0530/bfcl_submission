#!/bin/bash

# BFCL Submission Deployment Script
# This script prepares the BFCL submission for deployment to Hugging Face

set -e

echo "BFCL Submission Deployment Script"
echo "================================="

# Check if we're in the right directory
if [ ! -f "handler.py" ]; then
    echo "Error: handler.py not found. Please run this script from the bfcl_submission directory."
    exit 1
fi

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model weights (if large)
model/model_weights/
*.bin
*.safetensors

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
EOF

# Create model card
echo "Creating model card..."
cat > MODEL_CARD.md << EOF
# BFCL Submission Model

## Model Description

This model is a submission for the Berkeley Function-Calling Leaderboard (BFCL), designed to evaluate LLM function-calling capabilities.

## Model Details

- **Model Type**: Causal Language Model
- **Base Model**: microsoft/DialoGPT-medium
- **Mode**: fc (native function-calling)
- **Capabilities**: Function calling, web search, memory management, mathematical calculations

## Function Calling Capabilities

The model can execute the following functions:

1. **web_search**: Search the web for information
2. **get_weather**: Get current weather information
3. **calculate**: Perform mathematical calculations
4. **store_memory**: Store information in memory
5. **retrieve_memory**: Retrieve information from memory

## Usage

The model is designed to be used with the BFCL evaluation framework. The main entry point is the `process_message` function in `handler.py`.

## Evaluation

This model will be automatically evaluated by the BFCL team using their pinned evaluator version.

## License

[Specify your license here]

## Citation

If you use this model, please cite:

\`\`\`
@misc{bfcl_submission_2024,
  title={BFCL Submission Model},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/your-username/your-model-name}
}
\`\`\`
EOF

# Create deployment instructions
echo "Creating deployment instructions..."
cat > DEPLOYMENT.md << EOF
# BFCL Submission Deployment Instructions

## Prerequisites

1. Hugging Face account
2. Git installed
3. Python 3.8+ with required dependencies

## Deployment Steps

### 1. Create Hugging Face Repository

1. Go to [Hugging Face](https://huggingface.co)
2. Create a new model repository
3. Choose "Model" as the repository type
4. Set visibility to "Public"

### 2. Clone and Setup

\`\`\`bash
# Clone your repository
git clone https://huggingface.co/your-username/your-model-name
cd your-model-name

# Copy files from this directory
cp -r /path/to/bfcl_submission/* .

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 3. Test Locally

\`\`\`bash
# Run tests
python test_handler.py

# Test the handler directly
python -c "from handler import process_message; print(process_message('Hello, test the handler'))"
\`\`\`

### 4. Commit and Push

\`\`\`bash
# Add all files
git add .

# Commit
git commit -m "Initial BFCL submission"

# Push to Hugging Face
git push
\`\`\`

### 5. Submit to BFCL

1. Go to the [BFCL submission page](https://gorilla.cs.berkeley.edu/leaderboard)
2. Submit your Hugging Face repository URL
3. Wait for evaluation results

## File Structure

\`\`\`
your-model-name/
├── handler.py              # Main handler implementation
├── requirements.txt        # Python dependencies
├── test_handler.py         # Test script
├── README.md              # Repository README
├── MODEL_CARD.md          # Model card
├── DEPLOYMENT.md          # This file
├── model/
│   ├── config.json        # Model configuration
│   └── README.md          # Model directory README
└── .gitignore             # Git ignore file
\`\`\`

## Important Notes

- The `handler.py` file must be at the repository root
- All dependencies must be listed in `requirements.txt`
- The model must be publicly accessible
- Test your submission locally before deploying

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Model loading errors**: Check if the base model is accessible
3. **Function call parsing errors**: Verify the function call format in your prompts

### Getting Help

- Check the [BFCL documentation](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- Review the test output for specific error messages
- Ensure your handler follows the BFCL interface requirements
EOF

# Make test script executable
chmod +x test_handler.py

echo ""
echo "Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review the files in this directory"
echo "2. Update the model information in model/config.json"
echo "3. Run 'python test_handler.py' to test the handler"
echo "4. Follow the instructions in DEPLOYMENT.md to deploy to Hugging Face"
echo "5. Submit your repository URL to BFCL"
echo ""
echo "Files created:"
echo "- .gitignore"
echo "- MODEL_CARD.md"
echo "- DEPLOYMENT.md"
echo "- Made test_handler.py executable"
echo ""
echo "Ready for deployment!"
