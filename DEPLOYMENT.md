# BFCL Submission Deployment Instructions

## Prerequisites

1. GitHub account
2. Git installed
3. Python 3.8+ with required dependencies
4. Apache 2.0 License compliance
5. aiboards application submission

## Deployment Steps

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Create a new repository named `bfcl_submission`
3. Set visibility to "Public"
4. Initialize with README (optional)

### 2. Clone and Setup

```bash
# Clone your repository
git clone https://github.com/sweetdream0530/bfcl_submission.git
cd bfcl_submission

# Copy files from this directory
cp -r /path/to/bfcl_submission/* .

# Install dependencies
pip install -r requirements.txt
```

### 3. Test Locally

```bash
# Run tests
python test_handler.py

# Test the handler directly
python -c "from handler import process_message; print(process_message('Hello, test the handler'))"
```

### 4. Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial BFCL submission"

# Push to GitHub
git push origin main
```

### 5. Submit to BFCL

1. Submit via the [aiboards application](https://aiboards.com)
2. Provide your GitHub repository URL: https://github.com/sweetdream0530/bfcl_submission
3. Ensure Apache 2.0 license compliance
4. Declare mode as "fc" (native function-calling)
5. Disclose base checkpoint: microsoft/DialoGPT-medium
6. Disclose parameter count: ~345M parameters
7. Wait for evaluation results

## File Structure

```
bfcl_submission/
├── handler.py              # Main handler implementation
├── requirements.txt        # Python dependencies
├── test_handler.py         # Test script
├── README.md              # Repository README
├── MODEL_CARD.md          # Model card
├── DEPLOYMENT.md          # This file
├── LICENSE                # Apache 2.0 License
├── SUBMISSION_CHECKLIST.md # Compliance checklist
├── model/
│   ├── config.json        # Model configuration
│   └── README.md          # Model directory README
└── .gitignore             # Git ignore file
```

## Important Notes

- The `handler.py` file must be at the repository root
- All dependencies must be listed in `requirements.txt`
- The model must be publicly accessible
- Must be licensed under Apache 2.0
- Must be submitted via aiboards application
- Must declare mode as "fc" (native function-calling)
- Must disclose base checkpoint and parameter count
- Uses only BFCL-styled tools (no external services)
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
