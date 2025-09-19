#!/usr/bin/env python3
"""
BFCL Submission HuggingFace Deployment Script

This script deploys your BFCL submission to HuggingFace Hub.
Make sure you have a HuggingFace account and token ready.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
import json

def main():
    print("BFCL Submission HuggingFace Deployment")
    print("=====================================")
    
    # Check if we're in the right directory
    if not Path("handler.py").exists():
        print("Error: handler.py not found. Please run this script from the bfcl_submission directory.")
        sys.exit(1)
    
    # Get HuggingFace credentials
    print("\nHuggingFace Authentication Required")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' permissions")
    print("3. Enter your token below:")
    
    token = input("HuggingFace Token: ").strip()
    if not token:
        print("Error: Token is required")
        sys.exit(1)
    
    # Login to HuggingFace
    try:
        login(token=token)
        print("✓ Successfully authenticated with HuggingFace")
    except Exception as e:
        print(f"Error authenticating: {e}")
        sys.exit(1)
    
    # Get username and model name
    print("\nRepository Information")
    username = input("HuggingFace Username: ").strip()
    if not username:
        print("Error: Username is required")
        sys.exit(1)
    
    model_name = input("Model Name (default: bfcl-submission): ").strip()
    if not model_name:
        model_name = "bfcl-submission"
    
    repo_id = f"{username}/{model_name}"
    repo_url = f"https://huggingface.co/{repo_id}"
    
    print(f"\nRepository Details:")
    print(f"  Repository ID: {repo_id}")
    print(f"  Repository URL: {repo_url}")
    
    # Confirm deployment
    confirm = input("\nProceed with deployment? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Deployment cancelled")
        sys.exit(0)
    
    # Initialize HuggingFace API
    api = HfApi()
    
    try:
        # Create repository
        print(f"\nCreating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print("✓ Repository created successfully")
        
        # Prepare files for upload
        print("\nPreparing files for upload...")
        
        # Files to upload (excluding unnecessary files)
        files_to_upload = [
            "handler.py",
            "requirements.txt", 
            "test_handler.py",
            "README.md",
            "MODEL_CARD.md",
            "LICENSE",
            "model/config.json",
            "model/README.md"
        ]
        
        # Check which files exist
        existing_files = []
        for file_path in files_to_upload:
            if Path(file_path).exists():
                existing_files.append(file_path)
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        
        if not existing_files:
            print("Error: No files found to upload")
            sys.exit(1)
        
        # Upload files
        print(f"\nUploading {len(existing_files)} files...")
        for file_path in existing_files:
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✓ Uploaded {file_path}")
            except Exception as e:
                print(f"  ✗ Failed to upload {file_path}: {e}")
        
        # Create README.md for HuggingFace
        print("\nCreating HuggingFace README...")
        hf_readme_content = f"""---
license: apache-2.0
base_model: microsoft/DialoGPT-medium
tags:
- bfcl
- function-calling
- dialogpt
- berkeley-function-calling-leaderboard
---

# BFCL Submission Model

This model is a submission for the Berkeley Function-Calling Leaderboard (BFCL), designed to evaluate LLM function-calling capabilities.

## Model Details

- **Model Type**: Causal Language Model
- **Base Model**: microsoft/DialoGPT-medium
- **Mode**: fc (native function-calling)
- **Parameter Count**: ~345M parameters
- **License**: Apache 2.0

## Function Calling Capabilities

The model can execute the following functions:

1. **web_search**: Search the web for information
2. **get_weather**: Get current weather information
3. **calculate**: Perform mathematical calculations
4. **store_memory**: Store information in memory
5. **retrieve_memory**: Retrieve information from memory

## Usage

The model is designed to be used with the BFCL evaluation framework. The main entry point is the `process_message` function in `handler.py`.

```python
from handler import process_message

# Process a message
result = process_message("Hello, how are you?")
print(result)
```

## Evaluation

This model will be automatically evaluated by the BFCL team using their pinned evaluator version.

## Repository Information

- **GitHub**: https://github.com/sweetdream0530/bfcl_submission
- **HuggingFace**: {repo_url}
- **BFCL Submission**: Use this HuggingFace URL for BFCL evaluation

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
        
        # Upload README
        api.upload_file(
            path_or_fileobj=hf_readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✓ HuggingFace README uploaded")
        
        print(f"\n🎉 Deployment completed successfully!")
        print(f"\nYour BFCL submission is now available at:")
        print(f"  {repo_url}")
        print(f"\nUse this URL for your BFCL submission:")
        print(f"  {repo_url}")
        
        # Save the URL for reference
        with open("huggingface_url.txt", "w") as f:
            f.write(repo_url)
        print(f"\n✓ HuggingFace URL saved to huggingface_url.txt")
        
    except Exception as e:
        print(f"\nError during deployment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
