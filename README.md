# BFCL Submission

This repository contains a submission for the Berkeley Function-Calling Leaderboard (BFCL).

## Model Information

- **Mode**: `fc` (native function-calling)
- **Base Model**: microsoft/DialoGPT-medium
- **Parameter Count**: ~345M parameters
- **License**: Apache 2.0
- **Fine-tuning**: None (base model used as-is)

## Repository Structure

```
bfcl_submission/
├── handler.py              # BFCL handler interface implementation
├── model/                  # Model weights and configuration
├── requirements.txt        # Python dependencies
├── test_handler.py         # Test script
├── LICENSE                # Apache 2.0 License
├── SUBMISSION_CHECKLIST.md # Complete compliance checklist
└── README.md              # This file
```

## Usage

The `handler.py` file implements the BFCL handler interface and can be used for evaluation on the Berkeley Function-Calling Leaderboard.

## Evaluation

This model will be automatically evaluated by the BFCL team using the pinned evaluator version for reproducibility.

## GitHub Repository

This submission is hosted on GitHub: https://github.com/sweetdream0530/bfcl_submission

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
