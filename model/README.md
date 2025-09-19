# BFCL Model Configuration

This directory contains the model configuration and weights for the BFCL submission.

## Files

- `config.json`: Model configuration and metadata
- `model_weights/`: Directory containing model weights (if using custom fine-tuned model)

## Model Information

- **Base Model**: microsoft/DialoGPT-medium
- **Mode**: fc (native function-calling)
- **Capabilities**: Function calling, web search, memory management, calculations

## Usage

The model will be automatically loaded by the handler.py when the BFCL evaluator runs the submission.

## Custom Model

If you have a fine-tuned model, place the model weights in the `model_weights/` directory and update the `config.json` file accordingly.
