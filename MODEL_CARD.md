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

The model is designed to be used with the BFCL evaluation framework. The main entry point is the  function in .

## Evaluation

This model will be automatically evaluated by the BFCL team using their pinned evaluator version.

## License

[Specify your license here]

## Citation

If you use this model, please cite:

```
@misc{bfcl_submission_2024,
  title={BFCL Submission Model},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/your-username/your-model-name}
}
```
