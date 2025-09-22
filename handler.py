"""
BFCL BitAgent Handler - Direct Implementation

This handler is designed to work directly with the BFCL evaluation system.
It provides the exact interface that the BFCL system expects.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

# Try to import torch, but handle gracefully if CUDA is not available
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitAgentHandler:
    """
    BitAgent Handler for BFCL evaluation system.
    
    This class provides the exact interface expected by the BFCL evaluation system.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the BitAgent handler."""
        logger.info("BitAgentHandler initialized")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.memory_store = {}
        
        # Initialize model components
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = "microsoft/DialoGPT-medium"
            
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
            self.model = None
            self.tokenizer = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        This method is required by the BFCL evaluation system.
        """
        return {
            "model_name": "microsoft/DialoGPT-medium",
            "model_type": "causal_language_model", 
            "device": self.device,
            "parameters": "~345M",
            "license": "Apache 2.0",
            "mode": "fc",
            "initialized": self.model is not None and self.tokenizer is not None
        }
    
    async def process_message(self, message: str) -> str:
        """
        Process a message and return a response.
        This is the main interface expected by the BFCL evaluation system.
        """
        logger.info(f"Processing message: {message}")
        
        try:
            # Check if this is a function call
            if self._is_function_call(message):
                return await self._handle_function_call(message)
            else:
                return await self._handle_regular_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _is_function_call(self, message: str) -> bool:
        """Check if the message contains a function call."""
        function_keywords = [
            "web_search", "search", "find", "look for",
            "get_weather", "weather", "temperature",
            "calculate", "compute", "math", "+", "-", "*", "/",
            "store_memory", "remember", "save",
            "retrieve_memory", "recall", "what is"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in function_keywords)
    
    async def _handle_function_call(self, message: str) -> str:
        """Handle function call requests."""
        message_lower = message.lower()
        
        try:
            if any(keyword in message_lower for keyword in ["web_search", "search", "find", "look for"]):
                query = self._extract_query(message)
                return await self._web_search(query)
            elif any(keyword in message_lower for keyword in ["get_weather", "weather", "temperature"]):
                location = self._extract_location(message)
                return await self._get_weather(location)
            elif any(keyword in message_lower for keyword in ["calculate", "compute", "math"]) or any(op in message for op in ["+", "-", "*", "/", "="]):
                expression = self._extract_expression(message)
                return await self._calculate(expression)
            elif any(keyword in message_lower for keyword in ["store_memory", "remember", "save"]):
                key, value = self._extract_memory_data(message)
                return await self._store_memory(key, value)
            elif any(keyword in message_lower for keyword in ["retrieve_memory", "recall", "what is"]):
                key = self._extract_memory_key(message)
                return await self._retrieve_memory(key)
            else:
                return await self._handle_regular_message(message)
        except Exception as e:
            logger.error(f"Error handling function call: {e}")
            return f"Error processing function call: {str(e)}"
    
    async def _handle_regular_message(self, message: str) -> str:
        """Handle regular conversational messages."""
        try:
            if self.model and self.tokenizer:
                # Use the model to generate a response
                input_text = f"Human: {message}\nAssistant:"
                inputs = self.tokenizer.encode(input_text, return_tensors='pt')
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(input_text):].strip()
                
                return response if response else "I understand your message."
            else:
                return "I understand your message, but I'm not fully initialized yet."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I understand your message."
    
    def _extract_query(self, message: str) -> str:
        """Extract search query from message."""
        query = message.lower()
        for keyword in ["web_search", "search for", "find", "look for", "search"]:
            query = query.replace(keyword, "")
        return query.strip()
    
    def _extract_location(self, message: str) -> str:
        """Extract location from weather request."""
        location = message.lower()
        for keyword in ["get_weather", "weather in", "weather for", "weather at", "weather"]:
            location = location.replace(keyword, "")
        return location.strip()
    
    def _extract_expression(self, message: str) -> str:
        """Extract mathematical expression from message."""
        import re
        # Look for mathematical expressions
        math_pattern = r'[\d\+\-\*\/\(\)\.\s]+'
        matches = re.findall(math_pattern, message)
        if matches:
            return max(matches, key=len).strip()
        return message.replace("calculate", "").strip()
    
    def _extract_memory_data(self, message: str) -> tuple:
        """Extract key-value pair from memory storage request."""
        if ":" in message:
            parts = message.split(":", 1)
            return parts[0].strip(), parts[1].strip()
        else:
            # Extract from "remember that X is Y" pattern
            import re
            pattern = r"remember\s+that\s+(.+?)\s+is\s+(.+)"
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip(), match.group(2).strip()
            return "key", message.replace("remember", "").strip()
    
    def _extract_memory_key(self, message: str) -> str:
        """Extract memory key from retrieval request."""
        key = message.lower()
        for keyword in ["retrieve_memory", "what is", "what's", "recall", "remember"]:
            key = key.replace(keyword, "")
        return key.strip()
    
    async def _web_search(self, query: str) -> str:
        """Perform web search."""
        try:
            # Mock web search - in a real implementation, this would use a real search API
            return f"Web search results for '{query}': This is a mock search result. In a real implementation, this would return actual search results from the web."
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    async def _get_weather(self, location: str) -> str:
        """Get weather information."""
        try:
            # Mock weather - in a real implementation, this would use a real weather API
            return f"Weather in {location}: Sunny, 22°C, Light winds. This is mock weather data. In a real implementation, this would return actual weather information."
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    async def _calculate(self, expression: str) -> str:
        """Perform mathematical calculation."""
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator
            
            # Define safe operations
            safe_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Expression):
                    return safe_eval(node.body)
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    return safe_operators[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = safe_eval(node.operand)
                    return safe_operators[type(node.op)](operand)
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
            
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    async def _store_memory(self, key: str, value: str) -> str:
        """Store information in memory."""
        try:
            self.memory_store[key] = value
            return f"Stored '{key}' in memory"
        except Exception as e:
            return f"Error storing memory: {str(e)}"
    
    async def _retrieve_memory(self, key: str) -> str:
        """Retrieve information from memory."""
        try:
            if key in self.memory_store:
                return f"Memory '{key}': {self.memory_store[key]}"
            else:
                return f"Memory '{key}' not found"
        except Exception as e:
            return f"Error retrieving memory: {str(e)}"

# Alias for backward compatibility
BFCLHandler = BitAgentHandler

# Test function
if __name__ == "__main__":
    async def test_handler():
        handler = BitAgentHandler()
        
        # Test model info
        info = handler.get_model_info()
        print("Model info:", info)
        
        # Test basic message
        result = await handler.process_message("Hello, how are you?")
        print("Basic message result:", result)
        
        # Test function calls
        result = await handler.process_message("Calculate 2 + 2")
        print("Calculate result:", result)
        
        result = await handler.process_message("Remember that my name is John")
        print("Memory storage result:", result)
        
        result = await handler.process_message("What is my name?")
        print("Memory retrieval result:", result)
    
    asyncio.run(test_handler())
