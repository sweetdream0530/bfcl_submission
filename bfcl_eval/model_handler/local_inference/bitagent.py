"""
BFCL BitAgent Handler Module

This module provides the BitAgentHandler class that the BFCL evaluation system expects.
It imports and wraps the main BFCLHandler from the root handler.py file.
"""

import sys
import os
from pathlib import Path

# Add the root directory to the Python path so we can import handler.py
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Lazy import to avoid torch/CUDA issues during module loading
def _get_bitagent_handler():
    """Lazy import of BitAgentHandler to avoid torch/CUDA issues during module loading."""
    try:
        from handler import BitAgentHandler
        return BitAgentHandler
    except ImportError:
        # Fallback: try to import BFCLHandler and create BitAgentHandler wrapper
        try:
            from handler import BFCLHandler
            
            class BitAgentHandler(BFCLHandler):
                """
                Wrapper class to make BFCLHandler compatible with BFCL evaluation system
                that expects a BitAgentHandler class.
                """
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    print("BitAgentHandler wrapper initialized - delegating to BFCLHandler")
                
                # All methods are inherited from BFCLHandler
            return BitAgentHandler
        except ImportError as e:
            print(f"Error importing handler: {e}")
            raise ImportError("Could not import handler.py. Make sure it exists in the root directory.")

# Create a module-level BitAgentHandler class that will be imported when needed
class BitAgentHandler:
    """Placeholder class that will be replaced with the actual implementation when imported."""
    
    def __new__(cls, *args, **kwargs):
        # Get the actual BitAgentHandler class
        actual_class = _get_bitagent_handler()
        # Create an instance of the actual class
        return actual_class(*args, **kwargs)

# Export the BitAgentHandler class
__all__ = ['BitAgentHandler']
