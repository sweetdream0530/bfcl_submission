"""
BFCL BitAgent Handler Module - Direct Import

This module directly imports BitAgentHandler from the root handler.py file.
This is the module that the BFCL evaluation system will import from.
"""

import sys
import os
from pathlib import Path

# Add the root directory to the Python path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Direct import of BitAgentHandler from handler.py
try:
    from handler import BitAgentHandler
    print(f"Successfully imported BitAgentHandler from handler.py")
except ImportError as e:
    print(f"Failed to import BitAgentHandler from handler.py: {e}")
    # Create a fallback BitAgentHandler
    try:
        from handler import BFCLHandler
        
        class BitAgentHandler(BFCLHandler):
            """Fallback BitAgentHandler wrapper."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                print("Fallback BitAgentHandler initialized")
    except ImportError as e2:
        print(f"Failed to import BFCLHandler: {e2}")
        raise ImportError("Could not import any handler class from handler.py")

# Export the BitAgentHandler class
__all__ = ['BitAgentHandler']
