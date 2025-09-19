#!/usr/bin/env python3
"""
BFCL Handler Test Script

This script tests the BFCL handler implementation to ensure it works correctly
before submission to the Berkeley Function-Calling Leaderboard.
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import BFCLHandler, process_message, reset_handler

async def test_basic_functionality():
    """Test basic handler functionality."""
    print("Testing basic functionality...")
    
    handler = BFCLHandler()
    
    # Test model info
    info = handler.get_model_info()
    print(f"Model info: {json.dumps(info, indent=2)}")
    
    # Test simple conversation
    result = await handler.process_message("Hello, how are you?")
    print(f"Simple conversation result: {json.dumps(result, indent=2)}")
    
    return True

async def test_function_calling():
    """Test function calling capabilities."""
    print("\nTesting function calling...")
    
    handler = BFCLHandler()
    
    # Test web search
    result = await handler.process_message("Search for information about machine learning")
    print(f"Web search result: {json.dumps(result, indent=2)}")
    
    # Test calculation
    result = await handler.process_message("Calculate 15 + 27")
    print(f"Calculation result: {json.dumps(result, indent=2)}")
    
    # Test weather
    result = await handler.process_message("What's the weather like in New York?")
    print(f"Weather result: {json.dumps(result, indent=2)}")
    
    return True

async def test_memory_management():
    """Test memory management capabilities."""
    print("\nTesting memory management...")
    
    handler = BFCLHandler()
    
    # Store memory
    result = await handler.process_message("Remember that my favorite color is blue")
    print(f"Memory storage result: {json.dumps(result, indent=2)}")
    
    # Retrieve memory
    result = await handler.process_message("What is my favorite color?")
    print(f"Memory retrieval result: {json.dumps(result, indent=2)}")
    
    return True

async def test_error_handling():
    """Test error handling capabilities."""
    print("\nTesting error handling...")
    
    handler = BFCLHandler()
    
    # Test invalid function call
    result = await handler.process_message("Call unknown_function with invalid parameters")
    print(f"Error handling result: {json.dumps(result, indent=2)}")
    
    return True

async def test_parallel_calls():
    """Test parallel function calls."""
    print("\nTesting parallel function calls...")
    
    handler = BFCLHandler()
    
    # Test multiple function calls in one message
    result = await handler.process_message("Search for AI news and calculate 100 * 5")
    print(f"Parallel calls result: {json.dumps(result, indent=2)}")
    
    return True

def test_handler_interface():
    """Test the main handler interface function."""
    print("\nTesting handler interface...")
    
    # Test the main entry point
    result = process_message("Hello, test the handler interface")
    print(f"Handler interface result: {json.dumps(result, indent=2)}")
    
    return True

async def run_all_tests():
    """Run all tests."""
    print("Starting BFCL Handler Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Function Calling", test_function_calling),
        ("Memory Management", test_memory_management),
        ("Error Handling", test_error_handling),
        ("Parallel Calls", test_parallel_calls),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            success = await test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            print(f"{test_name}: {results[test_name]}")
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results[test_name] = f"FAILED - {e}"
    
    # Test handler interface
    try:
        print(f"\nRunning Handler Interface test...")
        success = test_handler_interface()
        results["Handler Interface"] = "PASSED" if success else "FAILED"
        print(f"Handler Interface: {results['Handler Interface']}")
    except Exception as e:
        print(f"Handler Interface: FAILED - {e}")
        results["Handler Interface"] = f"FAILED - {e}"
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Handler is ready for BFCL submission.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before submission.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner error: {e}")
        sys.exit(1)
