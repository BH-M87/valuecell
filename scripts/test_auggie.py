#!/usr/bin/env python3
"""
Test script for Auggie CLI integration.

This script helps verify that auggie is properly installed and configured.

Usage:
    python scripts/test_auggie.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ {text}")


def check_auggie_installed():
    """Check if auggie CLI is installed."""
    print_header("Checking Auggie Installation")
    
    try:
        result = subprocess.run(
            ["auggie", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Auggie is installed: {version}")
            return True
        else:
            print_error("Auggie command failed")
            return False
            
    except FileNotFoundError:
        print_error("Auggie is not installed")
        print_info("Install auggie:")
        print_info("  macOS/Linux: curl -fsSL https://install.augmentcode.com | sh")
        print_info("  Windows: irm https://install.augmentcode.com/windows | iex")
        return False
        
    except subprocess.TimeoutExpired:
        print_error("Auggie command timed out")
        return False


def check_auggie_auth():
    """Check if auggie is authenticated."""
    print_header("Checking Auggie Authentication")
    
    try:
        result = subprocess.run(
            ["auggie", "account"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("Auggie is authenticated")
            print_info("Account info:")
            for line in result.stdout.strip().split('\n')[:5]:  # Show first 5 lines
                print(f"  {line}")
            return True
        else:
            print_error("Auggie is not authenticated")
            print_info("Run: auggie login")
            return False
            
    except Exception as e:
        print_error(f"Failed to check authentication: {e}")
        return False


def check_env_config():
    """Check environment configuration."""
    print_header("Checking Environment Configuration")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print_error(".env file not found")
        print_info("Copy .env.example to .env and configure it")
        return False
    
    print_success(".env file exists")
    
    # Load .env file
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    # Check USE_AUGGIE
    use_auggie = env_vars.get('USE_AUGGIE', 'false')
    if use_auggie.lower() == 'true':
        print_success("USE_AUGGIE=true (auggie enabled)")
    else:
        print_info("USE_AUGGIE=false (auggie disabled)")
        print_info("Set USE_AUGGIE=true in .env to enable auggie")
    
    # Check model IDs
    model_vars = [
        'PLANNER_MODEL_ID',
        'RESEARCH_AGENT_MODEL_ID',
        'SEC_PARSER_MODEL_ID',
    ]
    
    print_info("\nConfigured models:")
    for var in model_vars:
        value = env_vars.get(var, 'not set')
        print(f"  {var}: {value}")
    
    return True


def test_auggie_client():
    """Test the auggie client."""
    print_header("Testing Auggie Client")
    
    try:
        from valuecell.utils.auggie_client import AuggieClient
        
        print_info("Creating auggie client...")
        client = AuggieClient(
            model="google/gemini-2.5-flash",
            max_turns=1,
            quiet=True
        )
        
        print_info("Sending test prompt...")
        result = client.invoke(
            "What is 2+2? Answer with just the number.",
            timeout=30
        )
        
        if result and "4" in str(result):
            print_success("Auggie client test passed")
            print_info(f"Response: {result}")
            return True
        else:
            print_error("Auggie client test failed")
            print_info(f"Unexpected response: {result}")
            return False
            
    except Exception as e:
        print_error(f"Auggie client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_factory():
    """Test the model factory with auggie."""
    print_header("Testing Model Factory Integration")
    
    # Save original env
    original_use_auggie = os.getenv("USE_AUGGIE")
    
    try:
        # Enable auggie
        os.environ["USE_AUGGIE"] = "true"
        os.environ["RESEARCH_AGENT_MODEL_ID"] = "google/gemini-2.5-flash"
        
        print_info("Importing model factory...")
        from valuecell.utils.model import get_model
        
        print_info("Getting model...")
        model = get_model("RESEARCH_AGENT_MODEL_ID")
        
        print_info(f"Model type: {type(model).__name__}")
        
        # Check if it's an auggie adapter
        if hasattr(model, 'client'):
            from valuecell.utils.auggie_client import AuggieClient
            if isinstance(model.client, AuggieClient):
                print_success("Model factory returns auggie adapter")
                return True
        
        print_error("Model factory did not return auggie adapter")
        return False
        
    except Exception as e:
        print_error(f"Model factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original env
        if original_use_auggie is not None:
            os.environ["USE_AUGGIE"] = original_use_auggie
        else:
            os.environ.pop("USE_AUGGIE", None)


def list_available_models():
    """List available models from auggie."""
    print_header("Available Models")
    
    try:
        result = subprocess.run(
            ["auggie", "model", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("Available models:")
            print(result.stdout)
            return True
        else:
            print_error("Failed to list models")
            return False
            
    except Exception as e:
        print_error(f"Failed to list models: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Auggie CLI Integration Test")
    print("=" * 60)
    
    results = []
    
    # Run checks
    results.append(("Installation", check_auggie_installed()))
    
    if results[-1][1]:  # Only check auth if installed
        results.append(("Authentication", check_auggie_auth()))
    
    results.append(("Environment", check_env_config()))
    
    # Run tests if auggie is available
    if all(r[1] for r in results[:2]):  # If installed and authenticated
        results.append(("Client Test", test_auggie_client()))
        results.append(("Model Factory", test_model_factory()))
        
        # List models
        list_available_models()
    
    # Print summary
    print_header("Test Summary")
    
    for name, passed in results:
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    # Overall result
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ All tests passed!")
        print("  Auggie integration is ready to use.")
    else:
        print("  ✗ Some tests failed.")
        print("  Please fix the issues above.")
    print("=" * 60 + "\n")
    
    # Print next steps
    if all_passed:
        print("Next steps:")
        print("  1. Set USE_AUGGIE=true in your .env file")
        print("  2. Run: python examples/auggie_example.py")
        print("  3. Start using ValueCell with auggie!")
    else:
        print("Troubleshooting:")
        print("  1. Make sure auggie is installed: auggie --version")
        print("  2. Authenticate with auggie: auggie login")
        print("  3. Check your .env configuration")
        print("  4. See docs/AUGGIE_INTEGRATION.md for details")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

