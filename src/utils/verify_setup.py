#!/usr/bin/env python3
"""
Verification script for Checkpoint 0 setup
"""

import sys
import importlib
import os

def check_package(package_name):
    """Check if a package is installed and importable"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'src',
        'src/neural_nets',
        'src/rnn_lstm', 
        'src/attention',
        'src/transformer',
        'src/data',
        'src/training',
        'src/models',
        'src/api',
        'src/interface',
        'src/utils',
        'data',
        'models',
        'tests'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Directory {dir_name} is missing")
            all_exist = False
    
    return all_exist

def main():
    print("🔍 Verifying Checkpoint 0 Setup...\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    print("\n📦 Checking required packages:")
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'fastapi', 'uvicorn', 'gradio', 'transformers',
        'tokenizers', 'datasets', 'tqdm', 'wandb'
    ]
    
    all_packages_installed = True
    for package in required_packages:
        if not check_package(package):
            all_packages_installed = False
    
    # Check directory structure
    print("\n📁 Checking directory structure:")
    dirs_exist = check_directory_structure()
    
    # Summary
    print("\n" + "="*50)
    if all_packages_installed and dirs_exist:
        print("✅ Setup verification PASSED!")
        print("You can proceed to Checkpoint 1")
    else:
        print("❌ Setup verification FAILED!")
        print("Please fix the issues above before proceeding")
    
    return all_packages_installed and dirs_exist

if __name__ == "__main__":
    main()