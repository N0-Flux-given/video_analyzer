from setuptools import setup, find_packages

setup(
    name="analyze_video",  # Package name
    version="0.1.0",  # Initial version
    description="Utilities for using LLaVA with Ollama",
    author="Your Name",
    packages=find_packages(),  # Automatically find modules
    install_requires=[
        "requests",
        "matplotlib",
    ],
    python_requires=">=3.8",  # Or your required version
)
