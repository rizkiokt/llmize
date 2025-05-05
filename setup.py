from setuptools import setup, find_packages

setup(
    name="llmize",
    version="0.1.4",
    packages=find_packages(include=['llmize', 'llmize.*']),
    install_requires=[
        "numpy>=1.21.0",
        "google-genai>=1.5.0",
        "colorama>=0.4.6",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
    },
    python_requires=">=3.11",
    author="Rizki Oktavian",
    author_email="rizki@bwailabs.com",
    description="A library to use Large Language Models (LLMs) as numerical optimizers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rizkiokt/llmize",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 