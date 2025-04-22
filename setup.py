from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmize",
    version="0.1.1",
    author="Rizki Oktavian",
    author_email="rizki@bwailabs.com",
    description="A Python package for LLM-based optimization tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rizkiokt/llmize",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    include_package_data=True,
) 