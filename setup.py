from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="llmize",
    version="0.1.0",
    author="Rizki Oktavian",
    author_email="rizki@bwailabs.com",
    description="A Python package for LLM-based optimization tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rizkiokt/llmize",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
) 