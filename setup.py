#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    line.strip()
    for line in open("requirements.txt").readlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="maxa-ai",
    version="0.1.0",
    description="Maxa AI: Eternal Inference with Theory of Mind",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/maxa-ai",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai nlp machine-learning chatbot theory-of-mind",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/maxa-ai/issues",
        "Source": "https://github.com/yourusername/maxa-ai",
    },
)
