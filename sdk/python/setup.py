from setuptools import setup, find_packages

setup(
    name="payment-decision-sdk",
    version="1.0.0",
    description="High-throughput, low-latency Python client SDK for payment decisioning and gateway routing",
    author="Payment Decision Factory Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Point-Of-Sale",
    ],
)
