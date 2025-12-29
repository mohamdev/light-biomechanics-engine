"""Setup script for lean-bio-engine package."""

from setuptools import setup, find_packages

setup(
    name="lean-bio-engine",
    version="0.1.0",
    description="Lightweight biomechanics inference pipeline for mobile deployment",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "peft>=0.7.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bioengine-benchmark=scripts.benchmark:main",
            "bioengine-demo=scripts.demo:main",
        ],
    },
)
