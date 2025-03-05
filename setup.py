from setuptools import setup, find_packages

setup(
    name="reasoning_distillation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.2",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Research project for reasoning enhancement in language models",
    keywords="nlp, language-models, reasoning, ai",
    python_requires=">=3.9",
)
