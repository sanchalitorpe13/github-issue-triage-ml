from setuptools import setup, find_packages

setup(
    name="triage-ml",
    version="2.0.0",
    description="AI-powered GitHub Issue Classification and Priority Prediction Platform",
    author="GitHub Issue Triage ML Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
)
