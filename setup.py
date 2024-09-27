from setuptools import setup, find_packages

setup(
    name="graphrag-chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "PyYAML",
        "requests",
        "openai",
        "ollama",
        "spacy==3.5.0",
        "streamlit",
        "sqlite-utils",
        "sqlite-vec",
        "PyPDF2",
        "langchain",
        "langchain-community",
        "tiktoken",
        "langchain_experimental",
        "json-repair",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "graphrag-api=api.api:main",
            "graphrag-app=app.app:main",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A GraphRAG-based chatbot system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphrag-chatbot",
)