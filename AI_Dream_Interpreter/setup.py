from setuptools import setup, find_packages

setup(
    name="pdf_to_llama_finetune",
    version="1.0.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A pipeline for fine-tuning LLaMA 2 on PDF text data",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "PyPDF2",
        "fastapi",
        "uvicorn",
        "pydantic",
        "yaml",
        "bitsandbytes",
        "peft",
        "accelerate"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
