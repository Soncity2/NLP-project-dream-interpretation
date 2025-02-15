from setuptools import setup, find_packages

setup(
    name="dream_interpreter",
    version="1.0.0",
    author="Soncity2",
    author_email="son.xoxo@gmail.com",
    description="Fine-tuning LLaMA 2 for Dream Interpretation using NLP.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Soncity2/NLP-project-dream-interpretation",  # Replace with your repo URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "yaml",
        "tqdm",
        "pypdf2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "fine_tune = scripts.fine_tune:fine_tune",
            "tokenize_dataset = scripts.tokenize_dataset:tokenize_dataset",
            "evaluate = scripts.evaluate:evaluate",
            "dataset_preparation = scripts.dataset_preparation:prepare_dataset",
            "pdf_processing = scripts.pdf_processing:process_pdfs"
        ]
    },
)
