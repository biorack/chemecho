from setuptools import setup, find_packages

setup(
    name="chemecho",
    version="0.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "rdkit",
        "tqdm",
        "msbuddy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)