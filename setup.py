from setuptools import setup, find_packages

setup(
    name="Tr4der",
    version="14.0.0",
    author="Sean Mullins",
    author_email="smullins998@gmail.com",
    description="Tr4der is a algorithmic trading library for quantitative strategy ideation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/smullins998/tr4der",
    packages=find_packages(),
    package_data={
        'tr4der': ['*.png', '*.jpg', '*.jpeg', '*.gif'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy",
        "openai",
        "pandas",
        "PyYAML",
        "seaborn",
        "yfinance",
        "tensorflow",
        "scikit-learn",
    ],
)
