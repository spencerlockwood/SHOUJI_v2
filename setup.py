from setuptools import setup, find_packages

setup(
    name='shouji',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'edlib>=1.3.9',
        'biopython>=1.79',
        'tqdm>=4.62.0',
    ],
)
