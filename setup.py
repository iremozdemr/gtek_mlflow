from setuptools import find_packages, setup

setup(
    name='mlflowlib',
    packages=find_packages(include=['mlflowlib']),
    version='0.2',
    license='MIT',    
    description='a general-purpose python package with MLflow integration',
    author='irem ozdemir',
    author_email = 'iremozdemirwww3@gmail.com', 
    keywords='["machine learning","deep learning","mlflow","tensorflow","model training"]',
    install_requires=[
        'mlflow>=1.21.0', 
        'tensorflow>=2.5.0',  
    ],
    # url = 'https://github.com/iremozdemr/gtekmlflow', 
    # download_url = 'https://github.com/iremozdemr/gtekmlflow/archive/refs/tags/0.2.tar.gz',
)