from setuptools import find_packages, setup

setup(
    name='gtek_mlflow',
    packages=find_packages(include=['gtek_mlflow']),
    version='0.1',
    license='MIT',    
    description='a general-purpose python package with MLflow integration',
    author='irem ozdemir',
    author_email = 'iremozdemirwww3@gmail.com', 
    keywords='["machine learning","deep learning","mlflow","tensorflow","model training"]',
    install_requires=[
        'mlflow>=1.21.0', 
        'tensorflow>=2.5.0',  
    ],
    url = '', 
    download_url = '',
)