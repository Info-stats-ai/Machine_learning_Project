from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->list[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
setup(
    name='mlproject',
    packages=find_packages(),
    version='0.1.0',
    description='End to End ML project',
    author='Omkar Thakur',
    author_email='othakur@umd.edu',
    install_requires=get_requirements('requirements.txt'),
)
