from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    '''
    This function returns the list of requirements
    '''

    requirements = []
    with open(file_path) as f:
        requirements = [line.strip() for line in f.readlines()]
    return requirements


setup(
    name = 'Resume-screening_Model',
    version = '0.0.1',
    author = 'Wonderboi693',
    author_email = 'nlmhuy0101@gmail.com',
    packages = find_packages(),
    install_requires = [get_requirements('requirements.txt')]   
)