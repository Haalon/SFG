from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Separated grammar research package'


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


# Setting up
setup(
        name="SFG", 
        version=VERSION,
        author="Igor Zaytsev",
        author_email="haalon@yandex.com",
        description=DESCRIPTION,
        long_description=open('README.md', encoding='utf-8').read(),
        packages=find_packages(),
        install_requires= _parse_requirements("requirements.txt"),
        
        keywords=['python', 'first package'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: OS Independent",
        ]
) 
