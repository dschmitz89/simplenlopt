from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == '__main__':
    
    setup(
        name='simplenlopt',   
        version='1.0.1', 
        description='A scipy.optimize like API for nlopt',
        url='https://github.com/dschmitz89/simplenlopt/',
        download_url='https://github.com/dschmitz89/simplenlopt/archive/refs/tags/1.0.tar.gz',
        author='Daniel Schmitz',
        license='Apache 2',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=['simplenlopt'],
        install_requires=[
            'numpy',
            'scipy>1.2',
            'six',
            'nlopt>2.6'
        ]            
    )