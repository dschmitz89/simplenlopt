from setuptools import setup

if __name__ == '__main__':
    
    setup(
        name='simplenlopt',   
        version='1.0', 
        description='A scipy.optimize like API for nlopt',
        url='https://github.com/dschmitz89/simplenlopt/',
        download_url='https://github.com/dschmitz89/simplenlopt/archive/refs/tags/1.0.tar.gz',
        author='Daniel Schmitz',
        license='MIT',
        packages=['simplenlopt'],
        install_requires=[
            'numpy',
            'scipy>1.2',
            'six',
            'nlopt>2.6'
        ]            
    )