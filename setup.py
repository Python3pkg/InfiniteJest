from distutils.core import setup
# To upload to PyPi run python setup.py sdist upload -r pypi
setup(
    name='InfiniteJest',
    version='0.0.1',
    author='Matthew Robinson',
    author_email='mthw.wm.robinson@gmail.com',
    packages=['InfiniteJest','InfiniteJest.test'],
    scripts=[],
    url='https://github.com/MthwRobinson/InfiniteJest',
    license='LICENSE.txt',
    description='package for system design',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy',
        'pulp'
     ]
    )
