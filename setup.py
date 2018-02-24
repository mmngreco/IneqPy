import versioneer
from setuptools import setup
with open('README.md') as f:
    description = f.read()

setup(name='IneqPy',
      version=versioneer.get_version(),
      description='A Python Package To Quantitative Analysis Of Inequality',
      long_description=description,
      author='Maximiliano Greco',
      author_email='mmngreco@gmail.com',
      url='https://github.com/mmngreco/IneqPy',
      # download_url='https://github.com/mmngreco/IneqPy/tarball/',
      packages=['ineqpy', 'ineqpy.grouped'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5'],
      install_requires=['numpy', 'pandas', 'numba'],
      cmdclass=versioneer.get_cmdclass(),
     )
