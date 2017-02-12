from distutils.core import setup

description = open('README.rst').readlines()

setup(name='IneqPy',
      version='0.0.1',
      description='A Python Package To Quantitative Analysis Of Inequality',
      long_description=open('README.rst').read(),
      author='Maximiliano Greco',
      author_email='mmngreco@gmail.com',
      url='https://github.com/mmngreco/IneqPy',
      download_url='https://github.com/mmngreco/IneqPy/tarball/0.0.1',
      packages=['ineqpy'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                    ],
      install_requires=['numpy', 'pandas'],
     )
