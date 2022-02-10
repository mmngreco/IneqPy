import versioneer
from setuptools import setup, find_packages


with open('README.md') as f:
    description = f.read()


setup(
    name="IneqPy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A Python Package To Quantitative Analysis Of Inequality",
    long_description=description,
    long_description_content_type='text/markdown',
    author="Maximiliano Greco",
    author_email="mmngreco@gmail.com",
    url="https://github.com/mmngreco/IneqPy",
    package_dir={'': 'src'},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["numpy", "pandas", "numba"],
)
