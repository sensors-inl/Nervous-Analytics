from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
about = {}

with open(os.path.join(here, 'nervous_analytics', '__version__.py'), mode='r', encoding='utf-8') as f:
    exec(f.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    author="Tristan Habémont, Bertrand Massot",
    author_email="bertrand.massot@insa-lyon.fr",
    description="A package to extract ECG and EDA features localization in real-time.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/INL-Biomedical-Sensor/Nervous-Analytics", #TODO
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        '': ['LICENSE'],
    },
)