from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(name='Forecast_2000',
      version = '0.1',
      description="Forecast_2000",
      license="MIT",
      install_requires=requirements,
      packages=find_packages(),
      )
