from setuptools import setup, find_packages

setup(
    name="NewYorkAirbnbOpenData",
    version="0.0.5",
    author="Pio",
    author_email="p990301@gmail.com",
    description="New York Airbnb Open Data package",
    url="https://github.com/pjs990301/New_York_City_Airbnb_Open_Data",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=['numpy>=1.21.6',
                      'scikit-learn>=1.0.2',
                      'seaborn>=0.11.2']
)
