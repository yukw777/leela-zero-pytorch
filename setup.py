from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setup(
    name="leela-zero-pytorch",
    version="0.1.0",
    packages=find_packages(),  # This will install my_ext package
    install_requires=requirements(),
)
