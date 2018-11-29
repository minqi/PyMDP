import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=['numpy', 'torch', 'nose']

setuptools.setup(
    name="pymdp",
    version="0.0.1",
    author="Minqi Jiang",
    author_email="mnqjng@gmail.com",
    description="Markov decision processes in Python",
    long_description="An easy-to-use library for constructing and solving Markov decision processes.",
    long_description_content_type="text/markdown",
    url="https://github.com/minqi/PyMDP",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)