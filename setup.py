import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Azrules", # Replace with your own username
    version="0.0.1",
    author="Alex Zades",
    author_email="az@st4r.io",
    description="Data visualization and analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexZades/modernlab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)