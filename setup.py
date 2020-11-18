import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modernlab", 
    version="0.0.1",
    author="Alex Zades",
    author_email="az@st4r.io",
    description="Data visualization and analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexZades/modernlab",
    packages=['modernlab'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'PyYAML',
        'scipy.optimize>=1.5.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0'
    ]
)