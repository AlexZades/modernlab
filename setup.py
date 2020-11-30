
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='modernlab',  # Required
    version='0.1.1',  # Required
    description='Data visualization and analysis toolst',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/AlexZades/modernlab',  # Optional
    author='Alex Zades',  # Optional
    author_email='az@st4r.io',  # Optional
    packages=find_packages(),  # Required
    python_requires='>=3.5',
    install_requires=['numpy','scipy'],  # Optional
    project_urls={  # Optional
        'Wiki': 'https://github.com/AlexZades/modernlab/wiki'
        },
)