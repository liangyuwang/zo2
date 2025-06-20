from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='zo2',
    version='0.1.1',
    author='liangyuwang',
    author_email='liangyu.wang@kaust.edu.sa',
    description='ZO2 (Zeroth-Order Offloading), a framework for full parameter fine-tuning 175B LLMs with 18GB GPU memory',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,  # List of dependencies, read from requirements.txt
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    include_package_data=True,
    zip_safe=False
)
