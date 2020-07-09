from setuptools import setup, Extension
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('LICENSE', 'r') as f:
    license_ = f.read()

with open('README.md', 'r') as f:
    readme = f.read()



setup(
    name='nlp-tools',
    version='1.0.0',
    description='Collection of metrics use for Chinese generation quality evalution',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='yuyusica',
    url='https://github.com/yuyusica/NLP_tools',
    keywords='chinese text generations evaluation metrics',
    packages=['nlp_tools'],
    package_data={'nlp_tools': ['nlp_tools/*']},
    include_package_data=True,
    install_requires=[
        'nltk'
    ],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)