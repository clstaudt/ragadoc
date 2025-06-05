from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='ragadoc',
    version='0.1.0',
    author='Christian Staudt',
    author_email='',  # Leave empty or add contact email
    description='An AI document assistant that answers questions about your PDFs with citations and highlights them directly in the document.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/clstaudt/ragadoc', 
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Markup',
        'Topic :: Office/Business :: Office Suites',
    ],
    keywords='ai, pdf, document-analysis, rag, question-answering, streamlit, ollama, local-ai',
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ragadoc=app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'ragadoc': ['themes/*.json'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/ragadoc/issues',
        'Documentation': 'https://github.com/yourusername/ragadoc#readme',
        'Source': 'https://github.com/yourusername/ragadoc',
    },
) 