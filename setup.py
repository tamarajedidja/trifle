from setuptools import setup, find_packages

setup(
    name='trifle',
    version='0.1',
    packages=find_packages(include=["trifle_module", "trifle_module.*"]),
    include_package_data=True,
    package_data={
        'trifle_module': [
            'templates/Smith20.nii.gz',
            'templates/Smith70.nii.gz',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'nibabel',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'nipype'
    ],  
    entry_points={
        'console_scripts': [
            'trifle_run=trifle_module.trifle_main:main',
            'trifle_spatial=trifle_module.trifle_spatial:main',
            'trifle_temporal=trifle_module.trifle_temporal:main',
            'trifle_tvmixing=trifle_module.trifle_timevaryingmixing:main'
        ],
    },
    author='TJ de Kloe',
    author_email='tamara.dekloe@donders.ru.nl',
    description='Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tamarajedidja/trifle',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)