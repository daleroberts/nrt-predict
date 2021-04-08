from setuptools import setup, find_packages

setup(
    name='nrtpredict',
    version='0.1',
    description='Near Real Time Library',
    url='https://github.com/daleroberts/nrt-predict',
    author='Dale Roberts',
    author_email='dale.roberts@anu.edu.au',
    license='To Be Determined',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nrtpredict = nrtpredict.cli:cli_entry',
        ],
    },
    install_requires=[
          'requests',
          'pyyaml',
          'psutil',
          'pytest',
          'boto3',
          'numpy',
          'scikit-image',
          'gdal',
          'joblib'
      ],
    package_data={
        'nrtpredict': [ 'nrtpredict/config/nrt-default-config.yaml' ]
    },
    include_package_data=True,
    zip_safe=False
)
