from setuptools import setup

setup(
    name='nrt-predict',
    version='0.1',
    description='Near Real Time Library',
    url='https://github.com/daleroberts/nrt-predict',
    author='Dale Roberts',
    author_email='dale.roberts@anu.edu.au',
    license='To Be Determined',
    packages=['models'],
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
    scripts=[
        'nrt_predict.py'
    ],
    package_data={
        "" : ["nrt_predict.yaml"]
    },
    include_package_data=True,
    zip_safe=False
)
