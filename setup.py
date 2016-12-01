from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

import sys


def readme():
    with open('README.rst') as f:
        return f.read()


def version():
    with open('numbamisc/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split(r"'")[1]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(name='numbamisc',
      version=version(),

      description='Miscellaneous utilities based on numba and jinja.',
      long_description=readme(),
      keywords='convolution filter median numba',
      platforms=["all"],  # maybe sometime also "Mac OS-X", "Unix"

      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],

      license='Apache License Version 2.0',

      url='https://github.com/MSeifert04/numbamisc',

      author='Michael Seifert',
      author_email='michaelseifert04@yahoo.de',

      packages=[
          'numbamisc',
          'numbamisc.utils',
          ],

      install_requires=[
          'numba',
          'numpy',
          'scipy',
          'jinja2',
          ],

      setup_requires=[
          ] + pytest_runner,

      tests_require=[
          'pytest',
          ],

      include_package_data=True,
      zip_safe=False,
      )
