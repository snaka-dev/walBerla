#!/usr/bin/env python

# install package localy with 'pip install -e .'

from setuptools import setup
setup(name='walberla_codegen',
      version='1.0',
      description='Python packages for WaLBerla kernel generation',
      keywords='walberla pystencils lbmpy',
      author='LSS',
      author_email='cs10-contact@fau.de',
      url='https://walberla.net/',
      packages=['pystencils_walberla', 'lbmpy_walberla'],
      install_requires=['pystencils', 'lbmpy', 'sympy>=1.1', 'numpy', 'jinja2'],
      package_data={
          # Include all template files
          "": ['.tmpl.*'],
          # Or specifics files:
          "hello": ["*.msg"],
      },
      extras_require={
          'cpuinfo': ['py-cpuinfo'],
      },
      python_requires='>=3.6',
      )
