""" growcut package configuration """

import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext

setup(
    name='growcut',
    version='0.1',
    description='Xuxuzinho.',
    author='Edson Cavalcanti',
    packages=['gc3D'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("gc3D.growcut3D_cy", ["gc3D/growcut3D_cy.pyx"])
        ],
    include_dirs=[numpy.get_include(), ],
    install_requires=[
        'matplotlib',
	'scikit-image',
	'skimage',
	'cython',
        'numpy'])
