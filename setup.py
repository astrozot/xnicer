import sys
from distutils.core import setup
from distutils.extension import Extension
# from setuptools import setup
# from setuptools.extension import Extension
import numpy as np

with open('README.md', 'r') as fh:
    long_description = fh.read()

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        'xnicer.kde.kde',
        ['xnicer/kde/kde' + ext]
    ),
    Extension(
        'xnicer.xdeconv.em_step',
        ['xnicer/xdeconv/em_step' + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='xnicer',
    version='0.1.0',
    author='Marco Lombardi',
    author_email='marco.lombardi@gmail.com',
    description='The XNICER/XNICEST algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/astropy/xnicer',
    packages=[], # was setuptools.find_packages(),
    python_requires='>=3.6',
    setup_requires=['numpy','scipy','sklearn'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'License:: OSI Approved:: GNU Lesser General Public License v3(LGPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='xnicer',
    ext_modules=extensions
)
