import sys
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    setuptools.extension.Extension(
        'xnicer.kde',
        ['xnicer/kde' + ext],
        language='c++'
        # add include_dirs, libraries, and library_dirs (all string list)
        # if you need specific compilation flags
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setuptools.setup(
    name='xnicer',
    version='0.1.0',
    author='Marco Lombardi',
    author_email='marco.lombardi@gmail.com',
    description='The XNICER/XNICEST algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/astropy/xnicer',
    packages=setuptools.find_packages(),
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
