import setuptools
from Cython.Build import cythonize

with open('README.md', 'r') as fh:
    long_description = fh.read()

extensions = [
    setuptools.extension.Extension(
        'xnicer.kde',
        ['xnicer/kde.pyx']
        # add include_dirs, libraries, and library_dirs (all string list)
        # if you need specific compilation flags
    ),
]
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'License:: OSI Approved:: GNU Lesser General Public License v3(LGPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='xnicer',
    ext_modules=cythonize(extensions)
)
