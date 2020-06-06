import os
from setuptools import setup, find_packages
from distutils.core import setup as setup_cython
from Cython.Build import cythonize
from distutils.extension import Extension


def make_extension(package_path, name, include_dirs=None):
    if include_dirs is None:
        include_dirs = []
    cy_dir = os.path.join(*package_path)
    package_prefix = '.'.join(package_path)+'.'
    ext = Extension(package_prefix+name,
                    [os.path.join(cy_dir, name+'.pyx')],
                    include_dirs=include_dirs)
    return ext


def discover_extensions(root_dir):
    for (dirname, subdir, filenames) in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pyx'):
                pkg = dirname.split(os.sep)
                name = os.path.splitext(filename)[0]
                yield make_extension(pkg, name)
                #print(pkg, name)


extensions = list(discover_extensions('rlil'))

setup_cython(
    ext_modules=cythonize(extensions)
)


setup(
    name="pytorch-rl-il",
    version="0.0.1",
    description=(
        "A library for building reinforcement learning and imitation learning agents in Pytorch"),
    packages=find_packages(),
    url="https://github.com/syuntoku14/pytorch-rl-il",
    author="Toshinori Kitamura",
    author_email="syuntoku14@gmail.com",
    install_requires=[
        "gym[atari,box2d]",    # atari environments
        "numpy",         # math library
        "matplotlib",    # plotting library
        "seaborn",    # plotting library
        "pandas",
        "opencv-python",  # used by atari wrappers
        "pybullet",      # continuous environments
        "autopep8",      # code quality tool
        "torch-testing",  # testing library for pytorch
        "ray",  # multiprocessing tool
        "pytest",  # python testing library
        "cpprb",  # fast replay buffer library
        "pytest-benchmark",
        "gitpython"
        # these should be installed globally:
        # "tensorflow",  # needed for tensorboard
        # "torch",       # deep learning library
        # "torchvision", # install alongside pytorch
    ],
    extras_require={
        "pytorch": [
            "torch",
            "torchvision",
            "tensorboard"
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "sphinx-automodapi"
        ]
    },
)
