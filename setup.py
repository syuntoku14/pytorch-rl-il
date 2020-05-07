from setuptools import setup, find_packages

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
