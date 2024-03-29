from setuptools import setup, find_packages

VERSION = '1.1.2'
DESCRIPTION = 'Get optimal control from URDF'

# Setting up
setup(
    name="urdf2optcontrol",
    version=VERSION,
    author="Andrea Boscolo Camiletto, Marco Biasizzo",
    author_email="<abcamiletto@gmail.com>, <marco.biasizzo@outlook.it>",
    url="https://github.com/abcamiletto/urdf2optcontrol",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['casadi', 'numpy>=1.16.6', 'matplotlib', 'jinja2'],
    keywords=['python', 'optimal_control', 'robotics', 'robots'],
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics"
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    package_data={'': ['template.html']},
    include_package_data=True
)
