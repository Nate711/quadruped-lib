import setuptools

setuptools.setup(
    name="Quadruped Library",
    version="0.1",
    author="Nathan Kau",
    author_email="nathankau@gmail.com",
    description="Quadruped dynamics, kinematics, ik",
    packages=["quadruped_lib"],
    install_requires=[
        "numpy",
        "pyyaml",
        "transforms3d",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
