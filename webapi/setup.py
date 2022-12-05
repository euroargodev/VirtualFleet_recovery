from setuptools import setup

setup(
    name='myapp',
    packages=['myapp'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)
