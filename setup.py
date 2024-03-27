from setuptools import setup, find_packages

setup(
    name='vfrecovery',
    version='2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'vfrecovery = vfrecovery.command_line_interface.virtualfleet_recovery:base_command_line_interface',
        ],
    },
)