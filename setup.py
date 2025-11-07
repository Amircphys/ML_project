from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description="Example of ml project",
    author="Abdullaev Gadzhimurad",
    entry_points={
        "console_scripts": [
            "ml_project_train = src.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn",
        "dataclasses==0.6",
        "pyyaml==5.3",
        "clearml==1.17.1",
        "marshmallow-dataclass==8.3.0",
        "pandas",
        "s3cmd"
    ],
    license="MIT",
)