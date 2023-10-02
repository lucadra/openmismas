from setuptools import setup, find_packages

setup(
    name='openmismas',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "opencv_python==4.8.0.76",
        "pandas==2.1.1",
        "torch==2.0.1",
        "tqdm==4.66.1",
        "ultralytics==8.0.188",
        "whisper_timestamped==1.12.20",
    ],
)
