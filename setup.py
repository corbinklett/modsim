from setuptools import setup

setup(
    name='modsim',
    version='0.1',
    packages=['modsim'],
    # Add more details like author, license, etc.
)

# python setup.py sdist bdist_wheel
# pip install . -- or -- pip install -e .



# import setuptools

# setuptools.setup(
#     name="modsim",
#     version="0.1",
#     author="Your Name",
#     author_email="your.email@example.com",
#     description="A short description of your package",
#     long_description=open('README.md').read(),
#     long_description_content_type="text/markdown",
#     url="https://github.com/yourusername/modsim",
#     packages=setuptools.find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     python_requires='>=3.6',
#     install_requires=[
#         # Add your package dependencies here
#         # 'numpy>=1.18.0',
#         # 'pandas>=1.0.0',
#         # etc.
#     ],
# )
