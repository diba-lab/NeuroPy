import setuptools

requires = [
    "numpy>=1.19",
    "scipy>=1.5",
    "matplotlib",
    "scikit-learn",
    "pandas",
    "joblib",
    "hmmlearn",
    "ipywidgets",
    "seaborn",
]
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuropy-bapungiri",
    version="0.0.1",
    author="Bapun Giri",
    author_email="bapung@umich.edu",
    maintainer=["Bapun Giri", "Nat Kinsky"],
    maintainer_email=["bapung@umich.edu", "nkinsky@umich.edu"],
    description="Package for ephys analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requires,
)