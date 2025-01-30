import setuptools

requires = [
    "numpy>=1.20",
    "scipy>=1.6",
    "matplotlib",
    "scikit-learn",
    "pandas",
    "joblib",
    "hmmlearn",
    "ipywidgets",
    "seaborn",
    "h5py",
    "hdf5storage",
    "numba",
    "python-benedict",
    "portion"
]
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuropy-pho",
    version="0.0.6",
    author="Bapun Giri, Pho Hale",
    author_email="bapung@umich.edu",
    maintainer=["Bapun Giri", "Nat Kinsky", "Pho Hale"],
    maintainer_email=["bapung@umich.edu", "nkinsky@umich.edu", "halechr@umich.edu"],
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
    python_requires=">=3.9",
    install_requires=requires,
)