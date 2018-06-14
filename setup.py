from setuptools import setup, find_packages

# Can be switched to "tensorflow"
tensorflow_version = "tensorflow"

setup(
    name='pandora',
    version="0.0.1a",
    description='A Tagger-Lemmatizer for Latin and the vernaculars',
    url='http://github.com/hipster-philology/HookTest',
    author='Mike Kestemont',
    license='MIT',
    packages=find_packages(exclude=("tests")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Latin",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=[
        "gensim==2.0.0",
        "h5py==2.7.0",
        "keras==2.0.4",
        "nltk==3.2.2",
        "numpy==1.13.3",
        "scikit-learn==0.18.1",
        "seaborn==0.7.1",
        "{}==1.1.0".format(tensorflow_version),
        "editdistance==0.3.1",
        "jsonpickle==0.9.5",
        "torchvision",
        "configparser"
    ],
    tests_require=[
        "mock==2.0.0"
    ],
    entry_points={
        'console_scripts': [
            'pandora-train=pandora.cli:cli_train',
            'pandora-tagger=pandora.cli:cli_tagger'
        ]
    },
    test_suite="tests",
    zip_safe=False
)