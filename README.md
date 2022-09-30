## CLAD: A Continual Learning benchmark for Autonomous Driving


## Classification
This is repo provides the code necessary to run the classification and detection benchmarks developed for the ICCV '21 
benchmark on Continual Learning: SSLAD 2021. At the moment only the classification track is up-to-date, but the 
detection track should be fixed and updated soon. See below for crude instructions on how to test the benchmark, 
these will be updated/elaborated soon-ish. Enjoy!

The bechmark works both with and without `avalanche`, so use whatever you want. The `examples` folder contains both
an example without (`classification.py`) and one with Avalanche (`classification_avalanche.py`). Feel free to test both.

The benchmark is based on the SODA10M dataset, see [here](https://soda-2d.github.io/index.html). We only use the 
labelled data, so only the labelled data should be downloaded. The original data doesn't include the necessary
time stamps, but these will be downloaded automatically the first time. 

### Steps to run the examples

1. Download the labeled trainval and test data [here](https://soda-2d.github.io/download.html).
The unlabeled data isn't used in this benchmark. Extract both to your `data_root` folder, keeping all
other file and directory names the same. 
2. Prepare the required libraries, but except the `PyTorch` libraries, no additional ones are required. 
Code is tested with Python `3.9` and the latest `pytorch` and `torchvision` libraries. 
You can also use the provided `env.yml` file and make a conda environment with: `conda env create -f env.yml`
3. Make sure that the `clad` library is added to your `PYTHONPATH`. On linux: 
`export PYTHONPATH=$PYTHONPATH:[path_to_clad]`
4. (Optional) Install `Avalanche` from their master branch (the default pip library lacks some required
features): `pip install git+https://github.com/ContinualAI/avalanche.git`. 
5. Test the `classification.py` or `classification_avalanche.py` examples.

Finally, you can check [this notebook](./examples/data_intro_c.ipynb) for further details on the benchmark,
its creation and how it is evaluated. 
