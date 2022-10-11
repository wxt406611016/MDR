# Make Data Reliable

Code for the paper "Make Data Reliable : An Explanation-powered Cleaning on Malware Dataset Against Backdoor Poisoning Attacks"

## Enviroment
---
OS System : Ubuntu 20.04

CPU : Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

Memory : 96G

python : 3.8.0

## Dependencies
---
This codebase has been developed and tested only with python 3.8.0.

This code depends on several packages. We recommend to use conda to build the dependencies for the code. However, the package louvain-python can not be found in conda repository. Therefore, for python-louvain package, use pip to install it. 
please follow the following instructions:
```
### build python 3.8.0
$ conda create -n mdr python=3.8.0

### add channel
$ conda config --append channels conda-forge

### activate the built enviroment
$ conda activate mdr

### install dependencies
# First, use conda to install some packages except python-louvain
$ conda install --yes --file requirements.txt

# Second, use pip to install python-louvain because it can not be found in conda repository
$ pip install python-louvain==0.16
```

## Build Poisoned Dataset
---
Before evaluating defense performance, we used two attack strategies mentioned in the paper of ["Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers"](https://github.com/ClonedOne/MalwareBackdoors) to poison the EMBER and Contagio datasets. 

For ease of access to the poisoned datasets, to avoid the lengthy operation of reproducing the attack process, the poisoned datasets feature numpy files are provided in the [LINK](https://github.com/wxt406611016/MDR/releases/tag/Poisoned_Datasets). Then Poisoned_Dataset.zip file can be easily downloaded.

The poisoned datasets feature numpy files can also be downloaded with wget command by running:
``wget https://github.com/wxt406611016/MDR/releases/download/Poisoned_Datasets/Poisoned_Dataset.zip``

After downloading the provided poisoned datasets feature numpy files (Poisoned_Dataset.zip) under the program folder (./MDR), unzip the file first with ``$ unzip Poisoned_Dataset.zip``. Then the program folder(./MDR) structure shows as below:
```
./MDR
  ├── backdoor_ember_17
  ├── backdoor_ember_8
  ├── backdoor_pdf_16
  ├── dataset
  ├── mw_backdoor
  ├── Defense_ember.py
  ├── Deployed_model_agnostic.py
  ├── Surrogate_model_agnostic.py
  ├── requirements.txt
  ├── readme.md
```

## Run Experiments
---
### Performacne Comparison On Poisoned Ember Dataset
* To reproduce the defense performance for no_attack scenario, shown in Table 1 of "No Attack" strategy, please run:

  ``python Defense_ember.py --target no_attack``

* To reproduce the defense performacne for combined attack scenario, shown in Table 1 of "Combined" strategy, please run:

  ``python Defense_ember.py --target combined``

* To reproduce the defense performacne for independent attack scenario, shown in Table 1 of "Independent" strategy, please run:

  ``python Defense_ember.py --target independent``

### Surrogate Model-agnostic evaluation
* To reproduce the surrogate model-agnostic evaluation for combined attack scenario, shown in Figure 5(a), please run:

  ``python Surrogate_model_agnostic.py --target combined``

* To reproduce the surrogate model-agnostic evaluation for independent attack scenario, shown in Figure 5(b), please run:

  ``python Surrogate_model_agnostic.py --target independent``

### Deployed Model-agnostic evaluation
* To reproduce the deployed model-agnostic evaluation for combined attack scenario, shown in Figure 6(a), please run:

  ``python Deployed_model_agnostic.py --target combined``

* To reproduce the deployed model-agnostic evaluation for independent attack scenario, shown in Figure 6(b), please run:

  ``python Deployed_model_agnostic.py --target independent``
