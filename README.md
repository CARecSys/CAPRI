# RecSys Framework

This repository contains a framework for Recommender Systems (RecSys), allowing users to choose a dataset on a model based on their demand.

## ☑️ Prerequisites

You will need below libraries to be installed before running the application:

- Python >= 3.4
- NumPy >= 1.19
- SciPy >= 1.6
- PyInquirer >= 1.0.3

For a simple solution, you can simply run the below command in the root directory:

```
pip install -r prerequisites.txt
```

## 🚀 Launch the Application

Start the project by running the `main.py` in the root directory.

## 🧩 Contribute Guide

If you want to add a new model, define its name in CapitalCase and its covered scopes in config.py, models
The same can goes foe datasets
If you want to add evaluation metrics, consider adding them in Evaluations folder
Define your models (with the exact same name in config.py) in Models directory
Your model shoul have a main.py to be able to be read from the main.py in Root
Then, Call your model as 'ModelNameMain' as a class in main.py

## ⚠️ TODOs

- Add a proper **caching policy** to check the *Generated* directory
- Enable reading configuration settings from the **config**  file in all components
- Add the impact of **fusions** when running models
- Some parameters e.g. topK should be selected by user
