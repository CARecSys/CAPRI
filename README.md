# CAPRI: Context-Aware interpretable Point-of-Interest Recommendation Framework

![CAPRI-Context-Aware interpretable Point-of-Interest Recommendation Framework](https://github.com/RecSys-lab/CAPRI/blob/main/_contents/cover.jpg "CAPRI-Context-Aware interpretable PoI Recommender")

This repository contains a framework for Recommender Systems (RecSys), allowing users to choose a dataset on a model based on their demand.

## ☑️ Prerequisites

You will need below libraries to be installed before running the application:

- Python >= 3.4
- NumPy >= 1.19
- SciPy >= 1.6
- PyInquirer >= 1.0.3

For a simple solution, you can simply run the below command in the root directory:

```python
pip install -r prerequisites.txt
```

## 🚀 Launch the Application

Start the project by running the `main.py` in the root directory. With this, the application settings are loaded from the `config.py` file. You can select from different options to choose a model (e.g. GeoSoCa, available on the **Models** folder) and a dataset (e.g. Yelp, available on the **Data** folder) to be processed by the selected model, along with a fusion operator (e.g. prodect or sum). The system starts processing data using the selected model and provides some evaluations on it as well. The final results will be added to the **Generated** folder, withe the name template representing which model has been emplyed on which dataset and with what item selection rate.

## 🧩 Contribution Guide

Contribution to the project can be done through various approaches:

### Adding a new dataset

All datasets can be found in **./Data/** directory. In order to add a new dataset, you should:

- Modify the **config.py** file and add a record to the datasets dictionary. The key of the item should be the dataset's name (CapitalCase) and the value is an array of strings containing the dataset scopes (all CapitalCase). For instance

```python
"DatasetName":  ["Scope1", "Scope2", "Scope3"]
```

- Add a folder to the **./Data/** directory with the exact same name selected in the previous step. This way, your configs are attached to the dataset. In the created folder, add files of the dataset (preferably camelCase, e.g. socialRelations). Note that for each of these files, a variable with the exact same name will be automatically generated and fed to the models section. You can find a sample for the dataset sturcture here:

```bash
+ Data/
	+ Dataset1
		+ datasetFile1
		+ datasetFile2
		+ datasetFile3
	+ Dataset2
		+ datasetFile4
		+ datasetFile5
		+ datasetFile6
```

### Adding a new model

Models can be found in **./Models/** directory. In order to add a new model, you should:

- Modify the **config.py** file and add a record to the models dictionary. The key of the item should be the model's name (CapitalCase) and the value is an array of strings containing the scopes that mode covers (all CapitalCase). For instance

```python
"ModelName":  ["Scope1", "Scope2", "Scope3"]
```

- Add a folder to the **./Models/** directory with the exact same name selected in the previous step. This way, your configs are attached to the model. In the created folder, add files of the model (preferably camelCase, e.g. socialRelations). Models contain a **main.py** file that holds the contents of the model. The file **main.py** contains a class with the exact name of the model and the letter 'Main' (e.g. ModelNameMain). This class should contain a main function with two argument: (i) datasetFiles dictionary, (ii) the name of the selected dataset. For a better description, check the code sample below:

```python
import numpy as np
...

class NewModelMain:
	def main(datasetFiles, selectedDataset):
		print('Other codes goes here')
```

There is a **utils.py** file in the **./Models/** directory that keeps the utilities that can be used in all models. If you are thinking about a customized utilities with other functions, you can add an **extendedUtils.py** file in the model's directory. Also, a **/lib/** directory is considered in each model folders that contains the libraries used in the model. You can find a sample for the dataset sturcture here:

```bash
+ Models/
	+ Model1/
		+ lib/
		+ __init__.py
		+ main.py
		+ extendedUtils.py
	+ utils.py
	+ __init__.py
```

Note: do not forget to add a **_init_**.py file to the directories you make.

### Adding a new evaluation

You can simply add the evaluations to the `./Evaluations/metrics.py` file.

## ⚠️ TODOs

- Add a proper **caching policy** to check the _Generated_ directory
- Unifying **saveModel** and **loadModel** in utils.py
- Enable reading configuration settings from the **config** file in all components
- Add the impact of **fusions** when running models
- Some parameters e.g. topK should be selected by user
- Unify utils in Models and Modes/GeoSoCa
- We will need a pre-processing stage for unifying Utils methods
