# RecSys Framework

## How to Start?

Run the command below:
pip install -r prerequisites.txt

If you want to add a new model, define its name in CapitalCase and its covered scopes in config.py, models
The same can goes foe datasets
If you want to add evaluation metrics, consider adding them in Evaluations folder
Define your models (with the exact same name in config.py) in Models directory
Your model shoul have a main.py to be able to be read from the main.py in Root
Then, Call your model as 'ModelNameMain' as a class in main.py

# TODO List

- Add a proper caching policy to check the Generated directory
- Read from config everywhere
- topK should be selected by user
