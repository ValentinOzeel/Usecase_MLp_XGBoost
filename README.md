Demonstration of MLp use case with Kaggle's Spaceship dataset using XGBoost for binary classification (spaceship tabular dataset).

Final performance on the 10% pre-splitted data: around 81%.

Easily try various preprocessing strategies with the Playground. 
Easily apply specific transformer/function transformer to specific columns. (all, list of cols, all categoricals, all numericals or all except these)
Sample data directly within pipelines (eg outlier removal; as dropped rows will be dropped in y too).
Easily create preprocessing and training/finetuning pipelines. 
Experiment tracking

conf folder contains a yaml configuration file and a short file with a function to import the config in main. 
Preprocessing steps used are declared in src/data_preprocess_steps.py
Model is declared in src/model.py
Hyperparams finetunning grids are declared in src/finetunning_steps.py