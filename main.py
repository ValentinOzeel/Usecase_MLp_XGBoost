import os
from pathlib import Path
from src.supp_functions import import_module_from_path

# Import project's configuration
from conf.conf_functions import get_project_conf_data
# Import MLp's config function
from MLp.conf.config_functions import set_config, get_config
# Set MLp's config as project's config
set_config('MLP_CONFIG', get_project_conf_data(wanted_data='MLP_CONFIG'))  
set_config('MLFLOW_CONFIG', get_project_conf_data(wanted_data='MLFLOW_CONFIG'))


from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas
pd = import_cpu_gpu_pandas()

# Import MLpBuilder and PlaygroundPreprocess classes 
from MLp.src.MLp_builder import MLpBuilder
from MLp.src.MLp_preprocessing import MLpPlaygroundPreprocessing


'''
Initial Spaceship dataset's columns: 
    'PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
    'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported'

Dataframe's head without 'Transported' target.
  PassengerId HomePlanet CryoSleep  Cabin    Destination   Age    VIP  RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name
0     0001_01     Europa     False  B/0/P    TRAPPIST-1e  39.0  False          0.0        0.0           0.0     0.0     0.0    Maham Ofracculy
1     0003_01     Europa     False  A/0/S    TRAPPIST-1e  58.0   True         43.0     3576.0           0.0  6715.0    49.0      Altark Susent
2     0003_02     Europa     False  A/0/S    TRAPPIST-1e  33.0  False          0.0     1283.0         371.0  3329.0   193.0       Solam Susent
3     0004_01      Earth     False  F/1/S    TRAPPIST-1e  16.0  False        303.0       70.0         151.0   565.0     2.0  Willy Santantines
4     0005_01      Earth     False  F/0/P  PSO J318.5-22  44.0  False          0.0      483.0           0.0   291.0     0.0  Sandie Hinetthews
'''


# Identify current path
current_path = os.getcwd()
# Get data info from config
data_access_config = get_project_conf_data(wanted_data='DATA_ACCESS')
# Identify the train and valid paths as well as target's name
train_path = os.path.join(current_path, Path(data_access_config['train_path']))
to_predict_path = os.path.join(current_path, Path(data_access_config['to_predict_path']))
target = data_access_config['target']

# Get the data preprocessing module name and step names from the YAML config
preprocess_config = get_project_conf_data(wanted_data='DATA_PREPROCESSING')
preprocessing_obj_name, sampling_obj_name = preprocess_config.get('preprocessing', None), preprocess_config.get('sampling', None)
preprocessing_module_path = os.path.join(current_path, Path(preprocess_config.get('path', None)))
# Import the data preprocessing and sampling steps
data_preprocess_module = import_module_from_path(preprocessing_module_path)
preprocessing_steps, sampling_steps = getattr(data_preprocess_module, preprocessing_obj_name, None), getattr(data_preprocess_module, sampling_obj_name, None)


# Access pre-initialized model
model_config = get_project_conf_data(wanted_data='MODEL')
model_name = model_config.get('model', None)
model_module_path = os.path.join(current_path, Path(model_config.get('path', None)))
# Import pre-initialized model
model = getattr(import_module_from_path(model_module_path), model_name, None)

# Import finetunning grids (grid_search/optuna)
from src.finetunning_steps import model_grid, optuna_hyperparameters





# Initiate a MLp Pipeline
pipeline = MLpBuilder()

# Initiallize training data in MLp with pre-split that we will save for final evaluation of model's performance 
X, X_test, y, y_test = pipeline.initialize_data(
                         data=pd.read_csv(train_path),
                         target=target,
                         X=None,
                         y=None,
                         print_data_info=True,
                         # pre_split is performed after printing data info (if print_data_info=True)
                         pre_split=0.1)



'''Can use the playground to try out various strategies'''
X, y = pipeline.X.copy(), pipeline.y.copy()
# Use playground to try various 
playground = MLpPlaygroundPreprocessing(X_copy=X, y_copy=y)
#X, y = playground.playground_transform('preprocessing', steps=preprocessing_steps)
X, y = playground.playground_transform('sampling', steps=sampling_steps)
## Print useful infos regarding our transformed dataset
#pipeline.print_data_info(playground.X_pg, playground.y_pg)
X, y = playground.delete_previous_transformations(n_steps=1)
##print(self.list_pipeline_pg)
'''End playground'''






## Create the preprocessing pipelines
preprocessing = pipeline.create_data_pipeline(
                name = 'preprocessing',
                #pipeline=playground.full_pipeline_pg,
                steps=preprocessing_steps
            )
sampling = pipeline.create_data_pipeline(
                name = 'sampling',
                #pipeline=playground.full_pipeline_pg,
                steps=sampling_steps
            )

## Define the XGB model
xgb_model = pipeline.define_model(
    use_model=model,
    model_name='model',
    calibration=None, 
    #calib_cv=5,
)

##### Run training pipeline
pipeline.run_pipeline('accuracy', test_size=0.15, n_splits=5, kf=None, n_jobs_cv=-1,
                      use_mlflow=False, run_name='xgb_training_1', use_mlflow_cv=None, run_name_cv=None) 

pipeline.feature_importance(method='shap', scoring='accuracy')

#pipeline.grid_search_tuning(model_grid, 'accuracy',
#                            data_pipelines=old_pipelines,
#                            use_mlflow=False, run_name='first_run', 
#                            kf=None, n_splits=5, cv_n_jobs=-1)

    


#pipeline.optuna_tuning(optuna_hyperparameters, 5, 'accuracy', 
#                        use_mlflow=False, run_name='first_run', 
#                        direction='maximize', custom_objective=None, 
#                        test_size=None, n_splits=5, kf=None, cv_n_jobs=-1)


#pipeline.best_hyperparams_fit('accuracy', 0.7)
# 0.7 correspond to the training size including presplit done at initialization step


#### Train the model on all data available and make predictions using 'pre_split' and 'to_predict' unseen data
pipeline.train_on_all_data(get_pipelines_model=True)
model_predictions = pipeline.make_test_inferences('pre_split', 'accuracy')

