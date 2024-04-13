from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']

from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas
pd = import_cpu_gpu_pandas()

import numpy as np


from MLp.src.preprocessing.MLp_preprocessing_transformers import SplitExpandTransformer, DropColsTransformer, OperationTransformer, BinningTransformer, \
                                      PCATransformer, StandardScalerTransformer, KMeansClusteringTransformer, OHEncodeTransformer, LabelEncodeTransformer


def spent_or_not(X):
    return pd.DataFrame((X.sum(axis=1) > 0).astype(int), columns=['spent_or_not'], index=X.index).astype('float64')
     

def spent_or_not_on_luxuries(X):
    return pd.DataFrame((X.sum(axis=1) > 0).astype(int), columns=['spent_or_not_on_luxuries'], index=X.index).astype('float64')
  
def group_size(X):
    groupby_series = X.groupby('PassengerId_1')['PassengerId_2'].transform('max')
    return pd.DataFrame({'group_size': groupby_series.astype(int)}, index=X.index)
  
def alone_or_not(X):
    return pd.DataFrame({'alone_or_not': (X['group_size'] > 1).astype(int)}, index=X.index)

def family_or_not(X):
    series = X.duplicated(subset=['PassengerId_1', 'Last_name'], keep=False)
    return pd.DataFrame({'family_or_not': series.astype(int)}, index=X.index)
  
def family_size(X):
    groupby_series = X.groupby('Last_name')['Last_name'].transform('count')
    return pd.DataFrame({'family_size': groupby_series}, index=X.index)

def fill_some_na_HomePlanet(X):
  '''Passengers on decks A, B, C or T came from Europa.
  Passengers on deck G came from Earth.
  Passengers on decks D, E or F came from multiple planets.'''
  copy = X.copy()
  # Decks A, B, C or T came from Europa
  copy.loc[(X['HomePlanet'].isna()) & (copy['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet']='Europa'
  # Deck G came from Earth
  copy.loc[(X['HomePlanet'].isna()) & (copy['Deck']=='G'), 'HomePlanet']='Earth'
  return copy.drop('Deck', axis=1)


def fill_na(X):
    X_copy = X.copy()
    for col_name in X_copy.columns:
      value_counts = X_copy[col_name].value_counts()
      # Create probability distribution for each home planet
      values = value_counts.index 
      count = value_counts.values
      proba = count/sum(count)
      
      np.random.seed(RANDOM_STATE)
      # Loc HomePlanet NaN values and assign them a value according to the values's distribution
      X.loc[X[col_name].isna(), col_name] = np.random.choice(values, X_copy[col_name].isna().sum(), p=proba)
    return X


SimpleImputer = import_cpu_gpu_sklearn('impute', 'SimpleImputer')



#####################################################################
#                      PREPROCESSING/ENGINEERING STEPS 
#####################################################################
           
# Define the preprocessing/feature engineering steps. Order will be followed during processing.
# Names that begin with '*' will not be wrapped in DynamicFeatureTransformer/FunctionTransformerTransformer preprocessing classes
preprocessing_steps = [
            
            ('*split_expand', SplitExpandTransformer, {'columns'  : ['PassengerId', 'Cabin', 'Name'], 'feature_splits':{'PassengerId':'_',
                                                                                                                       'Cabin':['/', ['Deck', 'Num', 'Side']],
                                                                                                                       'Name':[' ', ['First_name', 'Last_name']]}}),
            
            
            
            ('group_size', group_size, {'columns'  : ['PassengerId_1', 'PassengerId_2']}),  
            
            ('alone_or_not', alone_or_not, {'columns'  : ['group_size']}),
            
            ('family_or_not', family_or_not, {'columns'  : ['PassengerId_1', 'Last_name']}),                    
            
            ('fill_some_na_HomePlanet', fill_some_na_HomePlanet, {'columns' : ['HomePlanet', 'Deck'], 'drop':['HomePlanet']}),
#
            ('fill_na', fill_na, {'columns'  : ['HomePlanet', 'CryoSleep', 
                                               'Deck', 'Side', 'Destination', 
                                               'Age', 'VIP'],                     'drop':True}),

            ('*drop_cols', DropColsTransformer, {'columns'  : ['First_name', 'Last_name', 'PassengerId_2', 'PassengerId_1']}),
 
            ('num_imput', SimpleImputer, {'strategy':'median'}, {'_columns':'numericals', '_drop':True, '_col_name_prefix':''}),

            ('*global_spending', OperationTransformer, {'columns':['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
                                                       'multiple_cols_method':'sum', # ['sum', 'substration', 'range', 'product', 'ratio',  'mean', 'median', 'variance', 'std_dev']
                                                       'aggregation': True,
                                                       'feature_name':'global_spending',
                                                       'drop':False,
                                                       }),
            
            ('*luxuries_spending', OperationTransformer, {'columns':['RoomService', 'FoodCourt', 'Spa', 'VRDeck'],
                                                       'multiple_cols_method':'sum', # ['sum', 'substration', 'range', 'product', 'ratio',  'mean', 'median', 'variance', 'std_dev']
                                                       'aggregation': True,
                                                       'feature_name':'luxuries_spending',
                                                       'drop':False,
                                                       }),

            ('spent_or_not', spent_or_not, {'columns'  : ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']}),
            
            ('spent_or_not_on_luxuries', spent_or_not_on_luxuries, {'columns'  : ['RoomService', 'ShoppingMall', 'Spa', 'VRDeck']}),
          
            ('*binning', BinningTransformer, {'columns':['Age'],
                                             'custom_binning':    
                                                     {'labels':[1, 2 ,3, 4, 5],
                                                      'bins':[0 , 12, 18, 50, 65, 200]},
                                             'bins_as_numerical':True,
                                             'drop':True}),

            ('*binning', BinningTransformer, {'columns':['Num'],
                                             'equal_frequency_binning':False,
                                             'equal_width_binning':True,
                                             'n_groups':7,
                                             'bins_as_numerical':True,
                                             'drop':True}),

         
           ('*pca', PCATransformer, {'columns' : ['group_size', 'alone_or_not', 'family_or_not', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
                                                    'sum_global_spending', 'sum_luxuries_spending'],
                                    #'component_names' : ['pca_component_1', 'pca_component_2'],
                                    'preprocess_transformer':StandardScalerTransformer,
                                    'preprocess_params':{},
                                    'drop' : False, 'n_components' : 2}),

           
            ('*kmeans', KMeansClusteringTransformer, {'columns'    : 'numericals',
                                                        'preprocess_transformer' : StandardScalerTransformer,
                                                        'preprocess_params':{}, 
                                                        'drop' : False, 'n_clusters' : 30}),

            ('*oh_encode', OHEncodeTransformer, {'columns':'categoricals',
                                                        'max_cardinality':15,
                                                        'handle_unknown':"ignore",
                                                        'sparse_output':False}),

                           
#            ('*label_encode_transformer', LabelEncodeTransformer, {'columns':'categoricals', 'min_cardinality':15}),
 
#            ('*dropcols', DropColsTransformer, {'columns':'categoricals'})
]




#####################################################################
#                      SAMPLING STEPS 
#####################################################################


from MLp.src.preprocessing.MLp_preprocessing_transformers import LogScalerTransformer
from MLp.src.preprocessing.MLp_sampling_transformers import IQROutliersTransformer, DropNaTransformer


sampling_steps = [
    (
        '*log_scaler',
        LogScalerTransformer,
        {'columns': ['VRDeck', 'Spa', 'FoodCourt', 'ShoppingMall', 'RoomService']}
    ),
    (
        '*iqr_outliers',
        IQROutliersTransformer,
        {'columns': ['VRDeck', 'Spa', 'FoodCourt', 'ShoppingMall', 'RoomService'],
         'IQR_multiplier': 1.5},
        
    ),
    #('*drop_na', DropNaTransformer, {'columns':'all'})

]

