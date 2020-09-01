import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import os
#from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as mt
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

os.chdir("./Excel_data")

rs_list = []
Mechanical_param = 'Mean_strain_3'

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=5, 
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=16, 
#        scoring=scoring_fit,
        verbose=True
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred


df=pd.read_excel("Sim_ML_results_20200515.xlsx",sheet_name='For_ML_d20')       			
features = df.columns[1:28]

for i in range(0,500):
    print(i)
    first_time = True
       		
       		# Create a dataframe with the four feature variables
       	# Create two new dataframes, one with the training rows, one with the test rows
       	#if len(test)>12:
       	# Show the number of observations for the test and training dataframes
       	 
       	# Create a list of the feature column's names2
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[Mechanical_param],
                                                        test_size=0.10, random_state=i, 
                                                        stratify=df['Class'])       	# train['species'] contains the actual species names. Before we can use it,
       	# we need to convert each species name into a digit. So, in this case there
       	# are three species, which have been coded as 0, 1, or 2.
    
    print(y_test)
    model = GradientBoostingRegressor()
    param_grid = {
        'max_features': [4,5,6,7,8],
        #'alpha':[0.7,0.9],
        #'min_samples_leaf':[1,3,4],
        'min_samples_split':[2,3],
        'n_estimators': [1,2,5,10,20],
        #'min_weight_fraction_leaf':[0.0,0.1,0.2],
        'max_depth': [4,5,6,7],
        'subsample': [0.4, 0.6, 0.8]}
        #'learning_rate': [0.05,0.1,0.02]
    
        
        
    
    model_trained, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                     param_grid, cv=5)

    best_params = model_trained.best_params_
    df_model = pd.DataFrame.from_dict(best_params, orient='index') 


    mpe = abs(np.mean(abs(y_test - pred)/y_test))
    y_pred =pred
    r2 = mt.r2_score(y_test,y_pred)
    mse = mt.mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('MPE',mpe)       
    if r2 > 0.90 and mpe < 0.1:
        rs_list.append([r2,mpe,i])
        model_metrics = {"R2":r2 , "MSE":mse  ,"MPE":mpe, "i": i}
        df_metrics = pd.DataFrame.from_dict(model_metrics,orient='index')                                            
        df_model_metrics = pd.concat([df_metrics, df_model], axis=0, sort=False)
        if first_time:
            df_models = df_model_metrics
            first_time = False
        df_models = pd.concat([df_models,df_model_metrics],axis = 1)
        print('SEED:', i)
        print('R2:', r2)
        print('MSE',mse)      
        print('Model:', df_model)
        model_name = 'MeanS_Model_'+str(i)+'.pkl'
        pickle.dump(model_trained, open(model_name, "wb"))
                
mpe = abs(np.mean(abs(y_test - y_pred)/y_test))
r2 = mt.r2_score(y_test,y_pred)
mse = mt.mean_squared_error(y_test, y_pred)
model_metrics = {"R2":r2 , "MSE":mse  ,"MPE":mpe, "i": i}
print(df_models)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
save_obj(df_models,'GBR_models_90p_d20_MeanS')
# Root Mean Squared Error



