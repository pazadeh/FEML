import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import os
#from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as mt
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

os.chdir("./Excel_data")

rs_list = []


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


df=pd.read_excel("Sim_ML_results_20200515.xlsx",sheet_name='For_Classification')       			
features = df.columns[1:27]
df['Class'][:]=pd.factorize(df['Class'])[0]
for i in range(0,200):
    print(i)
    first_time = True
       		
       		# Create a dataframe with the four feature variables
       	# Create two new dataframes, one with the training rows, one with the test rows
       	#if len(test)>12:
       	# Show the number of observations for the test and training dataframes
       	 
       	# Create a list of the feature column's names2
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['Class'],
                                                        test_size=0.20, random_state=i, 
                                                        stratify=df['Class'])       	# train['species'] contains the actual species names. Before we can use it,
       	# we need to convert each species name into a digit. So, in this case there
       	# are three species, which have been coded as 0, 1, or 2.
    

    model = GradientBoostingClassifier()
    param_grid = {
        'min_samples_leaf':[1,2,3],
        'min_samples_split':[2,3],
        'max_features': [3,5,8],
        'n_estimators': [100,90,110,120,80],
        'min_weight_fraction_leaf':[0.0,0.1],
        'max_depth': [4,5,6,7],
        'subsample': [0.4, 0.7, 0.8],
        'learning_rate': [0.05,0.1,0.02]
    }
        
        
    
    model_trained, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                     param_grid, cv=5)

    best_params = model_trained.best_params_
    df_model = pd.DataFrame.from_dict(best_params, orient='index') 


    y_pred =pred
    F1_score    = mt.f1_score(y_test, y_pred)
    recal_score = mt.recall_score(y_test, y_pred)
    accuracy    = mt.accuracy_score(y_test, y_pred) 

    if accuracy > 0.85 and F1_score > 0.85:
        rs_list.append([accuracy,F1_score,recal_score])
        model_metrics = {"Accuracy":accuracy , "F1_score":F1_score  ,"recal_score":recal_score, "i": i}
        df_metrics = pd.DataFrame.from_dict(model_metrics,orient='index')                                            
        df_model_metrics = pd.concat([df_metrics, df_model], axis=0, sort=False)
        if first_time:
            df_models = df_model_metrics
            first_time = False
        df_models = pd.concat([df_models,df_model_metrics],axis = 1)
        print('SEED:', i)
        print('Accuracy:', accuracy)
        print('F1_score',F1_score)      
        print('Model:', df_model)
        model_name = 'Classification_Model_'+str(i)+'.pkl'
        pickle.dump(model_trained, open(model_name, "wb"))
                

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
save_obj(df_models,'GBR_models_90p_d2_FR')
# Root Mean Squared Error



