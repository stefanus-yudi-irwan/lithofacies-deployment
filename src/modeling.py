import warnings
warnings.filterwarnings('ignore')

import numpy as np     # for calculation
import pandas as pd     # for manipulating DataFrame
import yaml     # for interacting with config.yaml  
import util as util     # import common function
import os     # library for interacting with directory

# import machine learning library
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# import machine learning evaluation library
from sklearn.metrics import confusion_matrix

def create_model_object() -> list:
    knn = KNeighborsClassifier()       # create object k-nearest neighbors classifier
    dct = DecisionTreeClassifier()     # create object decision tree classifier
    logreg = LogisticRegression()      # create object logistic regression
    svm = SVC()                        # create object support vector machine
    rfc = RandomForestClassifier()     # create object random forest classifier
    xgb = XGBClassifier()              # create object extreme gradient boosting classifier
    
    list_of_model = [
        {"model_name" : knn.__class__.__name__, "model_object":knn},
        {"model_name" : dct.__class__.__name__, "model_object":dct},
        {"model_name" : logreg.__class__.__name__, "model_object":logreg},
        {"model_name" : svm.__class__.__name__, "model_object":svm},
        {"model_name" : rfc.__class__.__name__, "model_object":rfc},
        {"model_name" : xgb.__class__.__name__, "model_object":xgb}]
    return list_of_model

def model_hyperparameter(model_name:str) -> dict:
    knn_hyper_parameter = {
        "algorithm" : ["ball_tree", "kd_tree", "brute"],
        "n_neighbors" : [2, 5, 10, 25],
        "leaf_size" : [2, 5, 10, 25],
    }
    dct_hyper_parameter = {
        "criterion" : ["gini", "entropy", "log_loss"],
        "max_depth" : [1,5,10],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4]
    }
    logreg_hyper_parameter = {
        "penalty" : ["l2","l1","elasticnet"],
        "C" : [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "max_iter" : np.arange(100,210,10)    
    }
    svm_hyper_parameter = {
        "C" : [0.001, 0.05, 0.1, 1, 5, 10, 25, 50],
        "kernel" : ["linear","rbf"]
    }
    rfc_hyper_parameter = {
        "criterion" : ["gini", "entropy", "log_loss"],
        "n_estimators" : [1,5,10],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4]
    }
    xgb_hyper_parameter = {
        "n_estimators" : [1,5,10,25,50,100]
    }
        
    list_of_hyper_parameter = {
        "KNeighborsClassifier" : knn_hyper_parameter,
        "DecisionTreeClassifier" : dct_hyper_parameter,
        "LogisticRegression" : logreg_hyper_parameter,
        "SVC" : svm_hyper_parameter,
        "RandomForestClassifier" : rfc_hyper_parameter,
        "XGBClassifier" : xgb_hyper_parameter
    }
    
    return list_of_hyper_parameter[model_name]

def brute_force_modeling(train_data:dict, test_data:dict) -> dict:
    print("========== Starting Train Model ==========")

    list_of_model = create_model_object()
    modeling_summary = {}

    for model in list_of_model:
        print(f"-----Start Training model for {model['model_name']}-----")
        # create variable for storing indformation
        model_summary = {}
        model_summary['model_highest_accuracy']=0 

        for data_type in ["unbalance","rus","ros","smote"]:
            # Experiment by GridSearch CV
            print(f"Training Model {model['model_name']} for {data_type} data...")
            model_cv = GridSearchCV(estimator = model['model_object'], param_grid=model_hyperparameter(model['model_name']), cv=10)
            model_cv.fit(X = train_data["X_train"][data_type].values, y=train_data["y_train"][data_type])
        

            # Create the model with best params
            # KNeighborsClassifier Modeling
            if model['model_name'] == 'KNeighborsClassifier' :
                model_train = KNeighborsClassifier(algorithm = model_cv.best_params_['algorithm'],
                                            n_neighbors = model_cv.best_params_['n_neighbors'],
                                            leaf_size = model_cv.best_params_['leaf_size'])
            # DecisionTreeClassifier Modeling
            elif model['model_name'] == "DecisionTreeClassifier" :
                model_train = DecisionTreeClassifier(criterion = model_cv.best_params_['criterion'],
                                               max_depth = model_cv.best_params_['max_depth'],
                                               min_samples_leaf = model_cv.best_params_['min_samples_leaf'],
                                               min_samples_split = model_cv.best_params_['min_samples_split'])
            # LogisticRegression Modeling 
            elif model['model_name'] == 'LogisticRegression' :
                model_train = LogisticRegression(penalty = model_cv.best_params_["penalty"],
                                           max_iter = model_cv.best_params_["max_iter"],
                                           C = model_cv.best_params_["C"])
            # SVC Modeling
            elif model['model_name'] == 'SVC' :
                model_train = SVC(C = model_cv.best_params_["C"],
                            kernel = model_cv.best_params_["kernel"])
            # RandomForestClassifier Modeling
            elif model['model_name'] == 'RandomForestClassifier' :
                model_train = RandomForestClassifier(n_estimators = model_cv.best_params_["n_estimators"],
                                               criterion = model_cv.best_params_["criterion"],
                                               min_samples_leaf = model_cv.best_params_["min_samples_leaf"],
                                               min_samples_split = model_cv.best_params_['min_samples_split'])                
            # XGBClassifier Modeling
            else :
                model_train = XGBClassifier(n_estimators=model_cv.best_params_["n_estimators"],
                                            num_class=9)


            # Fit best model to train data
            model_train.fit(X=train_data["X_train"][data_type].values,
                        y=train_data["y_train"][data_type])
            
            # Predict output using train data
            y_train_pred  = model_train.predict(train_data["X_train"][data_type].values)
            
            # create confusion matrix for training data
            conf_train = confusion_matrix(y_true=train_data["y_train"][data_type],
                                        y_pred=y_train_pred)
            
            # Predict output using test data
            y_test_pred = model_train.predict(test_data['X_test'].values)
            
            
            # create confusion matrix for testing data
            conf_test = confusion_matrix(y_true=test_data['y_test'], 
                                        y_pred=y_test_pred)
            
            # summarize the modeling
            model_summary[data_type] = {"model":model_train, 
                                    "cv_score":model_cv.best_score_,
                                    "acc_train":util.accuracy(conf_train), 
                                    "adj_acc_train":util.accuracy_adjacent(conf_train),
                                    "acc_test": util.accuracy(conf_test),
                                    "adj_acc_test":util.accuracy_adjacent(conf_test)}
            
            # get the best model from accuracy value
            if model_summary[data_type]["acc_test"] > model_summary['model_highest_accuracy']:
                model_summary['model_highest_accuracy'] = model_summary[data_type]["acc_test"] 
                model_summary['best_model'] = model_summary[data_type]['model']
                model_summary['best_data'] = data_type
                model_summary['model_best_score'] = {"cv_score" : model_summary[data_type]["cv_score"],
                                                     "acc_test" : model_summary[data_type]["acc_test"],
                                                     "adj_acc_test" : model_summary[data_type]["adj_acc_test"]}

        # put modeling information into a dictionary
        print(f"-----End Training model for {model['model_name']}-----")
        modeling_summary[model['model_name']] = model_summary
    print("========== Ending Train Model ==========")     
    return modeling_summary

def train_model(train_data:dict, test_data:dict, retrain_model:bool=False):
    if os.path.exists("models/production_model.pkl"):
        if retrain_model:
            modeling_summary = brute_force_modeling(train_data=train_data, test_data=test_data)
        else :
            modeling_summary = util.load_pickle(config['modeling_summary'])
    else:
        modeling_summary = brute_force_modeling(train_data=train_data, test_data=test_data)
    return modeling_summary  

def pick_best_model(modeling_summary:dict):

    model_best_accuracy = 0
    best_model = []
    for model in modeling_summary:
        if modeling_summary[model]['model_highest_accuracy'] > model_best_accuracy:
            model_best_accuracy = modeling_summary[model]['model_highest_accuracy']
            best_model = modeling_summary[model]['best_model']
    
    return best_model

if __name__ == "__main__":
    config = util.load_config()
    # import data from pickle
    train_data = util.load_pickle(file_path=config['data_modeling_path'][0])
    test_data = util.load_pickle(file_path=config['data_modeling_path'][1])
    modeling_summary = train_model(train_data = train_data, test_data = test_data, retrain_model = False)
    util.dump_pickle(data=modeling_summary, file_path=config['modeling_summary'])
    production_model = pick_best_model(modeling_summary=modeling_summary)
    util.dump_pickle(data=production_model, file_path=config['production_model'])