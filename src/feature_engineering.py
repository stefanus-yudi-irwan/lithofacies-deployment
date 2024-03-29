import warnings
warnings.filterwarnings('ignore')
import numpy as np     # for calculation
import pandas as pd     # for manipulating DataFrame
import yaml     # for interacting with config.yaml  
import util as util     # import common function
from sklearn.preprocessing import OneHotEncoder     # for manipulating categorical data
from sklearn.preprocessing import StandardScaler     # for normalize data
from imblearn.under_sampling import RandomUnderSampler     # for creating random under sample data
from imblearn.over_sampling import RandomOverSampler, SMOTE     # for creating random over sample and SMOTE data


def split_numerical_categorical(data:pd.DataFrame):
    print("splitting numerical and categorical data .....")
    data = data[["GR","ILD_log10","DeltaPHI","PHIND","PE","NM_M"]]
    X_train_categorical = data["NM_M"]
    X_train_numerical = data.drop(columns="NM_M", axis=1)
    
    print("numerical and categorical data splitted")
    return X_train_numerical, X_train_categorical

def categorical_handling(data:pd.Series) -> pd.DataFrame:
    print("encode categorical data using OHE")
    non_marine_marine_label = {1:"non_marine",2:"marine"}
    data_marine = data.map(non_marine_marine_label)
    ohe_train = OneHotEncoder(sparse=False)
    ohe_train.fit(np.array(["non_marine","marine"]).reshape(-1,1))
    ohe_data = ohe_train.transform(np.array(data_marine.to_list()).reshape(-1,1))
    ohe_data = pd.DataFrame(data=ohe_data, columns=list(ohe_train.categories_[0]), index=data_marine.index)
    
    print("categorical data has been encoded using OHE")
    return ohe_data, ohe_train

def normalize_data(numerical_data:pd.DataFrame, categorical_data:pd.DataFrame) -> pd.DataFrame:
    print("normalize data ..... ")
    data = pd.concat([numerical_data, categorical_data], axis=1)
    standard_scaler_train = StandardScaler()
    standard_scaler_train.fit(data.values)
    normalized_data = standard_scaler_train.transform(data.values)
    normalized_data = pd.DataFrame(data=normalized_data, index=data.index, columns=data.columns)
    
    print("data has been normalized")
    return normalized_data, standard_scaler_train

def facies_encoder(data:pd.Series) -> pd.Series :
    print("encode label facies .....")
    facies_label_encoder = {'SS':0, 'CSiS':1, 'FSiS':2, 'SiSh':3, 'MS':4, 'WS':5, 'D':6, 'PS':7, 'BS':8}
    encoded_data = data.map(facies_label_encoder)
    
    print("facies label has been encoded")
    return encoded_data

def make_modeling_data(X_data:pd.DataFrame, y_data:pd.DataFrame) -> dict :
    print("create data train for modeling .....")
    rus = RandomUnderSampler()
    ros = RandomOverSampler()
    smote = SMOTE()
    
    X_train_rus, y_train_rus = rus.fit_resample(X = X_data.values, y = y_data)
    X_train_ros, y_train_ros = ros.fit_resample(X = X_data.values, y=y_data)
    X_train_smote, y_train_smote = smote.fit_resample(X = X_data.values,y = y_data)
    
    X_train_rus = pd.DataFrame(data=X_train_rus, columns=X_data.columns)
    X_train_ros = pd.DataFrame(data=X_train_ros, columns=X_data.columns)
    X_train_smote = pd.DataFrame(data=X_train_smote, columns=X_data.columns)
    
    modeling_data = {"X_train":{"unbalance":X_data, "rus":X_train_rus, "ros":X_train_ros, "smote": X_train_smote},
                     "y_train":{"unbalance":y_data, "rus":y_train_rus, "ros":y_train_ros, "smote": y_train_smote}}
    
    print("Unbalance, RUS, ROS, SMOTE, data has been generated")
    return modeling_data

def categorical_handling_test_data(data:pd.Series, ohe:OneHotEncoder) -> pd.DataFrame:
    print("encode categorical test data using OHE object from train data .....")
    non_marine_marine_label = {1:"non_marine",2:"marine"}
    data_marine = data.map(non_marine_marine_label)
    ohe_data = ohe.transform(np.array(data_marine.to_list()).reshape(-1,1))
    ohe_data = pd.DataFrame(data=ohe_data, columns=list(ohe.categories_[0]), index=data_marine.index)
    
    print("categorical test data has been encoded using OHE object from train data")
    return ohe_data

def normalize_test_data(numerical_data:pd.DataFrame, categorical_data:pd.DataFrame, scaler:StandardScaler) -> pd.DataFrame:
    print("normalize train data using StandardScaler train data ..... ")
    data = pd.concat([numerical_data, categorical_data], axis=1)
    normalized_data = scaler.transform(data.values)
    normalized_data = pd.DataFrame(data=normalized_data, index=data.index, columns=data.columns)
    
    print("test data has been normalized")
    return normalized_data

def create_test_data (X_data:pd.DataFrame, y_data:pd.DataFrame) -> dict:
    print("create test data for testing model .....")
    test_data = {'X_test':X_data, 'y_test':y_data}

    print("test data for testing model has been created")
    return test_data

if __name__ == "__main__":
    config = util.load_config()

    print("===== Feature Engineering on Train Data =====")
    # Load clean file from the directory
    X_train_clean = util.load_pickle(file_path=config['data_train_path'][0])
    y_train_clean = util.load_pickle(file_path=config['data_train_path'][1])
    X_test_clean = util.load_pickle(file_path=config['data_test_path'][0])
    y_test_clean = util.load_pickle(file_path=config['data_test_path'][1])

    # split numerical and categorical data
    X_train_numerical, X_train_categorical = split_numerical_categorical(data=X_train_clean)

    # Handling Categorical data
    X_train_categorical, ohe_train = categorical_handling(data=X_train_categorical)
    util.dump_pickle(data=ohe_train, file_path=config["ohe_train_path"])

    # Normalize Data
    X_train, standard_scaler_train = normalize_data(numerical_data=X_train_numerical,
                                                categorical_data=X_train_categorical)
    util.dump_pickle(data=standard_scaler_train, file_path=config['standard_scaler_path'])

    # Label encode output feature
    y_train_numerical = facies_encoder(data=y_train_clean)

    # create train data for modeling
    train_data = make_modeling_data(X_data=X_train, y_data=y_train_numerical)
    util.dump_pickle(data=train_data, file_path=config['data_modeling_path'][0])


    # Do the same for test data
    print("===== Feature Engineering on Test Data =====")
    X_test_numerical, X_test_categorical = split_numerical_categorical(data=X_test_clean)
    X_test_categorical = categorical_handling_test_data(data=X_test_categorical, ohe=ohe_train)
    X_test = normalize_test_data(numerical_data=X_test_numerical,
                            categorical_data=X_test_categorical,
                            scaler=standard_scaler_train)
    y_test_numerical = facies_encoder(y_test_clean)
    test_data = create_test_data(X_data=X_test, y_data=y_test_numerical)
    util.dump_pickle(data=test_data, file_path=config['data_modeling_path'][1])
