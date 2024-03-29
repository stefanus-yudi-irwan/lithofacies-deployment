import warnings
warnings.filterwarnings('ignore')
import os     # for interacting with directory
import numpy as np     # for calculation
import pandas as pd     # for manipulating DataFrame
import yaml     # for interacting with config.yaml  
import util as util     # import common function
from sklearn.model_selection import train_test_split     # for splitting train and test data

def populate_raw_data(directory:str) -> pd.DataFrame:
    print("creating raw data .....")
    raw_data = pd.DataFrame()
    for well_data in os.listdir(directory):
        raw_data = pd.concat([raw_data,pd.read_csv(directory+well_data)],
                             axis=0,
                             ignore_index=True)
    print("raw data created")
    return raw_data

def split_train_test_data(raw_data:pd.DataFrame, output:'str'='Facies') -> pd.DataFrame:
    print("splitting train and test data .....")
    X=raw_data.drop(columns=output, axis=1)
    y=raw_data[output]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123, stratify=y)
    print(f"Train data consist of X_train : {X_train.shape[0]} row and y_train : {y_train.shape[0]} row")
    print(f"Test data consist of X_test : {X_test.shape[0]} row and y_test : {y_test.shape[0]} row")

    print('train and test data splitted')
    return X_train, y_train, X_test, y_test

def impute_PE_data_train(X_data:pd.DataFrame, y_data:pd.DataFrame) -> pd.DataFrame:
    print("impute train data missing value .....")
    if X_data["PE"].isnull().sum() == 0:
        pass
    else :
        data = pd.concat([X_data, y_data], axis=1)
        PE_mean_data = data.loc[~(data["PE"].isnull()),["Facies","PE"]].groupby("Facies").agg('mean')
        PE_nan_data = data.loc[data["PE"].isnull(),["Facies","PE"]]
        facies_labels = data["Facies"].unique().tolist()
        for facies in facies_labels:
            index_PE_nan_by_categories = PE_nan_data[PE_nan_data['Facies']==facies].index.to_list()    
            data.loc[index_PE_nan_by_categories,'PE'] = PE_mean_data.loc[facies].values[0]
        X_data = data.drop(columns='Facies', axis=1)
        y_data = data['Facies']
    print(f"train data has been imputed")
    return X_data, y_data

def impute_PE_data_test(X_data:pd.DataFrame, y_data:pd.DataFrame, X_imputer:pd.DataFrame, y_imputer:pd.DataFrame) -> pd.DataFrame:
    print("impute test data missing value .....")
    if X_data["PE"].isnull().sum() == 0:
        pass
    else :
        data = pd.concat([X_data, y_data], axis=1)
        imputer_data = pd.concat([X_imputer, y_imputer], axis=1)
        PE_mean_data = imputer_data.loc[~(imputer_data["PE"].isnull()),["Facies","PE"]].groupby("Facies").agg('mean')
        PE_nan_data = data.loc[data["PE"].isnull(),["Facies","PE"]]
        facies_labels = data["Facies"].unique().tolist()
        for facies in facies_labels:
            index_PE_nan_by_categories = PE_nan_data[PE_nan_data['Facies']==facies].index.to_list()    
            data.loc[index_PE_nan_by_categories,'PE'] = PE_mean_data.loc[facies].values[0]
        X_data = data.drop(columns='Facies', axis=1)
        y_data = data['Facies']
    print(f"test data has been imputed")
    return X_data, y_data

def check_data_type(input_data:pd.DataFrame, output_data:pd.DataFrame, data_type:"str") -> pd.DataFrame :
    print(f"checking {data_type} data type ..... ")
    output_data = output_data.astype('str')
    input_data[['Formation','Well Name']] = input_data[['Formation','Well Name']].astype('str')
    input_data[['GR','ILD_log10','DeltaPHI','PHIND','PE','RELPOS']] = input_data[['GR','ILD_log10','DeltaPHI','PHIND','PE','RELPOS']].astype('float64')
    input_data['NM_M'] = input_data['NM_M'].astype('int32')
    
    print(f"{data_type} data type checked")
    return input_data, output_data

def check_data_range(input_data:pd.DataFrame, output_data:pd.DataFrame, data_type:"str") -> pd.DataFrame :
    print(f"checking {data_type} data range .....")
    # value check range
    gr_range = [10.1, 361.2]
    ild_range = [-0.03, 1.81]
    deltaphi_range = [-22, 19.4]
    phind_range = [0.54, 84.5]
    pe_range = [0.19, 8.11]
    nm_m_range = set([1,2])
    facies_range = set(['CSiS', 'FSiS', 'MS', 'PS', 'WS', 'SiSh', 'D', 'SS', 'BS'])
    
    assert (input_data['GR'] >= gr_range[0]).all() and (input_data['GR'] <= gr_range[1]).all(), "GR data out of range"
    assert (input_data['ILD_log10'] >= ild_range[0]).all() and (input_data['ILD_log10'] <= ild_range[1]).all(), "ILD_log10 data out of range"
    assert (input_data['DeltaPHI'] >= deltaphi_range[0]).all() and (input_data['DeltaPHI'] <= deltaphi_range[1]).all(), "DeltaPHI data out of range"
    assert (input_data['PHIND'] >= phind_range[0]).all() and (input_data['PHIND'] <= phind_range[1]).all(), "PHIND data out of range"
    assert (input_data['PE'] >= pe_range[0]).all() and (input_data['PE'] <= pe_range[1]).all(), "PE data out of range"
    assert set(input_data['NM_M'].unique().tolist()).issubset(nm_m_range), "NM_M data Out of Category"
    assert set(output_data.unique().tolist()).issubset(facies_range), "Facies data Out Of Category"
    
    print(f"=== All data {data_type} checked and verified ===")


if __name__ == "__main__":
    # load comfiguration file
    config = util.load_config()

    # create raw data
    raw_data = populate_raw_data(config["raw_dataset_dir"])
    util.dump_pickle(raw_data,config["raw_data_path"])     # dump pickle for raw data
    
    # split train and test data
    X_train_unclean, y_train_unclean, X_test_unclean, y_test_unclean = split_train_test_data(raw_data = raw_data, output='Facies')     # split train and test data with same proportion of category
    
    # impute missing value in train and test data
    X_train_clean, y_train_clean = impute_PE_data_train(X_data=X_train_unclean, y_data=y_train_unclean)     # impute train data 
    X_test_clean, y_test_clean = impute_PE_data_test(X_data=X_test_unclean, y_data=y_test_unclean,     # impute test data using data from train data
                                                 X_imputer=X_train_unclean, y_imputer=y_train_unclean)
    
    # casting data type for every column in train and test data
    X_train_clean, y_train_clean = check_data_type(input_data=X_train_clean,
                                               output_data=y_train_clean,
                                               data_type="train")
    X_test_clean, y_test_clean = check_data_type(input_data=X_test_clean,
                                               output_data=y_test_clean,
                                               data_type="test")

    # check data range for every column in train data and test data
    check_data_range(input_data=X_train_clean, output_data=y_train_clean, data_type="train")
    check_data_range(input_data=X_test_clean, output_data=y_test_clean, data_type="test")

    # dump pickle for clean data                                            
    util.dump_pickle(data=X_train_clean, file_path=config['data_train_path'][0])
    util.dump_pickle(data=y_train_clean, file_path=config['data_train_path'][1])
    util.dump_pickle(data=X_test_clean, file_path=config['data_test_path'][0])
    util.dump_pickle(data=y_test_clean, file_path=config['data_test_path'][1])