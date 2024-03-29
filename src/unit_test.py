import pandas as pd
import util as util
import data_preprocessing as prep
import feature_engineering as feng

config = util.load_config()

# ================================ Data Preprocessing Unit Test =================================================

# test output of populate_raw_data() function outputing all raw data ane data Frame
raw_data = prep.populate_raw_data(config["raw_dataset_dir"])

def test_populate_raw_data():
    raw_data = prep.populate_raw_data(config["raw_dataset_dir"])
    
    assert len(raw_data) == 4149
    assert type(raw_data) == pd.DataFrame

# test split_train_test() function resulting DataFrame and Series
def test_split_train_test_data():
    X_train_unclean, y_train_unclean, X_test_unclean, y_test_unclean = prep.split_train_test_data(raw_data = raw_data, output='Facies')
    
    assert type(X_train_unclean) == pd.DataFrame
    assert type(y_train_unclean) == pd.Series
    assert type(X_test_unclean) == pd.DataFrame
    assert type(y_test_unclean) == pd.Series

# test impute_PE_data_train() function resulting all nan value filled
def test_impute_PE_data_train():
    X_train_unclean, y_train_unclean, _, _ = prep.split_train_test_data(raw_data = raw_data, output='Facies')
    X_train_clean, y_train_clean = prep.impute_PE_data_train(X_data=X_train_unclean, y_data=y_train_unclean)
   
    assert X_train_clean.isnull().sum().sum() == 0
    assert y_train_clean.isnull().sum().sum() == 0

# test impute_PE_data_test() function resulting all nan value filled
def test_impute_PE_data_test():
    X_train_unclean, y_train_unclean, X_test_unclean, y_test_unclean = prep.split_train_test_data(raw_data = raw_data, output='Facies')
    X_test_clean, y_test_clean = prep.impute_PE_data_test(X_data=X_test_unclean, y_data=y_test_unclean,     # impute test data using data from train data
                                                 X_imputer=X_train_unclean, y_imputer=y_train_unclean)
    
    assert X_test_clean.isnull().sum().sum() == 0
    assert y_test_clean.isnull().sum().sum() == 0

# test check_data_type() function resulting correct data type for modeling features
def test_check_data_type():
    X_train_unclean, y_train_unclean, _, _ = prep.split_train_test_data(raw_data = raw_data, output='Facies')
    X_train_clean, y_train_clean = prep.impute_PE_data_train(X_data=X_train_unclean, y_data=y_train_unclean)
    X_train_clean, y_train_clean = prep.check_data_type(input_data=X_train_clean,
                                               output_data=y_train_clean,
                                               data_type="train")
   
    assert X_train_clean['GR'].dtype == 'float64'
    assert X_train_clean['ILD_log10'].dtype == 'float64'
    assert X_train_clean['DeltaPHI'].dtype == 'float64'
    assert X_train_clean['PHIND'].dtype == 'float64'
    assert X_train_clean['PE'].dtype == 'float64'
    assert X_train_clean['NM_M'].dtype == 'int32'

# test check_data_range() function true for all modeling features
def test_check_data_range():
    X_train_unclean, y_train_unclean, X_test_unclean, y_test_unclean = prep.split_train_test_data(raw_data = raw_data, output='Facies')
    X_train_clean, y_train_clean = prep.impute_PE_data_train(X_data=X_train_unclean, y_data=y_train_unclean)
    X_train_clean, y_train_clean = prep.check_data_type(input_data=X_train_clean,
                                               output_data=y_train_clean,
                                               data_type="train")
    prep.check_data_range(input_data=X_train_clean, output_data=y_train_clean, data_type="train")
    
# ================================ Feature Engineering Unit Test =================================================
X_train_clean = util.load_pickle(file_path=config['data_train_path'][0])
y_train_clean = util.load_pickle(file_path=config['data_train_path'][1])
X_test_clean = util.load_pickle(file_path=config['data_test_path'][0])
y_test_clean = util.load_pickle(file_path=config['data_test_path'][1])

# test split_numerical_categorical() function output DataFrame and Series
def test_split_numerical_categorical():
    X_train_numerical, X_train_categorical = feng.split_numerical_categorical(data=X_train_clean)
    
    assert type(X_train_numerical) == pd.DataFrame
    assert type(X_train_categorical) == pd.Series

# test categorical_handling() function output DataFrame
def test_categorical_handling():
    _, X_train_categorical = feng.split_numerical_categorical(data=X_train_clean)
    X_train_categorical, ohe_train = feng.categorical_handling(data=X_train_categorical)
    
    assert type(X_train_categorical) == pd.DataFrame 

# test normalize_data() functio output normalize data
def test_normalize_data():
    X_train_numerical, X_train_categorical = feng.split_numerical_categorical(data=X_train_clean)
    X_train_categorical, _ = feng.categorical_handling(data=X_train_categorical)
    X_train, _ = feng.normalize_data(numerical_data=X_train_numerical,
                                    categorical_data=X_train_categorical)

    assert X_train.mean().sum() > -0.01 and X_train.mean().sum() < 0.01
    assert X_train.std().sum() > 6.9 and X_train.std().sum() < 7.1

# test_facies_encoder() function outputing numerical data
def test_facies_encoder():
    y_train_numerical = feng.facies_encoder(data=y_train_clean)

    assert set(y_train_numerical).issubset({0,1,2,3,4,5,6,7,8})

# test make_modeling_data() function make unbalance, rus, ros, and smote
def test_make_modeling_data():
    X_train_numerical, X_train_categorical = feng.split_numerical_categorical(data=X_train_clean)
    X_train, _ = feng.normalize_data(numerical_data=X_train_numerical,
                                    categorical_data=X_train_categorical)
    y_train_numerical = feng.facies_encoder(data=y_train_clean)
    train_data = feng.make_modeling_data(X_data=X_train, y_data=y_train_numerical)

    assert set(train_data.keys()).issubset({'X_train','y_train'})
    assert set(train_data['X_train'].keys()).issubset({'ros','rus','unbalance','smote'})

# test categorical_hendling_test_data() outputing dataFrame
def test_categorical_handling_test_data():
    ohe_train = util.load_pickle(config['ohe_train_path'])
    X_test_numerical, X_test_categorical = feng.split_numerical_categorical(data=X_test_clean)
    X_test_categorical = feng.categorical_handling_test_data(data=X_test_categorical, ohe=ohe_train)

    assert type(X_test_categorical) == pd.DataFrame

# test normalize_test_data() function output normal data
def test_normalize_test_data():
    ohe_train = util.load_pickle(config['ohe_train_path'])
    standard_scaler_train = util.load_pickle(config['standard_scaler_path'])
    X_test_numerical, X_test_categorical = feng.split_numerical_categorical(data=X_test_clean)
    X_test_categorical = feng.categorical_handling_test_data(data=X_test_categorical, ohe=ohe_train)
    X_test = feng.normalize_test_data(numerical_data=X_test_numerical,
                            categorical_data=X_test_categorical,
                            scaler=standard_scaler_train)

    assert X_test.mean().sum() > -0.01 and X_test.mean().sum() < 0.01
    assert X_test.std().sum() > 6.9 and X_test.std().sum() < 7.1
    
# test create_test_data() outputing test data
def test_create_test_data():
    ohe_train = util.load_pickle(config['ohe_train_path'])
    standard_scaler_train = util.load_pickle(config['standard_scaler_path'])
    X_test_numerical, X_test_categorical = feng.split_numerical_categorical(data=X_test_clean)
    X_test_categorical = feng.categorical_handling_test_data(data=X_test_categorical, ohe=ohe_train)
    X_test = feng.normalize_test_data(numerical_data=X_test_numerical,
                            categorical_data=X_test_categorical,
                            scaler=standard_scaler_train)
    y_test_numerical = feng.facies_encoder(y_test_clean)                       
    test_data = feng.create_test_data(X_data=X_test, y_data=y_test_numerical)

    assert set(test_data.keys()).issubset({'X_test','y_test'})

