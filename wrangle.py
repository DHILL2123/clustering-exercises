import os
import pandas as pd 
import numpy as np
import matplotlib as plt
from env import protocol, user, host, password, db
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
#############################Acquire###################################################

def get_zillow():
    filename = "zillow_report.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/zillow"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql_query('''
                                
                                SELECT
                                prop.*,
                                predictions_2017.logerror,
                                predictions_2017.transactiondate,
                                air.airconditioningdesc,
                                arch.architecturalstyledesc,
                                build.buildingclassdesc,
                                heat.heatingorsystemdesc,
                                landuse.propertylandusedesc,
                                story.storydesc,
                                construct.typeconstructiondesc
                                FROM properties_2017 prop
                                JOIN (
                                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                                FROM predictions_2017
                                GROUP BY parcelid
                                ) pred USING(parcelid)
                                JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                                AND pred.max_transactiondate = predictions_2017.transactiondate
                                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                                LEFT JOIN storytype story USING (storytypeid)
                                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                                WHERE prop.latitude IS NOT NULL
                                AND prop.longitude IS NOT NULL
                                AND transactiondate <= '2017-12-31' 

                                ''', mysqlcon)  
        

         # renaming column names to one's I like better
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built', 
                              'structuretaxvaluedollarcnt':"structure_taxvalue", "taxamount" : "tax_amount",
                              "propertyzoningdesc" : "zoning_desc", "transactiondate" : "trans_date", "propertylandusedesc":"prop_landuse_desc",
                              "propertycountylandusecode":"county_landuse_code"})

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        return df
################################# Nulls by Row ##############################################
def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)    
################################# Nulls by Column ###########################################
def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)
################################# Get Numeric Columns ########################################
def get_numerics(df):
    '''
    Pulls all numerical columns from the dataframe and returns a 
    dataframe of only numerical columns 
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = df.select_dtypes(include=numerics)
    return numeric_cols
################################# Get Categorical Columns ####################################
def get_categorical(df):
    '''
    Takes in a dataframe and returns a dataframe of only categorical columns
    '''
    cat_cols = ['object']
    cat_cols = df.select_dtypes(include=cat_cols)
    return cat_cols
################################# Outliers Function ##########################################
def outlier_function(df, cols, k):
    '''
    This function takes in a dataframe, column, and k
    to detect and handle outlier using IQR rule
    '''
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df



################################# Sumarize Function #########################################
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

#################################### Drop Columns ###########################################
def remove_columns(df, cols_to_remove):
    '''
    This function takes in a dataframe 
    and the columns that need to be dropped
    then returns the desired dataframe.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
############################# Handle Missing Values with thresholds ############################################
def handle_missing_values(df, prop_required_columns=0.6, prop_required_rows=0.75):
    '''
    This function takes in a dataframe, the percent of columns and rows
    that need to have values/non-nulls
    and returns the dataframe with the desired amount of nulls left.
    '''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df
############################## Get columns with missing values the met our threshold ###################################################
def remaining_missing(df):
    '''
    This function takes in a dataframe 
    and list all columns with missing values
    '''
    df = df.columns[df.isnull().any()]

    return df
################################### imput remaining missing values #######################################
# impute missing values with most frequent value using 'most_frequent'
def impute_remaining(df):
    '''
    This function takes the a dataframe and imputes the misisng values 
    with the most frequent value in that column.
    '''
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df[['Unnamed: 0', 'id', 'parcelid', 'bathrooms', 'bedrooms',
       'buildingqualitytypeid', 'calculatedbathnbr', 'area',
       'finishedsquarefeet12', 'fips', 'fullbathcnt', 'heatingorsystemtypeid',
       'latitude', 'longitude', 'lotsizesquarefeet', 'propertylandusetypeid',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidzip', 'roomcnt', 'unitcnt', 'year_built', 'structure_taxvalue',
       'tax_value', 'assessmentyear', 'landtaxvaluedollarcnt', 'tax_amount',
       'censustractandblock', 'logerror']])

    df[['Unnamed: 0', 'id', 'parcelid', 'bathrooms', 'bedrooms',
       'buildingqualitytypeid', 'calculatedbathnbr', 'area',
       'finishedsquarefeet12', 'fips', 'fullbathcnt', 'heatingorsystemtypeid',
       'latitude', 'longitude', 'lotsizesquarefeet', 'propertylandusetypeid',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidzip', 'roomcnt', 'unitcnt', 'year_built', 'structure_taxvalue',
       'tax_value', 'assessmentyear', 'landtaxvaluedollarcnt', 'tax_amount',
       'censustractandblock', 'logerror']] = imputer.transform(df[['Unnamed: 0', 'id', 'parcelid', 'bathrooms', 'bedrooms',
       'buildingqualitytypeid', 'calculatedbathnbr', 'area',
       'finishedsquarefeet12', 'fips', 'fullbathcnt', 'heatingorsystemtypeid',
       'latitude', 'longitude', 'lotsizesquarefeet', 'propertylandusetypeid',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidzip', 'roomcnt', 'unitcnt', 'year_built', 'structure_taxvalue',
       'tax_value', 'assessmentyear', 'landtaxvaluedollarcnt', 'tax_amount',
       'censustractandblock', 'logerror']])
    
    return df
#####################################################################

########################### Prepare ########################
def pepare_zillow(df):
 
    df['fips'] = df['fips'].replace([6037.0, 6059.0,6111.0], ['Los Angeles County, CA', 'Orange County, CA','Ventura County, CA'])

    df = df.drop(columns=['calculatedbathnbr', 'finishedsquarefeet12'])

    # get distributions of numeric data
    #get_hist(df)
    #get_box(df)
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)

    df['zoning_desc'] = df['zoning_desc'].fillna(df.zoning_desc.mode())
    df['heatingorsystemdesc'] = df['heatingorsystemdesc'].fillna(df.heatingorsystemdesc.mode())


    df = pd.concat([df, pd.get_dummies(data=df, columns=['zoning_desc', 'prop_landuse_desc','county_landuse_code',
                                                                        'fips','heatingorsystemdesc'])], axis=1)
    df = df.drop(columns=['regionidcounty'])
    return df

################# Data Scalers ###########################################
def robust_scaler(x_train,x_validate,x_test, numeric_cols):
    scaler = sklearn.preprocessing.RobustScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled

def minmax_scaler(x_train, x_validate, x_test, numeric_cols):
    ######## Min Max Scaler (range calculations)
    scaler = sklearn.preprocessing.MinMaxScaler(copy=True)
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train[numeric_cols])
    ### Apply to train, validate, and test
    x_train_scaled_array = scaler.transform(x_train[numeric_cols])
    x_validate_scaled_array = scaler.transform(x_validate[numeric_cols])
    x_test_scaled_array = scaler.transform(x_test[numeric_cols])

 # convert arrays to dataframes
    x_train_scaled = pd.DataFrame(x_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([x_train.index.values])

    x_validate_scaled = pd.DataFrame(x_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([x_validate.index.values])

    x_test_scaled = pd.DataFrame(x_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([x_test.index.values])

    return x_train_scaled, x_validate_scaled, x_test_scaled

def standard_scaler(x_train,x_validate,x_test,numeric_cols):
    scaler = sklearn.preprocessing.StandardScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled

