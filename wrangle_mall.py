from sklearn.impute import SimpleImputer
import os
import pandas as pd 
import numpy as np
from env import protocol, user, host, password, db
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler



def get_mall_data():
    filename = "mall_customers.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/mall_customers"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql_query('''
                                
                                select * from customers;

                                ''', mysqlcon)  
        

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        return df