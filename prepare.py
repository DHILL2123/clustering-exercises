import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

X= ''
y= ''
target = ''
train_test_split=''

####### Split data #############
def train_test_split(df):
    x_train_and_validate, x_test = train_test_split(df, random_state=123)
    x_train, x_validate = train_test_split(x_train_and_validate)
    return x_train, x_validate, x_test, x_train_and_validate

################# Data Scalers ##################
def minmax_scaler(x_train, x_validate, x_test,):
    ######## Min Max Scaler (range calculations)
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)
    ### Apply to train, validate, and test
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_validate_scaled, x_test_scaled


########### Separate data to address outliers into above 75% and below 25% ###########
########### Removing the top 25% and bottom 25% working with the 50% in the middle ########
########### Wrapped code into a function #############
#iqr = df['bathrooms'].quantile(0.75) - df['bathrooms'].quantile(0.25)
#lower_bathroom_fence = df['bathrooms'].quantile(0.25) - (1.5*iqr)
#upper_bathroom_fence = df['bathrooms'].quantile(0.75) + (1.5*iqr)
#df[(df.bathrooms > lower_bathroom_fence) & (df.bathrooms < upper_bathroom_fence)].bathrooms.describe()

#col_qs = {}
#for col in cols:
#   col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])

#col_qs['bedrooms'][0.25]


def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns
    using the tukey method.
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        print(type(lower_fence))
        print(lower_fence)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df





#####  prep data titanic functions ####
def clean_data(df):
    '''
    This function will drop any duplicate observations, 
    drop ['deck', 'embarked', 'class', 'age'], fill missing embark_town with 'Southampton'
    and create dummy vars from sex and embark_town. 
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def clean_titanic(df_titanic):
    '''
    Clean titanic will take in a single pandas dataframe
    and will proceed to drop redundant couns 
    and nonuseful information 
    in addition to addressing null values
    and encoding categorical variables
    '''  
    #impute average age and most common embark_town:
    df_titanic['age'] = df_titanic['age'].fillna(df_titanic.age.mean())
    df_titanic['embark_town'] = df_titanic['embark_town'].fillna('Southhampton')

    #encode categorical values
    df_titanic = pd.concat(
    [df_titanic, pd.get_dummies(df_titanic[['sex','embark_town']], drop_first=True)], axis=1)
    
    df_titanic = df_titanic.drop(columns={'Unnamed: 0','passenger_id','embarked', 'deck','class','sex','embark_town'}).rename(columns={'sibsp' : 'sibling_spouse'})

     # convert column names to lowercase, replace '.' in column names with '_'
    df_titanic.columns = [col.lower().replace('.', '_') for col in df_titanic]
    # Drop duplicates...run just in case; reassign and check the shape of my data.
    df_titanic = df_titanic.drop_duplicates()
    
    return df_titanic

# drop rows where age or embarked is null, drop column 'deck', drop passenger_id

def prep_titanic(df):
    '''
    take in titanc dataframe, remove all rows where age or embarked is null, 
    get dummy variables for sex and embark_town, 
    and drop sex, deck, passenger_id, class, and embark_town. 
    '''

    df = df[(df.age.notna()) & (df.embarked.notna())]
    df = df.drop(columns=['deck', 'passenger_id', 'class'])

    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], prefix=['sex', 'embark'])

    df = pd.concat([df, dummy_df.drop(columns=['sex_male'])], axis=1)

    df = df.drop(columns=['sex', 'embark_town']) 

    df = df.rename(columns={"sex_female": "is_female"})
    # Drop duplicates...run just in case; reassign and check the shape of my data.
    df = df.drop_duplicates()

    return df

####### prep iris data functions ######

def prep_iris(df_iris):
    ''' prep iris will take in a single pandas dataframe
    that will presumably match the columns and shape that we 
    expect from our acquire modules get_iris_data 
    functional return 
    '''
    df_iris = df_iris.drop(columns={'species_id','measurement_id','Unnamed: 0'}).rename(columns={'species_name' : 'species'})
    df_iris = pd.concat([df_iris, dummy_df], axis=1)
    df_iris = df_iris

    return df_iris.head()

###### prep telco data function #######
def prep_telco_encode(X):
    
    '''defined prep_telco_encode(X) here but you can also call the function 
    using the prepart.py file. the function takes each column encodes the value 
    as a number representation and assigns it back to the column.'''
    
    X['gender']=LabelEncoder().fit_transform(X['gender'])
    X['partner']=LabelEncoder().fit_transform(X['partner'])
    X['dependents']=LabelEncoder().fit_transform(X['dependents'])
    X['phone_service']=LabelEncoder().fit_transform(X['phone_service'])
    X['multiple_lines']=LabelEncoder().fit_transform(X['multiple_lines'])
    X['online_security']=LabelEncoder().fit_transform(X['online_security'])
    X['online_backup']=LabelEncoder().fit_transform(X['online_backup'])
    X['device_protection']=LabelEncoder().fit_transform(X['device_protection'])
    X['streaming_tv']=LabelEncoder().fit_transform(X['streaming_tv'])
    X['streaming_movies']=LabelEncoder().fit_transform(X['streaming_movies'])
    X['paperless_billing']=LabelEncoder().fit_transform(X['paperless_billing'])
    X['tech_support']=LabelEncoder().fit_transform(X['tech_support'])
    X['churn']=LabelEncoder().fit_transform(X['churn'])
    
    #converted monthly_charges to an int datatype to make the data more uniform
    X['monthly_charges'] = X['monthly_charges'].astype('int')
    
    #converted total_charges from an object to a number. 
    #if any errors, errors='coerce', is used so any value
    #that can't be converted to a number will be set to NaN(Not a Number)
    #Got the median of total_charges, median = X['total_charges'].median(), 
    #The median will be used to fill any NaN values using 
    #X['total_charges'].fillna(median,inplace=True)

    X['total_charges'] = pd.to_numeric(X['total_charges'],errors='coerce')
    median = X['total_charges'].median()
    X['total_charges'].fillna(median,inplace=True)
    
    #dropped uneeded columns other columns represent the same values
    X = X.drop(columns=['contract_type', 'internet_service_type', 'customer_id'])


    
    #once all of the above is finished the results will 
    #be store in the X variable 
    return X



def prep_telco(df):
    df = df.drop(columns=['customer_id', 'contract_type_id', 'payment_type_id', 'internet_service_type_id'])
    dummy_df = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service',
       'multiple_lines', 'online_security', 'online_backup',
       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
       'paperless_billing', 'total_charges', 'churn', 'contract_type',
       'internet_service_type', 'payment_type']], 
                              prefix=['gender', 'partner', 'dependents', 'phone_service',
       'multiple_lines', 'online_security', 'online_backup',
       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
       'paperless_billing', 'total_charges', 'churn', 'contract_type',
       'internet_service_type', 'payment_type'])
    
    df = pd.concat([df, dummy_df.drop(columns=['gender_Female','partner_No','dependents_No','internet_service_type_None','churn_No','paperless_billing_No','streaming_movies_No','streaming_tv_No','tech_support_No','phone_service_No','device_protection_No','online_security_No','online_backup_No','multiple_lines_No','multiple_lines_No phone service','online_security_No internet service'])], axis=1)
   

    df = df.drop(columns=['gender', 'partner', 'dependents', 'phone_service',
       'multiple_lines', 'online_security', 'online_backup',
       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
       'paperless_billing', 'total_charges', 'churn', 'contract_type',
       'internet_service_type', 'payment_type'])
    df = df.drop_duplicates()
    df.columns = [col.lower()for col in df]
    return df


###### Train Validate Test Split Functions #####

def split_titanic_data(df_titanic):
    '''
     split titanic data will split data based on 
    the values present in a cleaned version of titanic
    that is from clean_titanic
    '''
    
    train_val, test = train_test_split(df_titanic, train_size=0.8, random_state=1349, stratify=df_titanic['survived'])
    
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349, stratify=df_titanic[' survived'])
    
    return train, validate, test

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test                        


def telco_train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test     

    #pass in original df as the aregument


######### useful code references below ###### 


######## loops #######

#for loop creating a list of models 
#stored in model_list= []
#dicitionary model_accuracies = {} 
#as the dictionary to hold the model scores
model_accuracies = {}
model_list = []

# for i in range(1,10):
#     nknn = KNeighborsClassifier(n_neighbors = i)
#     nknn.fit(X_train, y_train)
#     model_list.append(nknn)
#     model_accuracies[f'{i} - Neighbors'] = {'Train Score:':round(nknn.score(X_train, y_train),2),
#                                            'Validate Score:':round(nknn.score(X_validate, y_validate),2)}
    

#     # I perhaps would do the same thing I did before with a loop!
# rf_model_dict = {}
# for i in [pair for pair in zip(range(1,10), 
#                                 range(10,1,-1))]:
#     clf = RandomForestClassifier(min_samples_leaf=i[0],
#                                  max_depth=i[1])
#     clf.fit(X_train, y_train)
#     # make a dictionary inside of my model_dictionary
#     rf_model_dict[f'rf_{i[0]}'] =  {}
#     # in the sub-dictionary:
#     # assign the model object
#     rf_model_dict[f'rf_{i[0]}']['model'] = clf
#     #assign the train score
#     rf_model_dict[f'rf_{i[0]}']['train_score'] = \
#     clf.score(X_train, y_train)
#     # assign the validate score
#     rf_model_dict[f'rf_{i[0]}']['val_score'] = \
#     clf.score(X_val, y_val)
#     # assign the validation dropoff
#     rf_model_dict[f'rf_{i[0]}']['val_diff'] = \
#     clf.score(X_train, y_train) - clf.score(X_val, y_val)
# #-------------
#     [rf_model_dict[model]['train_score'] 
#         for model in rf_model_dict]

#     for model in rf_model_dict:
#         rf_model_dict[model]['train_accuracy']
        
# #------------     
#         accuracy_df = pd.DataFrame(
#  {
#      'model':[model for model in rf_model_dict],
#      'train_accuracy':[rf_model_dict[model]['train_score'] for model in rf_model_dict],
#      'val_accuracy': [rf_model_dict[model]['val_score'] for model in rf_model_dict],
#      'diff': [rf_model_dict[model]['val_diff'] for model in rf_model_dict]
#  }
# )


# ##### Chi 2 squared #####
# # Let's run a chi squared to compare proportions, to have more confidence
# alpha = 0.05
# null_hypothesis = "survival and class of ticket are independent"
# alternative_hypothesis = "there is a relationship between class of ticket and survival"
# # Setup a crosstab of observed survival to pclass
# observed = pd.crosstab(train.survived, train.pclass)

# chi2, p, degf, expected = stats.chi2_contingency(observed)

# if p < alpha:
#     print("Reject the null hypothesis that", null_hypothesis)
#     print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
# else:
#     print("Fail to reject the null")
#     print("Insufficient evidence to reject the null")
# p


# #### Check out distributions of numeric columns. ######
# num_cols = df.columns[[df[col].dtype == 'int64' for col in df.columns]]
# for col in num_cols:
#     plt.hist(df[col])
#     plt.title(col)
#     plt.show()


# ###### Use .describe with object columns. ######

# obj_cols = df.columns[[df[col].dtype == 'O' for col in df.columns]]
# for col in obj_cols:
#     print(df[col].value_counts())
#     print(df[col].value_counts(normalize=True, dropna=False))
#     print('----------------------')

###### Create bins for fare using .value_counts.#####
# # Using sort = false will sort by bin values as opposed to the frequency counts.
# df.fare.value_counts(bins=5, sort=False)


# ######Find columns with missing values and the total of missing values.#####
# missing = df.isnull().sum()
# missing[missing > 0]

# ###### Drop duplicates...run just in case; reassign and check the shape of my data.#####
# df = df.drop_duplicates()
# df.shape

# # Drop columns with too many missing values for now and reassign; check the shape of my data.
# cols_to_drop = ['deck', 'embarked', 'class', 'age']
# df = df.drop(columns=cols_to_drop)
# df.shape

# # Run .fillna() on the entire df.
# df['embark_town'] = df.embark_town.fillna(value='Southampton')

# # Using drop_first leaves sex_male, embark_town_Queenstown, and embark_town_Southampton.
# dummy_df = pd.get_dummies(df[['sex','embark_town']], dummy_na=False, drop_first=[True, True])
# dummy_df.head()

# ##verify null values###
# df.isna().sum()
# # Useful helper for checking for nulls
# # What proportion of each column is empty?
# df.isna().mean()

# ##confusion matrix ###
# confusion_matrix(df.actual, df.prediction,
#                  labels = ('no coffee', 'coffee'))
# ##crosstab##
# pd.crosstab(df.actual, df.prediction)
# Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
#df = df.replace(r'^\s*$', np.nan, regex=True)
