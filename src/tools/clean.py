# Functions to clean data

# --------------------------------------------------------------------------------------------------------------------
# Function to encode categorical values
def encode_categorical(df):
    '''This function takes a dataframe and encodes the variables 'cut', 'color' and 'clarity'.
    Returns the dataframe with the corresponding encoded values.
    '''
    # encoding dictionaries:
    dict_cut = {
    "Fair":0,
    "Good":1,
    "Very Good":2,
    "Premium":3,
    "Ideal":4
}
    dict_color = {
        "J":0,
        "I":1,
        "H":2,
        "G":3,
        "F":4,
        "E":5,
        "D":6
    }
    dict_clarity = {
        "I1":0,
        "SI2":1,
        "SI1":2,
        "VS2":3,
        "VS1":4,
        "VVS2":5,
        "VVS1":6,
        "IF":7,
        "FL":8
    }
    df = df.replace({'cut':dict_cut, 'color':dict_color,'clarity':dict_clarity}) # replacing
    return df

# --------------------------------------------------------------------------------------------------------------------
# Function to remove outliers
def remove_outliers(df, col_name):
    '''This function takes a dataframe and removes outliers from a given column.
    Returns the dataframe without outliers.
    '''
    q1 = df[col_name].quantile(0.25) # first quartile
    q3 = df[col_name].quantile(0.75) # third quartile
    iqr = q3-q1 # interquartile range
    fence_low  = q1 - (1.5*iqr) # defining fences
    fence_high = q3 + (1.5*iqr)
    df = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df

