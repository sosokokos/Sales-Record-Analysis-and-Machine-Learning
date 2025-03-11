import pandas as pd

ordinal_mapping = {
    "L": 1,
    "M": 2,
    "H": 3,
    "C": 4,
}
one_hot_mapping = {
    "Online": 1,
    "Offline": 0,
}

def ordinal_encode(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df

def one_hot_encode(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df

def checkForNA(df_input):
    initial_shape = df_input.shape
    df_cleaned = df_input.dropna()
    final_shape = df_cleaned.shape

    if initial_shape[0] > final_shape[0]:
        print(f"Rows were dropped. {initial_shape[0] - final_shape[0]} rows removed.")

    return df_cleaned.reset_index()

def cleanAndSortData(df_input, output_name):
    df_input = df_input.dropna()
    df_cleaned = df_input.drop_duplicates()
    df_cleaned = df_cleaned.drop('Order ID', axis=1)
    df_cleaned.reset_index(drop=True, inplace=True)

    df_encoded = ordinal_encode(df_cleaned, "Order Priority", ordinal_mapping)
    df_encoded = one_hot_encode(df_encoded, "Sales Channel", one_hot_mapping)

    df_encoded['Order Date'] = pd.to_datetime(df_encoded['Order Date'])
    df_encoded['Ship Date'] = pd.to_datetime(df_encoded['Ship Date'])

    data_sorted = df_encoded.sort_values(by=["Country", "Order Date"])

    data_sorted['Days to Ship'] = (data_sorted['Ship Date'] - data_sorted['Order Date']).dt.days

    data_sorted.to_csv(output_name, index=False)