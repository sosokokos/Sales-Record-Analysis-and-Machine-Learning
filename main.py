import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import statsmodels.api as sm


from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from clean_data import cleanAndSortData
from generate_plots import printDaysToShipPlot, categoriesBoxPlot, generateHeatmap, checkStats, checkVariances, checkDistributionsGraphs, PlotUnitPrice , PlotTotalProfit, UnitProfitPlot


# Set the Pandas display option to show all columns

def inspect_transformed_data(model, X):
    # Access the preprocessor step
    preprocessor = model.named_steps['preprocessor']

    # Transform the dataset using the preprocessor
    X_transformed = preprocessor.transform(X)

    # Get the feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # Convert the transformed data to a DataFrame for inspection
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    return transformed_df

def build_pipeline():
    # One-hot encoding transformer for nominal categorical features
    one_hot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', one_hot_transformer, ['Item Type', 'Region'])
        ],
        remainder='passthrough'  # Keep all other columns as they are
    )


    # Build the pipeline with a RandomForestClassifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=1, random_state=42))
    ])

    return model


def main(input):
    #Step 2.1: ETL
    dataFrame_raw = pd.read_csv(input)
    cleanAndSortData(dataFrame_raw, "SalesRecordClean.csv")

    #Step 2.2: Load Clean Data
    data_clean = pd.read_csv("SalesRecordClean.csv")

    #Step 2.3: Convert columns to datetime objects
    data_clean['Order Date'] = pd.to_datetime(data_clean['Order Date'])
    data_clean['Ship Date'] = pd.to_datetime(data_clean['Ship Date'])


    #STEP 1.1: Generate plot to see distributions in "Days to Ship" on the whole dataset
    # printDaysToShipPlot(data_clean)

    #STEP 1.2: Generate plot to see distribution of "Days to Ship" in Canada
    data_clean_canada = data_clean[data_clean['Country'] == "Canada"]
    # printDaysToShipPlot(data_clean_canada)

    #STEP 1.3: Generate plot to see distribution of "Days to Ship" in Canada in 2015
    data_clean_canada_2015 = data_clean[
             (data_clean['Country'] == "Canada") &
             (pd.to_datetime(data_clean['Order Date']).dt.year == 2015)
    ]
    # printDaysToShipPlot(data_clean_canada_2015)

    #STEP 1.4: Generate plot to see distribution of "Days to Ship" in Canada in January of 2015
    data_clean_canada_2015_jan = data_clean[
             (data_clean['Country'] == "Canada") &
             (pd.to_datetime(data_clean['Order Date']).dt.year == 2015) &
             (pd.to_datetime(data_clean['Order Date']).dt.month == 1) 
    ]
    # printDaysToShipPlot(data_clean_canada_2015_jan)

    #Using only data from 2015 in Canada
    data = data_clean

    #STEP 1.5: Generate boxplot of categories with low, medium, high, critical priority 
    # categoriesBoxPlot(data)

    #STEP 1.6: Generating Correlation Heat Map 
    # generateHeatmap(data)

    #STEP 1.7: Check DataFrame's count, mean, standard deviation, min, max using .describe() function
    # checkStats(data)

    #STEP 1.8 Check the Distributions by graphing individual columns
    # checkDistributionsGraphs(data)

    #STEP 1.9 Check the Variances of individual columns
    # checkVariances(data)

    #STEP 1.10 COMPARE UNIT PRICE WITH PRODUCTS THAT ARE SOLD WITH HIGHER UNIT COST (P VALUE COMPARE MEANS) #TODO EDIT

    #STEP 1.11: Plot Unit PriceExtract from the dataset
 
    # PlotUnitPrice(data["Unit Price"])

    #STEP 1.12 Extract the 'Total Profit' column from the dataset
    # PlotTotalProfit(data["Total Profit"])

    #STEP 1.13 Plot Scatter plot with Lienar regression line 
    # UnitProfitPlot(data)

    #STEP 1.14 OLS 
    # # Define the independent variable (Unit Price) and dependent variable (Total Revenue)
    # X = data_clean['Unit Price']  # Predictor
    # y = data_clean['Total Profit']  # Outcome

    # # Add a constant term to the predictor (intercept)
    # X = sm.add_constant(X)
    # # Fit the regression model using OLS
    # model = sm.OLS(y, X).fit()
    # # Print the regression summary
    # print(model.summary())


    ###### Random Forest #######
    
    # Label encoding the country as well because too much dimensionality for one hot encoding 
    # and random forest is okay with 
    data['Country'] = LabelEncoder().fit_transform(data['Country'])

    #STEP 3.1: Feature Selection
    X = data[['Sales Channel', 'Order Priority', 'Units Sold', "Item Type","Region",'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']]
    y = data['Days to Ship']

    #STEP 3.2: Train/Test Split (70,30) respectively
    #stratify = 1 to keep the same distribution of classes in the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


    #STEP 4.1: Model Selection
    model = build_pipeline()


    # model = make_pipeline ( #Best performance of them all but still worse than guessing max_depth=100        n_estimators=100, max_depth=50, min_samples_leaf=15, min_samples_split=10 , random_state=1 , n_jobs=-1
    #     #RandomForestClassifier(n_estimators=300, n_jobs=-1) 
    #     RandomForestClassifier(n_estimators=100, n_jobs=-1) 
    # )

    # model0 = DecisionTreeClassifier()

    # model1 = KNeighborsClassifier()
   
    
    # model1 = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)



    #STEP 4.2: Model training
    model.fit(X_train, y_train)


    transformed_df = inspect_transformed_data(model, X_train)
    transformed_df.to_csv("transformed_data.csv", index=False)

    #STEP 5.1: Analysis of results 
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    #Step 5.2: Verifying the output of the preduction
    #print(model.predict(X_test))


    #print(classification_report(y_test, model.predict(X_test))) TODO EDIT



if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)


