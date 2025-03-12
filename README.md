# Sales Record Analysis and Machine Learning
## **Overview** 
This project processes and analyzes sales data to generate insights and apply machine learning models to predict shipping times. The project involves cleaning data, exploratory data analysis, statistical testing, and machine learning model training.

---

## Table of Contents 
- Setup
- Structure
- Files Produced
- Running the Code
- Data Exploration  [Step 1.1 - 1.10]  
- ETL [Step 2]
- Feature Selection  [Step 3]
- Model Training  [Step 4]
- Analysis of Results [Step 5]

---

# Setup #
## Required Libraries ##
Make sure you have Python installed along with the following libraries:
- `matplotlib`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`

## Structure ##
- All of the functions are being run in main.py while their implementations are in clean_data.py and generate_plots.py 

## Files Produced #
- At the begining of main.py we are running function cleanAndSortData(dataFrame_raw, output_file_name) that creates SalesRecordClean.csv
which is a clean dataframe that is used for the data analysis, to learn about cleanAndSortData() look at [ETL / Step 2]


## Running the Code ##
Run The code using following command:
```
python3 main.py [INPUT_PATH]
```
Which in our case looked like:
```
python3 main.py SalesRecords/SalesRecords_5m.csv
```

# **Data Exploration - STEP 1** #
# STEP 1.1: 
    - Generate plot to see distributions in "Days to Ship" on the whole dataset

# STEP 1.2: 
    - Generate plot to see distribution of "Days to Ship" in Canada

# STEP 1.3: 
    - Generate plot to see distribution of "Days to Ship" in Canada in 2015

# STEP 1.4: 
    - Generate plot to see distribution of "Days to Ship" in Canada in January of 2015

# STEP 1.5: 
    - Generate boxplot of categories with low, medium, high, critical priority 

# STEP 1.6: 
    - Generating Correlation Heat Map 

# STEP 1.7: 
    - Check DataFrame's count, mean, standard deviation, min, max using .describe() function

# STEP 1.8: 
    - Check the Distributions by graphing individual columns

# STEP 1.9: 
    - Check the Variances of individual columns

# STEP 1.10: 
    - Check the Variances of individual columns

# **Extraxt-Transform-Load - STEP 2** #
# STEP 2: 
    Extract
        - Reading Data
    Transform
        - Fix Data types, Reshape Data, Aggregating Data, Filtering out data you don't care abt, joining multiple data, 
    Load
        - Once the data is transformed we want to save it to csv/json


# **Feature Selection - STEP 3** #
# STEP 3.1: 
    - Selecting appropreate features (columns) for the model training

# STEP 3.2: 
    - Train/Test Split (70,30) respectively


# **Model Training - STEP 4** #
# STEP 4.1: 
    - Model selection

# STEP 4.2: 
    - Model training


# **Analysis of Results - STEP 5** #
# STEP 5.1: 
    - Analysis of results

# STEP 5.2: 
    - Verifying the output of prediction



    

    




