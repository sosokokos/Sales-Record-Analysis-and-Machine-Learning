import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns

def printDaysToShipPlot(dataFrame):
    seaborn.set()
    plt.hist(dataFrame['Days to Ship'], bins = 51)
    plt.xlabel("Days to Ship")
    plt.ylabel("Number of Occurances")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def categoriesBoxPlot(dataFrame):
    seaborn.boxplot(x='Order Priority', y='Days to Ship', data=dataFrame)
    plt.title('Boxplot for Order Priority X Days to Ship')
    plt.xlabel('Order Prioriy')
    plt.ylabel('Days to Ship')
    plt.show()

def generateHeatmap(dataInput):
    dataFrame = dataInput
    
    del dataFrame['Country']
    del dataFrame['Region']
    del dataFrame['Item Type']

    correlation_matrix = dataFrame.corr()
    
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def checkVariances(dataInput):
    dataFrame = dataInput
    columns_to_keep = [col for col in dataInput.columns if col not in ['Country', 'Region', 'Item Type', 'Order Date', 'Ship Date']]
    dataFrame = dataInput[columns_to_keep]
    
    variances = dataFrame.var()
    print("Variance of each variable:")
    print(variances)

def checkDistributionsGraphs(dataFrame):
    for column in dataFrame.columns:
        if dataFrame[column].dtype in ['int64', 'float64']:
            plt.figure()
            dataFrame[column].hist(bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()


def checkStats(df_input):
    print(df_input[[ 'Units Sold', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit', 'Days to Ship']].describe())

def PlotUnitPrice(unit_price):
    plt.figure(figsize=(8, 6))
    plt.hist(unit_price, bins=51, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Unit Price")
    plt.xlabel("Unit Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def PlotTotalProfit(total_profit):
    plt.figure(figsize=(8, 6))
    plt.hist(total_profit, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Total Profit")
    plt.xlabel("Total Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def UnitProfitPlot(data_clean):
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=data_clean['Unit Price'],  
        y=data_clean['Total Profit'], 
        scatter_kws={'alpha': 0.5, 'color': 'blue'},  
        line_kws={'color': 'red'}
    )

    plt.title("Scatter Plot with Linear Fit: Unit Price vs Total Profit", fontsize=14)
    plt.xlabel("Unit Price", fontsize=12)
    plt.ylabel("Total Profit", fontsize=12)

    plt.tight_layout()
    plt.show()
