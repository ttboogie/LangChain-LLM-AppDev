import os
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.exceptions import OutputParserException
import json

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Function to load and clean data from various file formats
def load_and_clean_data(file_path):
    """
    Load data from a file and perform basic cleaning.
    Supports CSV, Excel, and JSON formats.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    df.dropna(inplace=True)  # Remove missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    return df

# Define data analysis functions
def summary_statistics(df, columns):
    """Return summary statistics for the selected columns."""
    return df[columns].describe()

def correlation_analysis(df, columns):
    """Perform correlation analysis on the selected columns and save a heatmap."""
    numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
    if len(numeric_cols) < 2:
        return "Not enough numeric columns to compute correlations."
    correlations = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    return correlations

def plot_histograms(df, columns):
    """Plot histograms for the selected numeric columns."""
    numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
    if len(numeric_cols) < 1:
        return "Not enough numeric columns to plot histograms."
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(f'{col}_histogram.png')
        plt.close()
    return "Histograms plotted."

def plot_boxplots(df, columns):
    """Plot boxplots for the selected numeric columns."""
    numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
    if len(numeric_cols) < 1:
        return "Not enough numeric columns to plot boxplots."
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(f'{col}_boxplot.png')
        plt.close()
    return "Boxplots plotted."

def value_counts(df, columns):
    """Return value counts for the selected categorical columns."""
    categorical_cols = [col for col in columns if not np.issubdtype(df[col].dtype, np.number)]
    if len(categorical_cols) == 0:
        return "Not enough categorical columns to count values."
    counts = {}
    for col in categorical_cols:
        counts[col] = df[col].value_counts().to_dict()
    return counts

def group_by_analysis(df, group_by_column, analysis_column):
    """Perform group by analysis and return the mean of the analysis column."""
    return df.groupby(group_by_column)[analysis_column].mean()

def time_series_analysis(df, date_column, analysis_column):
    """Perform time series analysis by resampling the data yearly and counting the analysis column."""
    df.set_index(date_column, inplace=True)
    return df.resample('YE')[analysis_column].count()

def create_pivot_table(df, index, columns, values, aggfunc='mean'):
    """Create a pivot table for the selected columns."""
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)

def plot_scatter(df, column_x, column_y):
    """Plot a scatter plot for the selected numeric columns."""
    if np.issubdtype(df[column_x].dtype, np.number) and np.issubdtype(df[column_y].dtype, np.number):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[column_x], y=df[column_y])
        plt.title(f'Scatter plot of {column_x} vs {column_y}')
        plt.savefig(f'{column_x}_vs_{column_y}_scatter.png')
        plt.close()
    return f'Scatter plot of {column_x} vs {column_y} plotted.'

# List of available analysis types
analysis_types = [
    "summary_statistics",
    "correlation_analysis",
    "plot_histograms",
    "plot_boxplots",
    "value_counts",
    "group_by_analysis",
    "time_series_analysis",
    "create_pivot_table",
    "plot_scatter"
]

# Function to extract valid analysis type
def extract_analysis_type(text):
    """Extract a valid analysis type from the given text."""
    for analysis_type in analysis_types:
        if analysis_type in text:
            return analysis_type
    return "Unknown analysis type"

# Chain for getting the analysis type
column_prompt = ChatPromptTemplate.from_template(
    """
    The selected columns are: {selected_columns}
    Given these columns, what would be the best data analysis techniques to compare these data columns?
    Choose a single value from the following options: {analysis_options} and return it exactly as it is given.
    """
)
chain_one = LLMChain(llm=llm, prompt=column_prompt, output_key="analysis_type")

# Chain for performing the analysis
analysis_prompt = ChatPromptTemplate.from_template(
    "Given the analysis type '{analysis_type}', perform the analysis on the selected columns: {selected_columns}\n"
)
chain_two = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis_result")

# Chain for refining the analysis type and performing the analysis
analyze_prompt = ChatPromptTemplate.from_template(
    """
    The best ways to analyze these columns: {selected_columns} is given by '{analysis_result}'\n
    Given these are the best relevant techniques, choose from the the analysis options and perform one: {analysis_options}
    """
)
chain_three = LLMChain(llm=llm, prompt=analyze_prompt, output_key="refined_analysis_result")

# Perform Analysis Function
def perform_analysis(df, columns, analysis_type):
    """Perform the chosen analysis on the selected columns."""
    if analysis_type == "summary_statistics":
        return summary_statistics(df, columns)
    elif analysis_type == "correlation_analysis":
        return correlation_analysis(df, columns)
    elif analysis_type == "plot_histograms":
        return plot_histograms(df, columns)
    elif analysis_type == "plot_boxplots":
        return plot_boxplots(df, columns)
    elif analysis_type == "value_counts":
        return value_counts(df, columns)
    elif analysis_type == "group_by_analysis":
        return group_by_analysis(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for group_by_analysis."
    elif analysis_type == "time_series_analysis":
        return time_series_analysis(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for time_series_analysis."
    elif analysis_type == "create_pivot_table":
        return create_pivot_table(df, columns[0], columns[1], columns[2]) if len(columns) > 2 else "Not enough columns for create_pivot_table."
    elif analysis_type == "plot_scatter":
        return plot_scatter(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for plot_scatter."
    else:
        return "Unknown analysis type."

class PerformAnalysisChain(LLMChain):
    def _call(self, inputs):
        df = inputs['df']
        columns = inputs['selected_columns']
        analysis_type = extract_analysis_type(inputs['analysis_type'])

        if analysis_type not in analysis_types:
            return {"analysis_result": "Unknown analysis type."}

        result = perform_analysis(df, columns, analysis_type)
        return {"analysis_result": result, "analysis_type": analysis_type}

# Overall sequential chain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three],
    input_variables=["selected_columns", "df", "analysis_options"],
    output_variables=["analysis_type", "analysis_result", "refined_analysis_result"],
    verbose=True
)

# Main program
if __name__ == "__main__":
    # Get the file path from the user
    file_path = input("Enter the path to the data file (CSV, Excel, JSON): ").strip()
    df = load_and_clean_data(file_path)
    print("Cleaned Data:")
    print(df.head())

    while True:
        # Get the columns from the user
        columns_input = input("Please enter the columns you want to analyze, separated by commas, or type 'quit' to exit: ")
        if columns_input.lower() == 'quit':
            break
        columns = [col.strip() for col in columns_input.split(',')]
        
        # Add the analysis options to the chain input
        chain_input = {
            "selected_columns": columns,
            "df": df,
            "analysis_options": ', '.join(analysis_types)
        }

        # Perform the analysis
        result = overall_chain(chain_input)
        perform_analysis(df, columns, result['analysis_type'])

        # Display the result
        print(f"Analysis Technique: {result['analysis_type']}")
        print(result["analysis_result"])
        print(result["refined_analysis_result"])
