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
    return df[columns].describe()

def correlation_analysis(df, columns):
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
    categorical_cols = [col for col in columns if not np.issubdtype(df[col].dtype, np.number)]
    if len(categorical_cols) == 0:
        return "Not enough categorical columns to count values."
    counts = {}
    for col in categorical_cols:
        counts[col] = df[col].value_counts().to_dict()
    return counts

def group_by_analysis(df, group_by_column, analysis_column):
    return df.groupby(group_by_column)[analysis_column].mean()

def time_series_analysis(df, date_column, analysis_column):
    df.set_index(date_column, inplace=True)
    return df.resample('YE')[analysis_column].count()

def create_pivot_table(df, index, columns, values, aggfunc='mean'):
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)

def plot_scatter(df, column_x, column_y):
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

# Chain for understanding user's intent and selecting relevant analysis techniques
intent_and_analysis_prompt = ChatPromptTemplate.from_template(
    """
    The user wants to analyze the data. Here is their explanation: {user_explanation}
    Based on this explanation, select all the relevant analysis techniques from the following options:
    {analysis_options}.
    Provide a list of the selected analysis techniques in the format of a comma-separated list.
    """
)
intent_and_analysis_chain = LLMChain(llm=llm, prompt=intent_and_analysis_prompt, output_key="selected_analysis_types")

# Chain for identifying relevant columns
columns_prompt = ChatPromptTemplate.from_template(
    """
    User wants to analyze the data with the following columns: {user_columns}.
    Given the user's explanation: {user_explanation}, identify the most relevant columns for the analysis.
    Provide a list of the selected columns in the format of a comma-separated list.
    """
)
columns_chain = LLMChain(llm=llm, prompt=columns_prompt, output_key="selected_columns")

# Overall sequential chain
overall_chain = SequentialChain(
    chains=[intent_and_analysis_chain, columns_chain],
    input_variables=["user_explanation", "user_columns", "df", "analysis_options"],
    output_variables=["selected_analysis_types", "selected_columns"],
    verbose=True
)

# Perform Analysis Function
def perform_multiple_analyses(df, columns, analysis_types):
    results = {}
    for analysis_type in analysis_types:
        if analysis_type == "summary_statistics":
            results[analysis_type] = summary_statistics(df, columns)
        elif analysis_type == "correlation_analysis":
            results[analysis_type] = correlation_analysis(df, columns)
        elif analysis_type == "plot_histograms":
            results[analysis_type] = plot_histograms(df, columns)
        elif analysis_type == "plot_boxplots":
            results[analysis_type] = plot_boxplots(df, columns)
        elif analysis_type == "value_counts":
            results[analysis_type] = value_counts(df, columns)
        elif analysis_type == "group_by_analysis":
            results[analysis_type] = group_by_analysis(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for group_by_analysis."
        elif analysis_type == "time_series_analysis":
            results[analysis_type] = time_series_analysis(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for time_series_analysis."
        elif analysis_type == "create_pivot_table":
            results[analysis_type] = create_pivot_table(df, columns[0], columns[1], columns[2]) if len(columns) > 2 else "Not enough columns for create_pivot_table."
        elif analysis_type == "plot_scatter":
            results[analysis_type] = plot_scatter(df, columns[0], columns[1]) if len(columns) > 1 else "Not enough columns for plot_scatter."
    return results

# Main program
if __name__ == "__main__":
    # Get the file path from the user
    file_path = input("Enter the path to the data file (CSV, Excel, JSON): ").strip()
    df = load_and_clean_data(file_path)
    print("Cleaned Data:")
    print(df.head())

    while True:
        # Get the user's explanation and columns
        user_explanation = input("Please explain what you want to analyze: ").strip()
        user_columns = input("Please enter the columns you want to analyze, separated by commas: ").strip()
        columns = [col.strip() for col in user_columns.split(',')]

        # Add the analysis options to the chain input
        chain_input = {
            "user_explanation": user_explanation,
            "user_columns": columns,
            "df": df,
            "analysis_options": ', '.join(analysis_types)
        }

        # Perform the analysis
        result = overall_chain(chain_input)

        # Parse selected columns and analysis types correctly
        selected_columns = [col.strip().strip("'") for col in result['selected_columns'].replace(' and ', ', ').replace('columns:', '').split(',')]
        selected_analysis_types = [analysis.strip() for analysis in result['selected_analysis_types'].split(', ')]
        
        # Debugging prints
        print(f"Selected Columns: {selected_columns}")
        print(f"Selected Analysis Types: {selected_analysis_types}")
        
        # Perform multiple analyses
        analysis_results = perform_multiple_analyses(df, selected_columns, selected_analysis_types)

        # Display the results
        for analysis_type, analysis_result in analysis_results.items():
            print(f"Analysis Technique: {analysis_type}")
            print(analysis_result)
