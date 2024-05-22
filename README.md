# LangChain-LLM-AppDev

Absolutely, here's a more detailed version of the `README.md` that includes additional explanations for the various code pieces, setup instructions, and running the script with specific file types.

```markdown
# Data Analysis Tool using LangChain

This project provides a comprehensive data analysis tool leveraging LangChain, OpenAI, and various Python libraries. The tool enables users to load data, clean it, and perform various analysis techniques based on user input.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Prompt Templates](#prompt-templates)
  - [Sequential Chains](#sequential-chains)
  - [Evaluating Chain Outputs](#evaluating-chain-outputs)
  - [Performing Analyses](#performing-analyses)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the root directory and add your OpenAI API key:**
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Setting Up the Environment

Ensure you have the following packages installed:

- `openai`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `dotenv`
- `langchain`
- `langchain_community`

### Running the Script

The script is designed to work with data files in CSV, Excel, or JSON format. Here’s how to run the script:

1. **Prepare your data file** (CSV, Excel, or JSON) and place it in an accessible directory.

2. **Run the main script**:
   ```bash
   python main.py
   ```

3. **Provide the required inputs** when prompted:
   - Path to the data file.
   - Explanation of what you want to analyze.
   - Columns you want to include in the analysis (comma-separated).

### Code Explanation

#### Loading and Cleaning Data

The `load_and_clean_data` function handles loading data from different file formats and cleaning it by removing missing values, duplicates, and extra whitespace from column names.

```python
def load_and_clean_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()
    return df
```

#### Data Analysis Functions

Several functions perform specific analyses on the data:

- **Summary Statistics**: Provides descriptive statistics for specified columns.
- **Correlation Analysis**: Computes and visualizes correlation between numeric columns.
- **Plot Histograms and Boxplots**: Generates histograms and boxplots for numeric columns.
- **Value Counts**: Counts occurrences of unique values in categorical columns.
- **Group By Analysis**: Groups data by a column and computes the mean of another column.
- **Time Series Analysis**: Analyzes data by resampling based on a date column.
- **Pivot Table Creation**: Creates pivot tables for specified columns.
- **Scatter Plot**: Plots scatter plots between two numeric columns.

```python
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
```

#### Prompt Templates

Prompt templates in LangChain are used to dynamically create prompts based on user input. In this project, we define two main prompt templates:

1. **Intent and Analysis Prompt**:
   ```python
   intent_and_analysis_prompt = ChatPromptTemplate.from_template(
       """
       The user wants to analyze the data. Here is their explanation: {user_explanation}
       Based on this explanation, select all the relevant analysis techniques from the following options:
       {analysis_options}.
       Provide a list of the selected analysis techniques in the format of a comma-separated list.
       """
   )
   ```

2. **Columns Prompt**:
   ```python
   columns_prompt = ChatPromptTemplate.from_template(
       """
       User wants to analyze the data with the following columns: {user_columns}.
       Given the user's explanation: {user_explanation}, identify the most relevant columns for the analysis.
       Provide a list of the selected columns in the format of a comma-separated list.
       """
   )
   ```

#### Sequential Chains

Sequential chains allow us to link multiple chains together. In this project, we use a sequential chain to combine the intent and analysis chain with the columns chain:

```python
overall_chain = SequentialChain(
    chains=[intent_and_analysis_chain, columns_chain],
    input_variables=["user_explanation", "user_columns", "df", "analysis_options"],
    output_variables=["selected_analysis_types", "selected_columns"],
    verbose=True
)
```

#### Evaluating Chain Outputs

After obtaining the output from the chains, we parse the selected columns and analysis types:

```python
# Parse selected columns and analysis types correctly
selected_columns = [col.strip().strip("'") for col in result['selected_columns'].replace(' and ', ', ').replace('columns:', '').split(',')]
selected_analysis_types = [analysis.strip() for analysis in result['selected_analysis_types'].split(', ')]
```

#### Performing Analyses

We then perform multiple analyses based on the selected columns and analysis types:

```python
# Perform multiple analyses
analysis_results = perform_multiple_analyses(df, selected_columns, selected_analysis_types)

# Display the results
for analysis_type, analysis_result in analysis_results.items():
    print(f"Analysis Technique: {analysis_type}")
    print(analysis_result)
```

## Environment Variables

The project relies on the following environment variables which should be defined in a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key.
