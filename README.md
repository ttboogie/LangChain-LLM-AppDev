# LangChain-LLM-AppDev

# Data Analysis Tool using LangChain

This project provides a comprehensive data analysis tool leveraging LangChain, OpenAI, and various Python libraries. The tool enables users to load data, clean it, and perform various analysis techniques based on user input.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Prompt Templates](#prompt-templates)
  - [Sequential Chains](#sequential-chains)
  - [Evaluating Chain Outputs](#evaluating-chain-outputs)
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

### Prompt Templates

Prompt templates in LangChain are used to dynamically create prompts based on user input. In this project, we define two main prompt templates:

1. **Intent and Analysis Prompt:**
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

2. **Columns Prompt:**
   ```python
   columns_prompt = ChatPromptTemplate.from_template(
       """
       User wants to analyze the data with the following columns: {user_columns}.
       Given the user's explanation: {user_explanation}, identify the most relevant columns for the analysis.
       Provide a list of the selected columns in the format of a comma-separated list.
       """
   )
   ```

### Sequential Chains

Sequential chains allow us to link multiple chains together. In this project, we use a sequential chain to combine the intent and analysis chain with the columns chain:

```python
overall_chain = SequentialChain(
    chains=[intent_and_analysis_chain, columns_chain],
    input_variables=["user_explanation", "user_columns", "df", "analysis_options"],
    output_variables=["selected_analysis_types", "selected_columns"],
    verbose=True
)
```

### Evaluating Chain Outputs

After obtaining the output from the chains, we parse the selected columns and analysis types:

```python
# Parse selected columns and analysis types correctly
selected_columns = [col.strip().strip("'") for col in result['selected_columns'].replace(' and ', ', ').replace('columns:', '').split(',')]
selected_analysis_types = [analysis.strip() for analysis in result['selected_analysis_types'].split(', ')]
```

Then, we perform multiple analyses based on the selected columns and analysis types:

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
