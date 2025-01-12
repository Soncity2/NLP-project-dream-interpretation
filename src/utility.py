import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to rename columns in a DataFrame
def rename_columns(data, column_mapping):
    """
    Rename columns in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame whose columns need to be renamed.
        column_mapping (dict): A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: A DataFrame with renamed columns.
    """
    # Check if all specified old columns exist in the DataFrame
    missing_columns = [col for col in column_mapping.keys() if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following columns are not present in the DataFrame: {missing_columns}")

    # Rename columns using the provided mapping
    data = data.rename(columns=column_mapping)

    return data

def load_data(file_path, index_column=None):

    # Get the file extension to determine how to load the file
    file_extension = file_path.split('.')[-1].lower()

    # Load the file based on its extension
    if file_extension == 'csv':
        data = pd.read_csv(file_path)  # Load CSV file
    elif file_extension == 'tsv':
        data = pd.read_csv(file_path, sep='\t')  # Load TSV file with tab delimiter
    elif file_extension == 'json':
        data = pd.read_json(file_path, encoding="utf-8")  # Load JSON file
    else:
        raise ValueError("Unsupported file type. Please provide a CSV, TSV, or JSON file.")

    # Set the index column if specified
    if index_column:
        data = rename_columns(data, {index_column : "ID"})
        index_column = "ID"
        if index_column in data.columns:
            data.set_index(index_column, inplace=True)
        else:
            raise ValueError(f"The specified index column '{index_column}' does not exist in the data.")

    return data

class GenericEDA:
    """
    A class to perform basic Exploratory Data Analysis (EDA) on any dataset.

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset to analyze.
    """

    def __init__(self, df: pd.DataFrame, dtypes=None):

        self.df = df.copy()  # Use a copy to avoid modifying the original data

        # Set column data types if specified
        if dtypes:
            self.df = self.df.astype(dtypes)

    def show_info(self):
        """Display basic information about the dataset."""
        print("Dataset Information:")
        print(self.df.info())
        print("\n")


    def show_summary_statistics(self):
        """Display summary statistics for numerical and categorical columns."""
        print("Summary Statistics (Numerical):")
        print(self.df.describe())
        print("\n")
        print("Summary Statistics (Categorical):")
        print(self.df.describe(include=["object", "category"]))
        print("\n")

    def check_missing_values(self):
        """Display the number of missing values in each column."""
        missing_values = self.df.isnull().sum()
        print("Missing Values in Each Column:")
        print(missing_values[missing_values > 0])
        print("\n")

    def show_unique_values(self):
        """Display the unique values and their counts for each column."""
        for col in self.df.columns:
            print(f"Unique values in '{col}':")
            print(self.df[col].value_counts())
            print("\n" + "=" * 40 + "\n")

    def categorical_frequency(self, filter_columns=None, long_columns=None):
        """
        Plot the frequency distribution of all categorical columns, with specific filtering
        for selected columns where categories with counts less than 10 are combined into 'Other'.

        Parameters:
        filter_columns (list of str): List of column names to apply filtering. If None, no filtering is applied.
        """
        # Get all categorical columns
        all_categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        
        # Count plots for all categorical columns
        for col in all_categorical_columns:
            if filter_columns and col in filter_columns:
                # Replace categories with count less than 30 with 'Other' for specified columns, if has a wide variety of categories, count less than 100
                counts = self.df[col].value_counts()
                if col in long_columns:
                    small_counts = counts[counts < 100].index
                else:
                    small_counts = counts[counts < 30].index
                self.df[col] = self.df[col].replace(small_counts, 'Other')
                
                # Filter out the 'Other' category for the specified columns
                filtered_df = self.df[self.df[col] != 'Other']
            
                # Plot the frequency distribution
                plt.figure(figsize=(8, 6))
                sns.countplot(y=col, data=filtered_df, order=filtered_df[col].value_counts().index)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.show()

    def plot_numerical_distribution(self):
        """
        Plot the distribution of all numerical columns

        Parameters:
        -----------
        """
        all_numerical_columns = self.df.select_dtypes(include=["int64", "float64"]).columns
        for column in all_numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.df[column], kde=True)
            plt.title(f"Distribution of {column}")
            plt.show()


    def plot_pairwise_distributions(self):
        """Plot pairwise distributions of numerical columns using a pairplot."""
        sns.pairplot(self.df)
        plt.show()

    def plot_correlation_matrix(self):
        """Plot a heatmap of the correlation matrix for numerical columns."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def detect_outliers(self, column):
        """
        Display a boxplot to visualize outliers in a numerical column.

        Parameters:
        -----------
        column : str
            The column name to check for outliers.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column} (Outliers Detection)")
        plt.show()

    def feature_correlation_with_target(self, target_column):
        """
        Show the correlation of each feature with the target variable.

        Parameters:
        -----------
        target_column : str
            The target variable to correlate with.
        """
        correlations = self.df.corr()[target_column].sort_values(ascending=False)
        print(f"Correlations with {target_column}:")
        print(correlations)
        print("\n")

    def plot_numeric_over_time(self, num_column, time_column):
        """
        Plot a line chart to observe how a numerical column changes over time.

        Parameters:
        -----------
        num_column : str
            The numerical column to plot.
        time_column : str
            The time column to plot along the x-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=self.df[time_column], y=self.df[num_column])
        plt.title(f"{num_column} Over Time")
        plt.xlabel(time_column)
        plt.ylabel(num_column)
        plt.xticks(rotation=45)
        plt.show()
    
    def clean_nan(self, fill_value=None):
        """
        Clean a specific column in the dataset by:
        - Removing rows where more than 70% of values in the specified column are missing.
        - Filling missing values in the remaining rows of the specified column with the given fill value.

        Parameters:
        -----------
        column : str
            The column to clean for NaN values.
        fill_value : Any
            The value used to fill missing data in rows with less than 30% missing.

        Conclusion:
        -----------
        This method allows column-specific cleaning of NaN values,
        helping tailor the cleaning process to each column's characteristics.
        """
        # Get column names that have null values
        columns_with_nulls = self.df.columns[self.df.isnull().any()].tolist()

        # Select only the columns with null values
        df_with_nulls = self.df[columns_with_nulls]
        for column in df_with_nulls:
            # Calculate the percentage of missing values in the specified column
            missing_percent_column = self.df[column].isnull().mean()

            if missing_percent_column > 0.7:
                # Remove rows where more than 70% of values in the specified column are missing
                self.df = self.df[self.df[column].notna()]
                print(f"Rows with more than 70% missing in '{column}' have been removed.\n")
            else:
                # Fill remaining NaN values in the specified column with the given fill_value
                fill_value = self.df[column].mode()[0] if fill_value is None else fill_value
                self.df[column] = self.df[column].fillna(fill_value)
                print(f"Missing values in '{column}' have been filled with {fill_value}.\n")
