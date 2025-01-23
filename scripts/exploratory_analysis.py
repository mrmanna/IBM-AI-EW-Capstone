import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def summary_statistics(df):
    try:
        summary = df.describe(include='all')
        logging.info("Summary statistics generated successfully")
        return summary
    except Exception as e:
        logging.error(f"Error generating summary statistics: {e}")
        return None

def plot_missing_values(df):
    try:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/missing_values_heatmap.png')
        plt.show()
        logging.info("Missing values heatmap saved successfully")
    except Exception as e:
        logging.error(f"Error plotting missing values heatmap: {e}")

def plot_distribution(df, column):
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], bins=50, kde=True)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'visualizations/{column}_distribution.png')
        plt.show()
        logging.info(f"{column} distribution plot saved successfully")
    except Exception as e:
        logging.error(f"Error plotting {column} distribution: {e}")

if __name__ == "__main__":
    filepath = '../data/cs-train/ts-data/ts-all.csv'
    df = load_data(filepath)

    if df is not None:
        # Generate summary statistics
        summary = summary_statistics(df)
        print(summary)

        # Plot missing values heatmap
        plot_missing_values(df)

        # Plot distributions for relevant columns
        columns_to_plot = ['purchases', 'unique_invoices', 'unique_streams', 'total_views', 'revenue']
        for column in columns_to_plot:
            plot_distribution(df, column)