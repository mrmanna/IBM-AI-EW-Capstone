import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import requests
from datetime import datetime, timedelta
from model import model_predict, model_load

def local_make_prediction(country, start_date, days=10):
    predictions = []

    for i in range(days):
        date = start_date + timedelta(days=i)
        result = model_predict(country, date.year, date.month, date.day)
        predictions.append({
            "date": date.strftime("%Y-%m-%d"),
            "revenue": result["y_pred"][0]
        })
    return pd.DataFrame(predictions)

def remote_make_prediction(base_url, country, start_date, days=5):
    predictions = []
    for i in range(days):
        date = start_date + timedelta(days=i)
        response = requests.post(f"{base_url}/predict", json={
            "country": country,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "test": False
        })
        if response.status_code == 200:
            result = response.json()
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "revenue": result["y_pred"][0]
            })
        else:
            print(f"Failed to get prediction for {date} - Status Code: {response.status_code} - Response: {response.text}")
    return pd.DataFrame(predictions)

def read_and_aggregate_production_data(production_dir, year, month, day, country='all', days=10):
    start_date = datetime(int(year), int(month), int(day))
    end_date = start_date + timedelta(days=days)
    
    all_data = []
    for filename in os.listdir(production_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(production_dir, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                all_data.extend(data)
    
    df = pd.DataFrame(all_data)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['revenue'] = df['price'] * df['times_viewed']  # Assuming revenue is price * times_viewed

    # Filter the data to include only the dates within the prediction range
    mask = (df['date'] >= start_date) & (df['date'] < end_date)
    df = df.loc[mask]

    # Aggregate data by country if specified, otherwise aggregate all data
    if country != 'all':
        df = df[df['country'] == 'United Kingdom']

    aggregated_df = df.groupby('date')['revenue'].sum().reset_index()
    return aggregated_df

def analyze_performance(base_url, production_dir, output_dir, year, month, day, country,days=10, local=False):
    # Build the start date for predictions
    start_date = datetime(int(year), int(month), int(day))



    # Make predictions for the next 30 days from the start date
    if local:
        predictions_df = local_make_prediction(country, start_date,days)
    else:
        predictions_df = remote_make_prediction(base_url, country, start_date)

    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    # Read and aggregate the production data
    actuals_df = read_and_aggregate_production_data(production_dir, year, month, day, country,days)
    # Print the values before plotting
    print("Predictions DataFrame:")
    print(predictions_df)
    print("\nActuals DataFrame:")
    print(actuals_df)
    # Plot the predicted and actual revenue
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df['date'], predictions_df['revenue'], label='Predicted Revenue')
    plt.plot(actuals_df['date'], actuals_df['revenue'], label='Actual Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.title('30-Day Revenue Projection vs Actual')
    plt.legend()

    # Save the plot to the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'revenue_projection_vs_actual.png')
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    base_url = "http://localhost:5000"
    production_dir = "../data/cs-production"
    output_dir = "visualizations"
    year = 2019
    month = 8
    day = 1
    country = 'united_kingdom'
    
    # Analyze performance using local predictions
    analyze_performance(base_url, production_dir, output_dir, year, month, day, country,5, local=True)
    
    # Analyze performance using remote predictions
    # analyze_performance(base_url, production_dir, output_dir, year, month, day, country, local=False)