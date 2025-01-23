import time, os, re, csv, sys, uuid, joblib
from datetime import date, datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learning model for time-series"
# end date of the training data
END_DATE_STR = "2019-06-30"

def _model_train(df, tag, model_type='rf', test=False):
    """
    example function to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 
    """

    ## start timer for runtime
    time_start = time.time()
    
    X, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples, replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
    
    if model_type == 'rf':
        ## train a random forest model with additional hyperparameter tuning
        param_grid_rf = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [10, 20, 30, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2', None]
        }

        pipe_rf = Pipeline(steps=[('scaler', StandardScaler()), ('rf', RandomForestRegressor(random_state=42))])
        
        grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        ## retrain using all data
        grid.fit(X, y)
        model = grid

    elif model_type == 'lstm':
        ## train an LSTM model
        # Helper function to create sequences
        def create_sequences(features, target, seq_length):
            X, y = [], []
            for i in range(len(features) - seq_length):
                X.append(features[i:i + seq_length])
                y.append(target[i + seq_length])
            return np.array(X), np.array(y)

        # Scale features and target
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Log transform and scale revenue
        y_log = np.log1p(y)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel()

        # Train-test split (keep time order for time-series)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.25, shuffle=False
        )

        # Create sequences
        seq_length = 14  # Extended sequence length for capturing trends
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

        # Build LSTM model
        model = Sequential([
            # First LSTM layer
            LSTM(256, return_sequences=True, input_shape=(seq_length, X_train.shape[1])),
            BatchNormalization(),
            Dropout(0.3),

            # Second LSTM layer
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),

            # Third LSTM layer
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),

            # Dense layers
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Compile model with a robust loss function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='logcosh'  # Log-cosh is robust to outliers
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0001)
        ]

        # Train the model
        history = model.fit(
            X_train_seq,
            y_train_seq,
            epochs=100,
            batch_size=8,  # Reduced batch size for better generalization
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Predict and inverse transform the results
        y_pred_scaled = model.predict(X_test_seq)
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.expm1(y_pred_log)  # Reverse log transformation
        eval_rmse = round(np.sqrt(mean_squared_error(y_test_seq, y_pred)))
        # Save scalers and model metadata
        model_metadata = {
            'seq_length': seq_length,
            'scalers': {
                'X': scaler_X,
                'y': scaler_y
            }
        }
        joblib.dump(model_metadata, os.path.join(MODEL_DIR, f"{tag}_metadata.joblib"))

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    ## update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])), {'rmse': eval_rmse}, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

def model_train(data_dir, model_type='rf', test=False):
    """
    function to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country, df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df, country, model_type=model_type, test=test)
    
def model_load(prefix='sl', data_dir=None, training=True, test=False):
    """
    example function to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("..", "data", "cs-train")
    
    prefix = 'test' if test else prefix
    models = [f for f in os.listdir(os.path.join(".", "models")) if re.search(prefix, f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]] = joblib.load(os.path.join(".", "models", model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X": X, "y": y, "dates": dates}
        
    return all_data, all_models

def model_predict(country, year, month, day, all_models=None, test=False):
    """
    Predict function handling both LSTM and RF models
    """
    # Initialize variables
    time_start = time.time()
    y_pred = None
    y_proba = None

    # Load models if needed
    if not all_models:
        all_data, all_models = model_load(training=False, test=test)
    
    # Input validation
    if country not in all_models.keys():
        raise Exception(f"ERROR (model_predict) - model for country '{country}' could not be found")

    for d in [str(year), str(month), str(day)]:
        if re.search("\D", d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    # Load model and data
    model = all_models[country]
    data = all_data[country]

    # Date handling
    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    print(f"Target date: {target_date}")

    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
    target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")

    if target_date_obj < end_date or target_date_obj > end_date + timedelta(days=90):
        raise Exception(f"ERROR (model_predict) - date {target_date} not in range {END_DATE_STR}-{(end_date + timedelta(days=30)).strftime('%Y-%m-%d')}")

    # Prediction logic
    if isinstance(model, Sequential):  # LSTM model
        print("Using LSTM model")
        seq_length = 30
        
        if target_date not in data['dates']:
            date_indx = len(data['dates']) - 1
        else:
            date_indx = np.where(data['dates'] == target_date)[0][0]
        
        start_idx = max(0, date_indx - (seq_length - 1))
        query = data['X'].iloc[start_idx : date_indx + 1].values
        query = query.reshape(1, seq_length, -1)
        
        y_pred = model.predict(query)
        
    else:  # RF model
        if target_date not in data['dates']:
            date_indx = (target_date_obj - end_date).days
            query = data['X'].iloc[[-1]].copy()
            query.index = [date_indx]
        else:
            date_indx = np.where(data['dates'] == target_date)[0][0]
            query = data['X'].iloc[[date_indx]]

        if data['dates'].shape[0] != data['X'].shape[0]:
            raise Exception("ERROR (model_predict) - dimensions mismatch")

        y_pred = model.predict(query)
        
        if hasattr(model, 'predict_proba') and hasattr(model, 'probability'):
            if model.probability:
                y_proba = model.predict_proba(query)

    # Time tracking and logging
    runtime = time.time() - time_start
    update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=test)
    
    return {'y_pred': y_pred, 'y_proba': y_proba}

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    # print("TRAINING MODELS")
    data_dir = os.path.join("..","data","cs-train")
    # model_train(data_dir, model_type='rf', test=False)
    model_train(data_dir, model_type='lstm', test=False)

    ## load the model
    # print("LOADING MODELS")
    # all_data, all_models = model_load()
    # print("... models loaded: ",",".join(all_models.keys()))

    # ## test predict
    # country='all'
    # year='2019'
    # month='08'
    # day='01'
    # result = model_predict('france',year,month,day)
    # print(result)
