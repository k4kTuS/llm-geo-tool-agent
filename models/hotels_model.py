import os

from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

def train_hotels_model(data, test_size=0.2, rd_seed=42, iters=5000, lr=0.01, depth=5, eval=False):
    # Prepare data
    X = data.drop(columns=['lodging'])
    y = data['lodging']
    
    # Handle missing values (if necessary)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rd_seed
    )
    
    # Initialize model
    model = CatBoostRegressor(
        iterations=iters,
        learning_rate=lr,
        depth=depth,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=rd_seed,
        early_stopping_rounds=50,
        logging_level=None,
    )
    
    # Fit model
    model.fit(
        X_train.loc[:,],
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        verbose=False, #Change if needed
        plot=False, #Change if needed
    )
    
    
    if eval:
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    return model

@st.cache_resource
def load_model():
    if not os.path.exists("../saved_models/hotels_cbm"):
        hotels_data = pd.read_csv("../data/hotels.csv", index_col=0)
        model = train_hotels_model(hotels_data)
        model.save_model("../saved_models/hotels_cbm")
        return model
    else:
        model = CatBoostRegressor()
        model.load_model("../saved_models/hotels_cbm")
        return model

@st.cache_data
def load_features():
    hotels_data = pd.read_csv("../data/hotels.csv", index_col=0)
    hotels_data_X = hotels_data.drop(columns=['lodging'])
    hotels_data_X = hotels_data_X.fillna(hotels_data_X.mean())
    return hotels_data_X
