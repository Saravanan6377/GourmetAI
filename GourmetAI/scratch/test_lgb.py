import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os

def test_lgb():
    # Generate dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'col_{i}' for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    train_data = lgb.Dataset(X, label=y)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1
    }
    
    gbm = lgb.train(params, train_data, num_boost_round=10)
    
    # Save model
    model_path = 'scratch/test_lgb_model.pkl'
    joblib.dump(gbm, model_path)
    
    # Load model
    loaded_gbm = joblib.load(model_path)
    
    # Predict
    test_df = pd.DataFrame(np.random.rand(1, 5), columns=[f'col_{i}' for i in range(5)])
    pred = loaded_gbm.predict(test_df)
    print("Prediction:", pred)
    print("Type of prediction:", type(pred))
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == '__main__':
    test_lgb()
