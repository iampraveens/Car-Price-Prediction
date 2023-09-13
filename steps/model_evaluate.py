import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.evaluation import MSE, R2_Score, MAE

@step
def evaluate_model(model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2_score"],
    Annotated[float, "mae"],
]:
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        
        r2_class = R2_Score()
        r2 = r2_class.calculate_scores(y_test, prediction)
        
        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, prediction)
        
        return mse, r2, mae
    except Exception as e:
        logging.error(f"Error in evaluating model {e}")
        raise e