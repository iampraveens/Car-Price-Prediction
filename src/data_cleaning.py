import logging
from typing import Union
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataStrategy(ABC):
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data = data.drop(columns=['name', 'max_power', 'torque'], axis=1)
            data['mileage'] = data['mileage'].str.replace('kmpl', '')
            data['mileage'] = data['mileage'].str.replace('km/kg', '')
            data['engine'] = data['engine'].str.replace('CC', '')
            
            data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')  # 'coerce' will convert non-numeric values to NaN
            data['engine'] = pd.to_numeric(data['engine'], errors='coerce')
            data['seats'] = pd.to_numeric(data['seats'], errors='coerce')
            
            data['mileage'].fillna(data['mileage'].mean(), inplace=True)
            data['engine'].fillna(data['engine'].mean(), inplace=True)
            data['seats'].fillna(data['seats'].mean(), inplace=True)
            
            data['engine'] = data['engine'].astype('float64')
            data['seats'] = data['seats'].astype('int64')
            data.drop_duplicates(inplace=True)
            
            iqr_year = data['year'].quantile(0.75) - data['year'].quantile(0.25)
            upper_thersold_year = data['year'].quantile(0.75) + (1.5 * iqr_year)
            lower_thersold_year = data['year'].quantile(0.25) - (1.5 * iqr_year)
            data['year'] = data['year'].clip(lower_thersold_year, upper_thersold_year)
            
            iqr_price = data['selling_price'].quantile(0.75) - data['selling_price'].quantile(0.25)
            upper_thersold_price = data['selling_price'].quantile(0.75) + (1.5 * iqr_price)
            lower_thersold_price = data['selling_price'].quantile(0.25) - (1.5 * iqr_price)
            data['selling_price'] = data['selling_price'].clip(lower_thersold_price, upper_thersold_price)
            
            iqr_km = data['km_driven'].quantile(0.75) - data['km_driven'].quantile(0.25)
            upper_thersold_km = data['km_driven'].quantile(0.75) + (1.5 * iqr_km)
            lower_thersold_km = data['km_driven'].quantile(0.25) - (1.5 * iqr_km)
            data['km_driven'] = data['km_driven'].clip(lower_thersold_km, upper_thersold_km)
            
            iqr_mileage = data['mileage'].quantile(0.75) - data['mileage'].quantile(0.25)
            upper_thersold_mileage = data['mileage'].quantile(0.75) + (1.5 * iqr_mileage)
            lower_thersold_mileage = data['mileage'].quantile(0.25) - (1.5 * iqr_mileage)
            data['mileage'] = data['mileage'].clip(lower_thersold_mileage, upper_thersold_mileage)
            
            iqr_engine = data['engine'].quantile(0.75) - data['engine'].quantile(0.25)
            upper_thersold_engine = data['engine'].quantile(0.75) + (1.5 * iqr_engine)
            lower_thersold_engine = data['engine'].quantile(0.25) - (1.5 * iqr_engine)
            data['engine'] = data['engine'].clip(lower_thersold_engine, upper_thersold_engine)
            
            iqr_seats = data['seats'].quantile(0.75) - data['seats'].quantile(0.25)
            upper_thersold_seats = data['seats'].quantile(0.75) + (1.5 * iqr_seats)
            lower_thersold_seats = data['seats'].quantile(0.25) - (1.5 * iqr_seats)
            data['seats'] = data['seats'].clip(lower_thersold_seats, upper_thersold_seats)
            
            data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission'], drop_first=True, dtype='int')
            data['owner'] = data['owner'].map({'Test Drive Car': 3, 'Fourth & Above Owner': 0, 'Third Owner': 1, 'Second Owner': 2, 'First Owner': 3})
            
            return data
        except Exception as e:
            logging.error(f"Error while preprocessing the data {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(columns=['selling_price'], axis=1)
            y = data['selling_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error while splitting the data {e}")
            raise e
        
class DataScaleStrategy(DataStrategy):
    
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X_train)
            X_test = scaler.transform(self.X_test)
            
            X_train = pd.DataFrame(X_train, columns=self.X_train.columns)
            X_test = pd.DataFrame(X_test, columns=self.X_train.columns)
            
            return X_train, X_test
        except Exception as e:
            logging.error(f"Error while scaling the data {e}")
            raise e
        
class DataCleaning:
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self):
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error while handling the data")
            raise e
        