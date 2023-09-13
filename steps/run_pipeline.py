import os

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    
    data_path = os.path.join('data', 'car_data.csv')
    train_pipeline(data_path=data_path)

