# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
import numpy as np
import TaxiFareModel.data
from sklearn.model_selection import train_test_split
import pandas as pd



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    
    
    def set_pipeline(self):
        '''defines the pipeline as a class attribute'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline
    
    def train(self):
        self.pipeline.fit(self.X, self.y)
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.train()
        return self.pipeline
    
        

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)
        rmse = TaxiFareModel.utils.compute_rmse(y_pred, y_test)
        return rmse
    
    


if __name__ == "__main__":
    # get data
    df = TaxiFareModel.data.get_data(nrows=10_000)
    # clean data
    df = TaxiFareModel.data.clean_data(df, test=False)
    # set X and y
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.run
    # evaluate
    trainer.evaluate(X_test,y_test)
    print('TODO')
