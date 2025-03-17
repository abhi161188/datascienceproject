import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories, save_json

#import os
#os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/abhi161188/datascienceproject.mlflow"
#os.environ['MLFLOW_TRACKING_USERNAME'] = "abhi161188"
#os.environ['MLFLOW_TRACKING_PASSWORD'] = "22a3edee0b18641c77ececd5d3b4713d07e0a9b0"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrcis(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def  log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis = 1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrcis(test_y, predicted_qualities)

            #saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path = Path(self.config.metric_file_name), data = scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model Registry does not work with file store
            if tracking_url_type_store != 'file':

                # Register the model
                # There are other ways to use the Model Registry, which dependa on the use case,
                # please refer to the doc for more information:registered_model_name
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name = "ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
