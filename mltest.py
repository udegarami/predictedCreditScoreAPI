import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, fbeta_score
from mlflow.tracking import MlflowClient

def train_xgboost(train_data, test_data, max_depth, learning_rate, n_estimators, gamma, subsample, colsample_bytree):
    with mlflow.start_run():
        # Train the XGBoost model
        model = xgb.XGBClassifier(max_depth=int(max_depth),
                                  learning_rate=learning_rate,
                                  n_estimators=int(n_estimators),
                                  gamma=gamma,
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  objective="binary:logistic",
                                  n_jobs=-1)
        model.fit(train_data.drop("target", axis=1), train_data["target"])

        # Make predictions on the test data
        predictions = model.predict(test_data.drop("target", axis=1))

        # Log the model to the tracking server
        mlflow.xgboost.log_model(model, "model")

        # Calculate and log the AUC score
        auc_score = roc_auc_score(test_data["target"], predictions)
        mlflow.log_metric("auc_score", auc_score)

        # Calculate and log the F1 Beta score
        fbeta_score = fbeta_score(test_data["target"], predictions, beta=0.5)
        mlflow.log_metric("fbeta_score", fbeta_score)

        return fbeta_score

if __name__ == "__main__":
    train_data = pd.read_csv("train_encoded.csv")
    test_data = pd.read_csv("test_encoded.csv")
    
    mlflow.run(".",
               parameters={
                   "max_depth": (3, 5),
                   "learning_rate": (0.01, 0.2),
                   "n_estimators": (50, 200),
                   "gamma": (0, 1),
                   "subsample": (0.5, 1),
                   "colsample_bytree": (0.5, 1)
               },
               experiment_name="xgboost_hyperparameter_tuning",
               version=None,
               entry_point="train_xgboost",
               backend="local")
    
    # Find the best run based on the F1 Beta score
    client = MlflowClient()
    experiment = client.get_experiment_by_name("xgboost_hyperparameter_tuning")