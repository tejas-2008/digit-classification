import itertools
import pytest
import joblib
import os
from sklearn.linear_model import LogisticRegression
model_dir = "models"
model_files = os.listdir(model_dir)
def test_loaded_model_is_logistic_regression():
    # Replace <rollno> and <solver_name> with actual values
    for model_path in model_files:
        model_path = os.path.join(model_dir, model_path)
        loaded_model = joblib.load(model_path)
        assert isinstance(loaded_model, LogisticRegression)
        


def test_solver_name_match_in_file_name_and_model():
    for model_path in model_files:
        model_path = os.path.join(model_dir, model_path)
        loaded_model = joblib.load(model_path)

        # Extract solver name from the file name
        file_solver_name = model_path.split("_")[-1].split(".")[0].split(":")[-1]
        print(file_solver_name)
        # Extract solver name from the loaded model
        model_solver_name = loaded_model.get_params()["solver"]
        print(model_solver_name)
        assert file_solver_name == model_solver_name

