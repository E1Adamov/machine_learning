import math
import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors, preprocessing
from matplotlib import pyplot as plt

from helpers import helpers

new_model_created = False

# the column that we're going to predict
predict = "class"

# read the data from csv
students_data_path = helpers.get_path_in_repo("data", "car.data")
initial_data = pd.read_csv(students_data_path)

# since the initial data is cathegorical (contains text values), we need to transform them
label_encoder = preprocessing.LabelEncoder()
for column_name in initial_data.columns:
    initial_column_data = initial_data[column_name]
    cleaned_up_column_data = label_encoder.fit_transform(initial_column_data)
    initial_data[column_name] = cleaned_up_column_data.reshape(-1, 1)

# split data into what we're going to feed to the model, and what it must predict
input_data = initial_data.drop(columns=[predict])
output_data = initial_data[predict]

(
    input_data_train,
    input_data_test,
    output_data_train,
    output_data_test,
) = model_selection.train_test_split(input_data, output_data, test_size=0.1)

# create the model and train it
k_nearest_neighbors_model = neighbors.KNeighborsClassifier(n_neighbors=9)
k_nearest_neighbors_model.fit(X=input_data_train, y=output_data_train)

# visualize predictions along with train and test data
predictions_int = k_nearest_neighbors_model.predict(X=input_data_test)
# since "class" is the last column in .csv, the label_encoded still holds data from it
predictions_str = label_encoder.inverse_transform(predictions_int)
expected_prediction_str = label_encoder.inverse_transform(output_data_test)
prediction_range = abs(max(output_data_train) - min(output_data_train))
test_model_df = pd.DataFrame(
    [
        {
            "input_test_data": input_data_test.iloc[index].values,
            "expected_prediction": expected_prediction_str[index],
            "actual_prediction": predictions_str[index],
            "accuracy": int(
                (
                    prediction_range
                    - abs(output_data_test.iloc[index] - predictions_int[index])
                )
                / prediction_range
                * 100
            ),
        }
        for index in range(len(predictions_int))
    ]
)
test_model_df["accuracy"] = test_model_df["accuracy"].apply(helpers.df_cell_red_color)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
print(test_model_df)
