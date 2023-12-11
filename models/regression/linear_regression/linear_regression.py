import math
import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model
from matplotlib import pyplot as plt

from helpers import helpers

new_model_created = False

# the column that we're going to predict
predict = "G3"

# read the data from csv
students_data_path = helpers.get_path_in_repo("data", "student-mat.csv")
full_data = pd.read_csv(students_data_path, sep=";")

# split data into what we're going to feed to the model, and what it must predict
input_data = full_data[
    ["G1", "G2", "studytime", "failures", "absences"]
]  # the model will take data from a few columns
output_data = full_data[[predict]]  # the model will predict G3 value

# split  x and y: 0.1 to test data, 0.9 to learning data
# this way, tests are done against data that has not been seen
(
    input_data_train,
    input_data_test,
    output_data_train,
    output_data_test,
) = model_selection.train_test_split(input_data, output_data, test_size=0.1)

# read the trained model if it already exists
model_path = os.path.join(os.path.dirname(__file__), "linear_model.pkl")
if os.path.isfile(model_path):
    with open(model_path, "rb") as f:
        linear_regression_model: linear_model.LinearRegression = pickle.load(f)
else:
    # create the model and train it
    linear_regression_model = linear_model.LinearRegression()
    linear_regression_model.fit(X=input_data_train, y=output_data_train)
    new_model_created = True

# try to re-train to improve accuracy
current_accuracy = linear_regression_model.score(X=input_data_test, y=output_data_test)
for attempt in range(10):
    # split  x and y: 0.1 to test data, 0.9 to learning data
    # this way, tests are done against data that has not been seen
    (
        input_data_train,
        input_data_test,
        output_data_train,
        output_data_test,
    ) = model_selection.train_test_split(input_data, output_data, test_size=0.1)
    new_model = linear_model.LinearRegression()
    new_model.fit(X=input_data_train, y=output_data_train)
    new_accuracy = linear_regression_model.score(X=input_data_test, y=output_data_test)
    if new_accuracy > current_accuracy:
        linear_regression_model = new_model
        new_model_created = True
        break

if new_model_created:
    # save the model
    with open(model_path, "wb") as f:
        pickle.dump(linear_regression_model, f)

# test the model
linear_model_accuracy = linear_regression_model.score(
    X=input_data_test, y=output_data_test
)
print(f"Model accuracy: {linear_model_accuracy}")

# output the model coefficients
data_importance_coefficients = pd.DataFrame(
    linear_regression_model.coef_, columns=input_data.columns
)
print(f"Importance coefficients of the input data:")
print(data_importance_coefficients)

# output the intercept - the point where the linear regression crosses the y-axis
print(f"Intercept: {linear_regression_model.intercept_}")

# visualize predictions along with train and test data
predictions: np.ndarray = linear_regression_model.predict(input_data_test)
prediction_range = abs(
    max(output_data_train[predict]) - min(output_data_train[predict])
)
test_model_df = pd.DataFrame(
    [
        {
            "input_test_data": input_data_test.iloc[index].values,
            "expected_prediction": output_data_test.iloc[index].iloc[0],
            "actual_prediction": predictions[index][0],
            "accuracy": int(
                (
                    prediction_range
                    - abs(output_data_test.iloc[index].iloc[0] - predictions[index][0])
                )
                / prediction_range
                * 100
            ),
        }
        for index in range(len(predictions))
    ]
)
test_model_df["accuracy"] = test_model_df["accuracy"].apply(helpers.df_cell_red_color)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
print(test_model_df)

# plot the correlation between initial data and G3
plt.style.use("ggplot")
n1 = int(math.sqrt(len(input_data.columns)))
n2 = len(input_data.columns) - n1
nrows = min(n1, n2)
ncols = max(n1, n2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 10))

i = 0
for row in axes:
    for ax in row:
        if i < len(input_data.columns):
            x_axis = input_data.columns[i]
            y_axis = output_data.columns[0]
            ax.scatter(x=input_data[x_axis], y=output_data[y_axis])
            ax.set_title(
                f'Correlation between "{x_axis}" and "{y_axis}" in the initial data'
            )
            ax.set_xlabel(f"{x_axis}: {linear_regression_model.coef_[0][i]}")
            ax.set_ylabel(y_axis)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.5)
            i += 1
        else:
            fig.delaxes(ax)
plt.tight_layout()
plt.show()
print()
