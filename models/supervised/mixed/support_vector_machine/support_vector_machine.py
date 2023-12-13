import pandas as pd
from sklearn import datasets, model_selection, svm

from helpers import helpers

cancer_dataset = datasets.load_breast_cancer()
# print(cancer_dataset.feature_names)
# print(cancer_dataset.target_names)

# data in this dataset is already split into what we're going to feed to the model, and what it must predict
input_data = cancer_dataset.data
output_data = cancer_dataset.target

(
    input_data_train,
    input_data_test,
    output_data_train,
    output_data_test,
) = model_selection.train_test_split(input_data, output_data, test_size=0.2)

# different kernel types: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
support_vector_machine_classifier = svm.SVC(kernel="linear")
support_vector_machine_classifier.fit(X=input_data_train, y=output_data_train)
predictions = support_vector_machine_classifier.predict(X=input_data_test)

accuracy_df = pd.DataFrame(
    {
        "expected_prediction": [
            cancer_dataset.target_names[num] for num in output_data_test
        ],
        "actual_prediction": [cancer_dataset.target_names[num] for num in predictions],
        "accuracy": [
            (1 - abs(expected - actual)) * 100
            for expected, actual in zip(output_data_test, predictions)
        ],
    }
)
accuracy_df["accuracy"] = accuracy_df["accuracy"].apply(helpers.df_cell_red_color)
pd.set_option("display.max_rows", 500)
print(accuracy_df)
print(
    f"Accuracy: {support_vector_machine_classifier.score(X=input_data_test, y=output_data_test)}"
)
