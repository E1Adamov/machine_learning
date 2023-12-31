from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tensorflow import keras  # noqa
import matplotlib.pyplot as plt


ANOMALIES_COUNT = 4000
NOISE_SCALE = 0.05
PREDICTION_THRESHOLD = 0.5
RANDOM_SEED = 42

# Set a random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Simulate minute time points
minutes = pd.date_range(start="2023-01-01", end="2023-01-07", freq="min")


def generate_cpu_load(load_by_hour: list[float], noise: float):
    """
    Simulate daily patterns with random noise and anomalies
    """
    # Generate time values
    time = np.linspace(
        start=0,
        stop=len(load_by_hour),
        num=len(load_by_hour) * 60,  # a point for each minute
        endpoint=False,
    )

    # Upsample load_by_hour and interpolate for a smooth curve
    load_schedule = np.interp(time, np.arange(len(load_by_hour)), load_by_hour)

    # Add some random noise
    noise = np.random.normal(loc=0, scale=noise, size=len(time))
    load_schedule += noise

    return load_schedule


def timestamp_to_sequential_minute(iso_timestamp: str) -> int:
    dt_object = datetime.fromisoformat(iso_timestamp)
    sequential_minute = dt_object.minute + 60 * dt_object.hour
    return sequential_minute


load_by_hours = [
    0.15,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.12,
    0.15,
    0.15,
    0.25,
    0.5,
    0.5,
    0.35,
    0.3,
    0.3,
    0.35,
    0.4,
    0.35,
    0.3,
    0.4,
    0.5,
    0.5,
    0.4,
    0.15,
]

# Generate CPU load for each day
cpu_load_data = np.concatenate(
    [
        generate_cpu_load(load_by_hour=load_by_hours, noise=NOISE_SCALE)
        for _ in range(len(minutes) // 60 // 24)
    ]
)

# Trim the excess data if needed
cpu_load_data = cpu_load_data[: len(minutes)]
minutes = minutes[: len(cpu_load_data)]

# Create a DataFrame with the simulated data
cpu_load_df = pd.DataFrame({"Timestamp": minutes, "CPU_Load": cpu_load_data})
cpu_load_df["MinuteOfDay"] = (
    cpu_load_df["Timestamp"].dt.minute + 60 * cpu_load_df["Timestamp"].dt.hour + 1
)

# Assuming cpu_load_data is your training data
window_size = 30  # Affects the smoothness of the mean, and therefore thresholds
threshold_scaling_factor = 3

# Calculate rolling mean and rolling standard deviation
rolling_mean = pd.Series(cpu_load_data).rolling(window=window_size, center=True).mean()
rolling_std = pd.Series(cpu_load_data).rolling(window=window_size, center=True).std()

# smoothen the rolling mean and standard deviation
rolling_mean = pd.Series(
    savgol_filter(x=rolling_mean, window_length=window_size, polyorder=2)
)
rolling_std = pd.Series(
    savgol_filter(x=rolling_std, window_length=window_size * 5, polyorder=2)
)

# Define the threshold range based on the rolling mean and rolling standard deviation
lower_threshold = rolling_mean - threshold_scaling_factor * rolling_std
upper_threshold = rolling_mean + threshold_scaling_factor * rolling_std

# Trim NaN values from the beginning and end of thresholds
lower_threshold = lower_threshold.dropna()
upper_threshold = upper_threshold.dropna()

# Trim the corresponding data accordingly
cpu_load_df = cpu_load_df.iloc[
    upper_threshold.index[0] : upper_threshold.index[-1] + 1  # noqa
]

# Add thresholds to the cpu_load_df
cpu_load_df["Lower_Threshold"] = lower_threshold.values
cpu_load_df["Upper_Threshold"] = upper_threshold.values

# Add anomaly data points
anomaly_indices = np.random.choice(
    len(cpu_load_df), size=ANOMALIES_COUNT, replace=False
)
anomaly_noise_scale = 5 * NOISE_SCALE
cpu_load_df.iloc[
    anomaly_indices, cpu_load_df.columns.get_loc("CPU_Load")
] += np.random.normal(loc=0, scale=anomaly_noise_scale, size=ANOMALIES_COUNT)

# Mark anomalies based on the thresholds
cpu_load_df["Anomaly"] = (  # noqa
    (cpu_load_df["CPU_Load"] < cpu_load_df["Lower_Threshold"])
    | (cpu_load_df["CPU_Load"] > cpu_load_df["Upper_Threshold"])
).astype(int)

# Prepare the training data
train_data = cpu_load_df[["CPU_Load", "Anomaly"]].copy()
train_data["Lower_Threshold"] = lower_threshold.values
train_data["Upper_Threshold"] = upper_threshold.values

# Visualize the training data
limit = 1400  # minutes in 1 day
plt.figure(figsize=(20, 6))
plt.xlim(train_data.index[0], train_data[:limit].index[-1])
plt.plot(train_data["CPU_Load"][:limit], label="CPU Load Data")
plt.plot(
    rolling_mean[train_data.index[0] : train_data[:limit].index[-1]],  # noqa
    label="Rolling Mean",
)
plt.scatter(
    train_data.loc[
        (train_data["Anomaly"] == 1) & (train_data.index <= limit + 100)
    ].index,
    train_data.loc[
        (train_data["Anomaly"] == 1) & (train_data.index < limit + 100), "CPU_Load"
    ][:limit],
    color="red",
    label="Anomalies in the training data",
)
plt.fill_between(
    x=train_data["Upper_Threshold"].index[:limit],
    y1=train_data["Lower_Threshold"][:limit],
    y2=train_data["Upper_Threshold"][:limit],
    color="orange",
    alpha=0.2,
    label="Anomaly Threshold Range",
)
plt.title("Training data - CPU Load with Simulated Anomalies")
plt.legend()
plt.show()

# Train-Test Split
train_split = 0.2
train_split = int(len(train_data) * train_split)
test_set = train_data[:train_split]
train_set = train_data[train_split:]

# Features and Labels
X_train = train_set[["CPU_Load", "Lower_Threshold", "Upper_Threshold"]]
y_train = train_set["Anomaly"]
X_test = test_set[["CPU_Load", "Lower_Threshold", "Upper_Threshold"]]
y_test = test_set["Anomaly"]

model = keras.Sequential(
    layers=[
        keras.layers.Dense(64, activation="relu", input_shape=(len(X_train.iloc[0]),)),
        keras.layers.Dense(units=32, activation="relu"),
        keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    x=X_train.values,
    y=y_train.values,
    epochs=16,
    batch_size=32,
    verbose=1,
    validation_data=(
        X_test,
        y_test,
    ),
)


# Evaluate on test set
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(20, 6))
plt.plot(X_test["CPU_Load"].index, X_test["CPU_Load"], label="CPU Load")
plt.scatter(
    X_test.loc[y_pred > PREDICTION_THRESHOLD].index,
    X_test.loc[y_pred > PREDICTION_THRESHOLD, "CPU_Load"],
    color="red",
    label="Predicted Anomalies",
)
plt.fill_between(
    x=X_test["Upper_Threshold"].index,
    y1=X_test["Lower_Threshold"],
    y2=X_test["Upper_Threshold"],
    color="orange",
    alpha=0.2,
    label="Anomaly Threshold Range",
)
plt.xlabel("Minute")
plt.ylabel("CPU Load")
plt.title("CPU Load with Predicted Anomalies - Sequential Model")
plt.legend()
plt.show()


# Simulate incoming json and prediction
for cpu_load in 0.53, 0.49:
    incoming_json = {
        "CPU_Load": cpu_load,
        "timestamp": "2023-12-28T15:30:45.123456",
    }

    minute_in_day = timestamp_to_sequential_minute(
        iso_timestamp=incoming_json["timestamp"]
    )
    known_data_at_this_time_of_day = cpu_load_df[
        cpu_load_df["MinuteOfDay"] == minute_in_day
    ]

    new_data_point = pd.DataFrame(
        {
            "CPU_Load": [incoming_json["CPU_Load"]],
            "Lower_Threshold": [
                known_data_at_this_time_of_day["Lower_Threshold"].mean()
            ],
            "Upper_Threshold": [
                known_data_at_this_time_of_day["Upper_Threshold"].mean()
            ],
        }
    )

    prediction = model.predict(new_data_point)[0]
    print(f"Data: {new_data_point}")
    print(f"Prediction score: {prediction}")
    print(f"Is this an anomaly? {prediction > PREDICTION_THRESHOLD}")
    print()
