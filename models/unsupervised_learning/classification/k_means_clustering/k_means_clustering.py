import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, cluster

from helpers import helper

# download the train data from the 2020 Force unsupervised learning contest
csv_file_path = helper.get_path_in_repo(
    "data", "force2020_data_unsupervised_learning.csv"
)
# read the csv
data_df = pd.read_csv(
    csv_file_path, index_col="DEPTH_MD"
)  # set DEPTH_MD as row indexes

# clear and prepare the data:
data_df.dropna(inplace=True)  # remove NaN otherwise KMeans model will fail
# check data uniformity
print(
    data_df.describe()
)  # we can see that data is not standardised: e.g. min/max values are different, this will result in incorrect model
# standardise the data
scaler = preprocessing.StandardScaler()
# add standardised columns
data_df[
    ["RHOB_STD", "GR_STD", "NPHI_STD", "PEF_STD", "DTC_STD"]
] = scaler.fit_transform(data_df[["RHOB", "GR", "NPHI", "PEF", "DTC"]])
pd.set_option("display.max_columns", 500)
print(
    data_df[["RHOB_STD", "GR_STD", "NPHI_STD", "PEF_STD", "DTC_STD"]].describe()
)  # we can see the data in ..._STD columns is standardised


# identify the optimal number of K clusters by method "elbow":
# create the KMeans model with different number of clusters and
# look at the inertia. The lower the inertia - the tighter and well-separated the clusters are
def visualize_elbow_plot(data: pd.DataFrame, max_k: int):
    means = []
    inertia = []

    for k_ in range(1, max_k):
        k_means_ = cluster.KMeans(n_clusters=k_, n_init="auto")
        k_means_.fit(data)

        means.append(k_)
        inertia.append(k_means_.inertia_)

    plt.plot(means, inertia, "o-")
    plt.title("Elbow plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


visualize_elbow_plot(data=data_df[["RHOB_STD", "NPHI_STD"]], max_k=10)
# we can see that after 3 clusters, the inertia is going down very gradually

k_means_model = cluster.KMeans(n_clusters=3, n_init="auto")
k_means_model.fit(data_df[["RHOB_STD", "NPHI_STD"]])
data_df[
    "k_means_3_cluster"
] = k_means_model.labels_  # add a columns showing which row belongs to which cluster

# visualise the result of training
plt.scatter(x=data_df["NPHI"], y=data_df["RHOB"], c=data_df["k_means_3_cluster"])
plt.xlabel("NPHI")
plt.ylabel("RHOB")
plt.show()


# visualize clustered data with different K values near earlier defined K to make sure the choice was correct
k_limit = 5
fig, axes = plt.subplots(nrows=1, ncols=k_limit - 1, figsize=(20, 5))
for k, ax in enumerate(fig.axes, start=2):
    k_means = cluster.KMeans(n_clusters=k, n_init="auto")
    k_means.fit(data_df[["RHOB_STD", "NPHI_STD"]])
    ax.scatter(x=data_df["NPHI"], y=data_df["RHOB"], c=k_means.labels_)
    ax.set_title(f"N Clusters: {k}")
    plt.tight_layout()
plt.show()
# so we can see that 4 clusters are good, so we can pick 4 as K
