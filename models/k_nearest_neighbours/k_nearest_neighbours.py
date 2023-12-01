import math
import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model, preprocessing
from matplotlib import pyplot as plt

from helpers import helpers


new_model_created = False

# read the data from csv
students_data_path = helpers.get_path_in_repo("data", "car.data")
full_data = pd.read_csv(students_data_path)
print(full_data.head())