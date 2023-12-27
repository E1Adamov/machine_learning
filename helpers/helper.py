import os
import string

import nltk
import numpy as np
import requests
from tensorflow import keras  # noqa

from helpers.protocols import HasLtMethod
from root import ROOT_PATH


def get_path_in_repo(*rel_path_parts: str) -> str:
    return os.path.join(ROOT_PATH, *rel_path_parts)


def num_to_str_red_color(num: HasLtMethod, threshold: int = 90) -> str:
    """
    casts the <value> to string and colors red if it's below <threshold>
    """
    color = "\033[91m" if num < threshold else ""  # ANSI escape code for red text
    return f"{color}{num}\033[0m"  # Reset color after the value


def download_large_file(url, save_path, chunk_size=8192):
    with requests.get(url, stream=True) as response:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)


def bag_of_words(
    sentence: str, vocabulary: list[str], max_length: int
) -> np.ndarray[np.ndarray[int]]:
    """

    :param sentence: arbitrary text
    :param vocabulary: known words
    :param max_length: limit or pad each bag length to this value
    :return: list of 0-padded to <max_length> ints from the <vocabulary>, that represent words in the <sentence>
    """
    pattern = nltk.word_tokenize(sentence)
    stemmer = nltk.stem.lancaster.LancasterStemmer()
    pattern = [stemmer.stem(w.lower()) for w in pattern if w not in string.punctuation]
    bag = [vocabulary[w] for w in pattern]
    bag = keras.preprocessing.sequence.pad_sequences(
        [bag],
        value=0,
        padding="post",
        maxlen=max_length,
    )
    return bag
