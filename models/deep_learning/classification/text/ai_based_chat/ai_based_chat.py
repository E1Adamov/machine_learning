import json
import os
import pickle
import random
import string
from typing import Tuple, Dict

import nltk
from sklearn import preprocessing
from tensorflow import keras  # noqa
import numpy as np

from helpers import helper


LOAD_MODEL = False
PATTERN_LENGTH_LIMIT = 255

intents_json_path = helper.get_path_in_repo("data", "intents.json")


def get_topic_predicting_model() -> Tuple[keras.Sequential, list[str], Dict[int, str]]:
    model_path = os.path.join(
        os.path.dirname(__file__), "chat_question_topic_model.pkl"
    )
    if LOAD_MODEL and os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            tag_predicting_model, vocabulary, labels_map = pickle.load(f)
    else:
        nltk.download("punkt")

        with open(intents_json_path) as f:
            data = json.load(f)

        vocabulary = []  # all the known unique vocabulary.lower() in their root form
        unique_labels = set()  # all unique tags
        patterns = []  # all patterns split, with vocabulary in root form
        tags_corresponding_to_patterns = (
            []
        )  # label (tag) corresponding to each pattern in docs_x

        stemmer = nltk.stem.lancaster.LancasterStemmer()

        for intent in data["intents"]:
            tag: str = intent["tag"].lower()

            unique_labels.add(tag)

            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(
                    pattern
                )  # split by " ", including punctuation

                wrds = [
                    stemmer.stem(w.lower()) for w in wrds if w not in string.punctuation
                ]
                patterns.append(wrds)
                vocabulary.extend(wrds)

                tags_corresponding_to_patterns.append(tag)

        vocabulary = sorted(list(set(vocabulary)))  # remove duplicates
        vocabulary = {
            word: code for word, code in zip(vocabulary, range(1, len(vocabulary) + 1))
        }
        unique_labels = sorted(list(unique_labels))
        label_encoder = preprocessing.LabelEncoder()
        encoded_labels = label_encoder.fit_transform(unique_labels)
        labels_map = {
            label_int: label_str
            for label_int, label_str in zip(encoded_labels, unique_labels)
        }

        encoded_labels_corresponding_to_patterns = label_encoder.fit_transform(
            tags_corresponding_to_patterns
        )

        # encoded patterns that say which word from <vocabulary> is used in a pattern
        # 1 means the word was used, agnostic of how many times it was used
        training: list[list[int]] = []

        longest_pattern = max(len(p) for p in patterns)
        global PATTERN_LENGTH_LIMIT
        PATTERN_LENGTH_LIMIT = max(longest_pattern + 20, PATTERN_LENGTH_LIMIT)

        for pattern in patterns:
            bag_of_words = [vocabulary[word] for word in pattern]
            bag_of_words = bag_of_words + [0] * (
                PATTERN_LENGTH_LIMIT - len(bag_of_words)
            )
            training.append(bag_of_words)

        training_data: np.ndarray = np.array(training)

        tag_predicting_model = keras.Sequential(
            layers=[
                keras.layers.Dense(
                    units=PATTERN_LENGTH_LIMIT,
                    input_shape=(PATTERN_LENGTH_LIMIT,),
                    activation="relu",
                ),  # Input layer
                keras.layers.Dense(units=16, activation="relu"),  # Hidden layer
                keras.layers.Dense(
                    units=len(unique_labels), activation="softmax"
                ),  # Output layer
            ]
        )

        tag_predicting_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        tag_predicting_model.fit(
            x=training_data,
            y=encoded_labels_corresponding_to_patterns,
            epochs=350,
            batch_size=8,
            verbose=1,
        )
        with open(model_path, "wb") as f:
            pickle.dump((tag_predicting_model, vocabulary, labels_map), f)

    return tag_predicting_model, vocabulary, labels_map


def chat():
    model, vocabulary, labels_map = get_topic_predicting_model()
    with open(intents_json_path) as f:
        data = json.load(f)

    print("Start communication with the bot ('q' to quit")
    while True:
        inp = input("You: ")

        if inp == "q":
            break

        bag_of_words = helper.bag_of_words(
            sentence=inp, vocabulary=vocabulary, max_length=PATTERN_LENGTH_LIMIT
        )
        prediction = model.predict(bag_of_words)
        index_of_the_predicted_label = np.argmax(prediction[0])
        predicted_label = labels_map[index_of_the_predicted_label]

        for intent in data["intents"]:
            tag: str = intent["tag"].lower()
            if tag == predicted_label:
                possible_answers = intent["responses"]
                response = random.choice(possible_answers)
                print(tag, response)


if __name__ == "__main__":
    chat()
