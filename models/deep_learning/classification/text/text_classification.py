import os
import pickle

from tensorflow import keras  # noqa
from tensorflow.python.util.lazy_loader import KerasLazyLoader


imdb_data: KerasLazyLoader = (
    keras.datasets.imdb
)  # https://ai.stanford.edu/~amaas/data/sentiment/

# this is a data set of movie reviews. In each review, each word is represented by an integer key from a global word map
# these are actually default values for imdb_data.load_data(), they are use here explicitly for better readability
start_char = 1  # The start of each review will be marked with this character
oov_char = 2  # out-of-vocabulary character. Words that were cut out because of the `num_words` or`skip_top`
index_from = 3  # Index actual words with this index and higher
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(
    num_words=10000,  # pick only 10K most frequent words so that we have only relevant data
    start_char=start_char,
    oov_char=oov_char,
    index_from=index_from,
)

# here's this global word map: keys are words, values are integers, so we need to invert this map
word_index = imdb_data.get_word_index()

# in order to make each review of equal length, we need to add 4 new "utility" words to the map, for later use.
# Index 0 is not used, so we shift all indexes by 3
word_index = {k: v + 3 for k, v in word_index.items()}
# and add new words for the 4 vacant indices: 0-3
word_index["<PAD>"] = 0
word_index["<START>"] = start_char
word_index["<UNK>"] = oov_char
word_index["<UNUSED>"] = 3

# same map, but here ints are keys and words are values
inverted_word_index = {index: word for word, index in word_index.items()}


def decode_review(encoded_review: list[int]) -> str:
    return " ".join(inverted_word_index[i] for i in encoded_review)


print(f"First movie review BEFORE preprocessing")
print(decode_review(train_data[0]))

# make each review 250 words long. Longer ones will be cut, shorter ones will be padded with 0's in the end
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],  # 0
    padding="post",  # add to the end
    maxlen=250,  # cut longer ones, and pad shorter ones to 250
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],  # 0
    padding="post",  # add to the end
    maxlen=250,  # cut longer ones, and pad shorter ones to 250
)

print()
print(f"First movie review AFTER preprocessing")
print(decode_review(train_data[0]))

model_path = os.path.join(os.path.dirname(__file__), "text_classification_model.h5")
if os.path.isfile(model_path):
    model = keras.models.load_model(model_path)
else:
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=max(inverted_word_index)
                + 1,  # number of neurons = number of words in the vocabulary
                output_dim=16,  # assign a 16-dimensional vector to each word's index
            ),  # output shape: [batch_size, 250, 16], so each movie review has 250 arrays of 16 vectors each
            # output shape: [batch_size, 16], so each movie review has an array of 16 vectors
            # which are averages along the axis with 250 words
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(units=16, activation="relu"),
            keras.layers.Dense(units=1, activation="sigmoid"),  # sigmoid - output 0-1
        ]
    )

    model.compile(
        optimizer="adam",  # Adaptive Moment Estimation
        loss="binary_crossentropy",  # used for 0-1 outputs, since we have "sigmoid" activation in the output layer
        metrics=["accuracy"],  # measures the percentage of correctly classified samples
    )

    x_val = train_data[:10000]
    x_train = train_data[10000:]
    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fit_model = model.fit(
        x=x_train,
        y=y_train,
        epochs=20,
        batch_size=512,
        validation_data=(
            x_val,
            y_val,
        ),  # the model needs some data to validate its accuracy during training cycles
        verbose=1,
    )

    model.save(model_path)

results = model.evaluate(test_data, test_labels)
print(results)
