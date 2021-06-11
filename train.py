from argparse import ArgumentParser
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard
from vit import VisionTransformer
import math
import os, sys
import pathlib
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import models, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Normalization
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(f'Running on Python {sys.version}, Tensorflow {tf.__version__}.')

# Data loading
seed = 69
sample_rate = 16000
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('s1_release')
labels = np.array(tf.io.gfile.listdir(str(data_dir)))
num_labels = len(labels)
print('Commands:', labels)

# load given train set
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:', len(tf.io.gfile.listdir(str(data_dir/labels[0]))))
print('Example file tensor:', filenames[0])

train_files = filenames[:round(num_samples*0.8)]  # first 80%
val_files = filenames[round(num_samples*0.8):]  # last 20%

print('Training set size', len(train_files))
print('Validation set size', len(val_files))

# load given test set
data_dir = pathlib.Path('s1_test_release')
test_files = tf.io.gfile.glob(str(data_dir) + '/*')  # provided
print('Test set size', len(test_files))


def preprocess(file_path):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
    waveform = tf.squeeze(audio, axis=-1)

    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([sample_rate] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    # Choose frame_length and frame_step parameters such that the generated spectrogram "image" is almost square.
    spectrogram = tf.signal.stft(equal_length, fft_length=1024, frame_length=1001, frame_step=31, pad_end=True)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.grayscale_to_rgb(spectrogram)  # RGB if using pretrained network that is trained on RGB images e.g. ImageNet, else can gray scale

    # get label_id
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label_id = tf.argmax(label == labels)

    spectrogram.set_shape([517, 513, 3])
    label_id.set_shape([])
    return spectrogram, label_id


def preprocess_ds(files):
    return tf.data.Dataset.from_tensor_slices(files).map(preprocess, num_parallel_calls=AUTOTUNE)


train_ds = preprocess_ds(train_files)
val_ds = preprocess_ds(val_files)

parser = ArgumentParser()
parser.add_argument("--logdir", default="logs")
parser.add_argument("--image-size", default=32, type=int)
parser.add_argument("--patch-size", default=4, type=int)
parser.add_argument("--num-layers", default=8, type=int)
parser.add_argument("--d-model", default=64, type=int)
parser.add_argument("--num-heads", default=4, type=int)
parser.add_argument("--mlp-dim", default=128, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--epochs", default=50, type=int)
args = parser.parse_args()

# Training
batch_size = 16
train_ds = train_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.cache().batch(batch_size).prefetch(AUTOTUNE)

model = VisionTransformer(
    image_size=args.image_size,
    patch_size=args.patch_size,
    num_layers=args.num_layers,
    num_classes=num_labels,
    d_model=args.d_model,
    num_heads=args.num_heads,
    mlp_dim=args.mlp_dim,
    channels=3,
    dropout=0.1,
)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
              metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1,
                                     mode='auto', baseline=None, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('./checkpoint', monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=4, verbose=1)
]
history = model.fit(train_ds, validation_data=val_ds,  epochs=args.epochs, callbacks=callbacks, use_multiprocessing=True, verbose=1)
plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, to_file='SC1_ViT.png')
model.summary()

# Generate prediction csv
to_predict_ds = preprocess_ds(list(map(str, test_files)))
i = 0
predicted_labels, filenames = [], []
for spectrogram, label in to_predict_ds.batch(1):
    filenames.append(os.path.basename(test_files[i]))
    prediction = model(spectrogram)
    prediction_value = tf.nn.softmax(prediction[0]).numpy()
    predicted_label = labels[np.argmax(prediction_value)]
    predicted_labels.append(predicted_label)
    i += 1

df = pd.DataFrame(list(zip(filenames, predicted_labels)))
df.to_csv('challenge_2_team_Tensor is not flowing.csv', index=False, header=False)  # tested submission file format passed

model.save('sc1_ViT')
