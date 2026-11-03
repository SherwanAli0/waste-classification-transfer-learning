# Waste Classification Using Transfer Learning with VGG16
# IBM AI Engineering Professional Certificate
# Module 3: Building Deep Learning Models with TensorFlow
#
# Run from the directory that should hold the dataset and the saved models
# (normally the repository root):  python waste_classification.py
# Importing this module has no side effects; everything happens in main().

import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib.image import imread
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from sklearn import metrics
import warnings

import requests
import zipfile
from tqdm import tqdm

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
file_name = "o-vs-r-split-reduced-1200.zip"
data_dir = 'o-vs-r-split'

# Same extensions Keras' flow_from_directory accepts.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.tif', '.tiff')
# The manual classification report uses only the first N images of each test class.
EVAL_PER_CLASS = 50


def dataset_present(root='.'):
    """True when the extracted train/ and test/ folders already exist under root."""
    return all(os.path.isdir(os.path.join(root, data_dir, split)) for split in ('train', 'test'))


def download_file(download_url, dest, timeout=60):
    """Stream download_url to dest. Never leaves a partial file at dest."""
    part = dest + '.part'
    try:
        with requests.get(download_url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(part, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.replace(part, dest)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise


def extract_file_with_progress(file_name, dest='.'):
    print("Extracting dataset...")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        members = zip_ref.infolist()
        with tqdm(total=len(members), unit='file') as progress_bar:
            for member in members:
                zip_ref.extract(member, dest)
                progress_bar.update(1)
    print("Dataset extraction complete")


def ensure_dataset(root='.'):
    """Download and extract the dataset under root unless it is already there."""
    if dataset_present(root):
        print(f"Dataset already present in '{os.path.join(root, data_dir)}'; skipping download.")
        return
    zip_path = os.path.join(root, file_name)
    print("Downloading dataset...")
    download_file(url, zip_path)
    extract_file_with_progress(zip_path, root)
    os.remove(zip_path)
    if not dataset_present(root):
        raise RuntimeError(
            f"Extraction finished but '{data_dir}/train' and '{data_dir}/test' were not found under '{root}'."
        )


def list_class_images(class_dir):
    """Image files directly inside class_dir, in sorted filename order.

    glob/os.listdir order is filesystem-dependent; Keras' flow_from_directory
    (and therefore test_generator) sorts filenames, so the manual evaluation does too.
    """
    return sorted(
        p for p in glob.glob(os.path.join(class_dir, '*'))
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTENSIONS)
    )


def select_eval_files(test_root, per_class=EVAL_PER_CLASS):
    """First `per_class` images of class O, then of class R (sorted filename order).

    Returns (files, n_available) where n_available is the number of test images
    found in both class folders, so the size of the subset can be reported.
    """
    files_O = list_class_images(os.path.join(test_root, 'O'))
    files_R = list_class_images(os.path.join(test_root, 'R'))
    return files_O[:per_class] + files_R[:per_class], len(files_O) + len(files_R)


class2num_lt = lambda l: [0 if x == 'O' else 1 for x in l]


def num2class_lt(l, threshold=0.5):
    """Sigmoid outputs (probability of class R, index 1) -> 'O' / 'R' labels."""
    return ['O' if p < threshold else 'R' for p in np.ravel(l)]


def exp_decay(epoch):
    initial_lrate = 1e-4
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate


class LossHistory_(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(epoch))
        print('Learning rate:', exp_decay(len(self.losses)))


def main():
    warnings.filterwarnings('ignore')

    print("TensorFlow version:", tf.__version__)

    ensure_dataset()

    img_rows, img_cols = 150, 150
    batch_size = 32
    n_epochs = 10
    n_classes = 2
    val_split = 0.2
    verbosity = 1
    path = data_dir + '/train/'
    path_test = data_dir + '/test/'
    input_shape = (img_rows, img_cols, 3)
    labels = ['O', 'R']
    seed = 42

    train_datagen = ImageDataGenerator(
        validation_split=val_split,
        rescale=1.0/255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        validation_split=val_split,
        rescale=1.0/255.0,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0
    )

    train_generator = train_datagen.flow_from_directory(
        directory=path,
        seed=seed,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        target_size=(img_rows, img_cols),
        subset='training'
    )

    val_generator = val_datagen.flow_from_directory(
        directory=path,
        seed=seed,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        target_size=(img_rows, img_cols),
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        directory=path_test,
        class_mode='binary',
        seed=seed,
        batch_size=batch_size,
        shuffle=False,
        target_size=(img_rows, img_cols)
    )

    # The manual evaluation below encodes O -> 0 and R -> 1; make sure Keras agrees.
    if test_generator.class_indices != {'O': 0, 'R': 1}:
        raise RuntimeError(f"Unexpected class indices: {test_generator.class_indices}")

    print("Training generator length:", len(train_generator))

    IMG_DIM = (100, 100)
    train_files = glob.glob('./o-vs-r-split/train/O/*')
    train_files = train_files[:20]
    train_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in train_files]
    train_imgs = np.array(train_imgs)
    train_labels = [Path(fn).parent.name for fn in train_files]

    img_id = 0
    O_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1], batch_size=1)
    O = [next(O_generator) for i in range(0,5)]

    fig, ax = plt.subplots(1,5, figsize=(16, 6))
    print('Sample augmented images:', [item[1][0] for item in O])
    l = [ax[i].imshow(O[i][0][0]) for i in range(0,5)]
    plt.suptitle("Data Augmentation Examples")
    plt.show()

    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    output = vgg.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    basemodel = Model(vgg.input, output)

    for layer in basemodel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(basemodel)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )

    checkpoint_path = 'O_R_tlearn_vgg16.keras'

    loss_history_ = LossHistory_()
    lrate_ = LearningRateScheduler(exp_decay)

    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, mode='min', min_delta=0.01),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]

    callbacks_list_ = [loss_history_, lrate_] + keras_callbacks

    print("Training extract features model...")
    extract_feat_model = model.fit(
        train_generator,
        steps_per_epoch=5,
        epochs=10,
        callbacks=callbacks_list_,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1
    )

    history = extract_feat_model

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve - Extract Features Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve - Extract Features Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    output = vgg.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    basemodel = Model(vgg.input, output)

    for layer in basemodel.layers:
        layer.trainable = False

    set_trainable = False
    for layer in basemodel.layers:
        if layer.name in ['block5_conv3']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    print("\nLayer trainability status:")
    for layer in basemodel.layers:
        print(f"{layer.name}: {layer.trainable}")

    model = Sequential()
    model.add(basemodel)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    checkpoint_path = 'O_R_tlearn_fine_tune_vgg16.keras'
    loss_history_ = LossHistory_()
    lrate_ = LearningRateScheduler(exp_decay)

    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, mode='min', min_delta=0.01),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]

    callbacks_list_ = [loss_history_, lrate_] + keras_callbacks

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )

    print("Training fine-tuned model...")
    fine_tune_model = model.fit(
        train_generator,
        steps_per_epoch=5,
        epochs=10,
        callbacks=callbacks_list_,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1
    )

    history = fine_tune_model

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve - Fine-Tuned Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve - Fine-Tuned Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Loading saved models for evaluation...")
    extract_feat_model = tf.keras.models.load_model('O_R_tlearn_vgg16.keras')
    fine_tune_model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')

    IMG_DIM = (150, 150)
    test_files, n_test_available = select_eval_files(path_test)
    print(f"\nEvaluation subset: {len(test_files)} of {n_test_available} test images "
          f"(first {EVAL_PER_CLASS} per class, sorted filename order). "
          "This is not the full test set.")
    test_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in test_files]
    test_imgs = np.array(test_imgs)
    test_labels = [Path(fn).parent.name for fn in test_files]

    test_imgs_scaled = test_imgs.astype('float32')
    test_imgs_scaled /= 255

    test_labels_enc = class2num_lt(test_labels)

    predictions_extract_feat_model = extract_feat_model.predict(test_imgs_scaled, verbose=0)
    predictions_fine_tune_model = fine_tune_model.predict(test_imgs_scaled, verbose=0)

    predictions_extract_feat_model = num2class_lt(predictions_extract_feat_model)
    predictions_fine_tune_model = num2class_lt(predictions_fine_tune_model)

    for preds in (predictions_extract_feat_model, predictions_fine_tune_model):
        if len(preds) != len(test_labels):
            raise RuntimeError(f"{len(preds)} predictions for {len(test_labels)} test images")

    print('\nExtract Features Model Classification Report:')
    print('=' * 50)
    print(metrics.classification_report(test_labels, predictions_extract_feat_model))

    print('\nFine-Tuned Model Classification Report:')
    print('=' * 50)
    print(metrics.classification_report(test_labels, predictions_fine_tune_model))

    def plot_image_with_title(image, model_name, actual_label, predicted_label):
        plt.figure(figsize=(6, 4))
        plt.imshow(image)
        plt.title(f"Model: {model_name}\nActual: {actual_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

    index_to_plot = 1

    plot_image_with_title(
        image=test_imgs[index_to_plot].astype('uint8'),
        model_name='Extract Features Model',
        actual_label=test_labels[index_to_plot],
        predicted_label=predictions_extract_feat_model[index_to_plot]
    )

    plot_image_with_title(
        image=test_imgs[index_to_plot].astype('uint8'),
        model_name='Fine-Tuned Model',
        actual_label=test_labels[index_to_plot],
        predicted_label=predictions_fine_tune_model[index_to_plot]
    )

    print("\nProject completed successfully!")
    print(f"Models trained; evaluated on a {len(test_files)}-image subset of the test set "
          f"({n_test_available} images available).")


if __name__ == "__main__":
    main()
