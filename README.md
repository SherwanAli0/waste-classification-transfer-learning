# Waste Classification Using Transfer Learning

**IBM AI Engineering Certificate Final Project**

I completed this project myself as the final project of the IBM AI Engineering Professional Certificate. The course supplied the dataset and the task structure (10 tasks); the implementation in this repository is my own work. It uses transfer learning with VGG16 to classify waste images as organic (`O`) or recyclable (`R`). It is an educational exercise, not a deployed or validated system.

## Project Overview

### Aim
Build a binary image classifier for organic vs recyclable waste using transfer learning, and compare a feature-extraction model with a partially fine-tuned one.

## Course Information

- **Program**: IBM AI Engineering Professional Certificate
- **Platform**: Coursera
- **Project Type**: Final project with 10 tasks

## Dataset

- **Source**: dataset supplied by the course, the archive `o-vs-r-split-reduced-1200.zip` (about 11 MB), downloaded from the URL in `waste_classification.py`. Its upstream origin and licence are not documented in this repository.
- **Categories**: 2 classes. Keras assigns class indices alphabetically, so Organic (`O`) = 0 and Recyclable (`R`) = 1.
- **Structure** (created in the current working directory when the script extracts the archive):
  ```
  o-vs-r-split/
  +-- train/
  |   +-- O/ (Organic waste images)
  |   +-- R/ (Recyclable waste images)
  +-- test/
      +-- O/ (Organic test images)
      +-- R/ (Recyclable test images)
  ```
- The archive and the images are not stored in this repository. The script downloads them only if `o-vs-r-split/train` and `o-vs-r-split/test` are missing; delete `o-vs-r-split/` to force a fresh download.

## Technical Approach

### Transfer Learning Strategy
1. **Base Model**: VGG16 pre-trained on ImageNet.
2. **Feature Extraction**: all VGG16 layers frozen, new dense classifier trained on top.
3. **Fine-Tuning**: same layout, but the VGG16 layers from `block5_conv3` onward are trainable. In practice this means only `block5_conv3` has trainable weights (`block5_pool` and `Flatten` have none). The dense head is trained from scratch in both models; the fine-tuned model does not start from the feature-extraction model's weights.

### Model Architecture
- **Input**: 150x150x3 RGB images
- **Base**: VGG16 (frozen / last conv layer unfrozen), flattened to 8192 features
- **Classifier**:
  - Dense(512) + ReLU + Dropout(0.3)
  - Dense(512) + ReLU + Dropout(0.3)
  - Dense(1) + Sigmoid (output is the probability of class `R`)

## Implementation Tasks

The project is structured around 10 tasks. The screenshots are from the original notebook run (see "Results and what is verified").

| Task | Description | Screenshot |
|---|---|---|
| 1 | Print TensorFlow version | [tensorflow_version.png](Screenshots/tensorflow_version.png) |
| 2 | Create test data generator | [test_generator.png](Screenshots/test_generator.png) |
| 3 | Print training generator length | [train_generator_len.png](Screenshots/train_generator_len.png) |
| 4 | Display model summary | [model_summary.png](Screenshots/model_summary.png) |
| 5 | Compile the model | [model_compile.png](Screenshots/model_compile.png) |
| 6 | Plot accuracy curves (extract features model) | [plot_accuracy_curve.png](Screenshots/plot_accuracy_curve.png) |
| 7 | Plot loss curves (fine-tuned model) | [plot_loss_curve.png](Screenshots/plot_loss_curve.png) |
| 8 | Plot accuracy curves (fine-tuned model) | [plot_finetune_model.png](Screenshots/plot_finetune_model.png) |
| 9 | Plot a test image with the extract features model's prediction | [extract_features_model.png](Screenshots/extract_features_model.png) |
| 10 | Plot a test image with the fine-tuned model's prediction | [finetuned_model.png](Screenshots/finetuned_model.png) |

## Key Features

### Data Processing
- **Image Augmentation** (training generator only): width/height shifts of 10%, horizontal flips
- **Normalization**: pixel values divided by 255 for training, validation, test and the manual evaluation, so training and inference preprocessing match. VGG16's own `preprocess_input` is not used; that is the preprocessing this project was built with, and changing it would require retraining.
- **Validation split**: 20% of `train/`, taken by a separate generator without augmentation. Keras assigns files to the split by sorted filename (not randomly), so training and validation images do not overlap.
- **Batch size**: 32

### Training Configuration
- **Epochs**: up to 10, with `steps_per_epoch=5` (5 batches of 32, about 160 augmented training images per epoch), so each run is a short one
- **Optimizer**: RMSprop (learning rate 1e-4) with an exponential-decay `LearningRateScheduler`
- **Loss Function**: Binary crossentropy
- **Random seed**: `seed=42` is passed to the data generators only; TensorFlow's own seed is not set, so reruns will not reproduce the same weights

### Callbacks
- **Early Stopping**: patience=4, monitoring validation loss
- **Model Checkpoint**: save the best model by validation loss
- **Learning Rate Scheduler**: exponential decay

## Results and what is verified

- **Historical (original notebook run, TensorFlow 2.17.0):** the screenshots show validation accuracy of roughly 0.83 (feature extraction) and roughly 0.9 (fine-tuned) at epoch 10, read by eye from the plots. No test-set metrics from that run were recorded, and no saved models are included. These figures are unverified.
- **Measured (re-run on 2026-11-03):** see the section below. It is one unseeded run of the unchanged `waste_classification.py`, so it is a measurement, not a reproduction of the historical figures.
- What has been checked without TensorFlow or the dataset, using test doubles, is listed under "Checks that need no data".

### Evaluation scope

The classification reports printed at the end are **not** computed on the whole test set. The script takes the first 50 images of `test/O` and the first 50 of `test/R` (images sorted by filename), 100 images in total, and prints the subset size and the number of test images available. The test folder holds 200 images (100 `O`, 100 `R`; the Task 2 screenshot and the 2026-11-03 re-run both report `Found 200 images belonging to 2 classes`), so the subset is half of it. `test_generator` is created for Task 2 but is not used for evaluation.

Earlier versions picked the "first 50" in the operating system's directory-listing order, which varies between filesystems. Sorting makes the subset reproducible, but it can select different images than the original notebook run did, so the sample image in Tasks 9 and 10 and any per-class numbers may differ from the historical screenshots.

## Re-run on 2026-11-03

I re-ran the unchanged `waste_classification.py` once to replace unverifiable figures with measured ones. Preprocessing, model and hyperparameters were not changed. The run used a scratch copy of the repository, so the downloaded dataset, the VGG16 weights and the saved models are not part of this repository.

- **Environment**: Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, scikit-learn 1.9.1; Windows 11, CPU only (12 logical cores; GPU is not used on native Windows for TensorFlow 2.11 and later).
- **Downloads**: the dataset archive from the URL in the script, and the standard Keras VGG16 ImageNet weights (`vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`).
- **Data**: 800 training, 200 validation and 200 test images (100 `O` and 100 `R` in the test folder); training generator length 25.
- **Seeds**: the script sets none for TensorFlow, NumPy or Python (`seed=42` goes to the data generators only). Results vary from run to run, so these numbers are one sample, not expected values.
- **Runtime**: `main()` took 692 s (about 11.5 min); the whole scratch run, including imports and the extra full-test evaluation below, took 744 s.
- **Early stopping** did not trigger. The evaluated models are the best-validation-loss checkpoints: epoch 10 (val loss 0.334) for the extract features model and epoch 7 (val loss 0.274) for the fine-tuned model.

### Validation accuracy per epoch

Validation uses `val_generator.samples // 32 = 6` batches, i.e. 192 of the 200 validation images, with shuffling on.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Extract features | 0.531 | 0.734 | 0.734 | 0.844 | 0.828 | 0.849 | 0.802 | 0.823 | 0.844 | 0.849 |
| Fine-tuned | 0.828 | 0.828 | 0.776 | 0.781 | 0.849 | 0.875 | 0.891 | 0.885 | 0.875 | 0.865 |

### Test results (accuracy, same inference path in both rows)

The inference path is the one in `main()`: files in sorted filename order, `load_img` at 150x150, divide by 255, `predict`, threshold 0.5. The full-test numbers come from the same functions with no per-class cap; they are reported separately from the 100-image subset the script prints.

| Evaluated on | Extract features | Fine-tuned |
|---|---|---|
| First 50 per class (100 images, what the script prints) | 0.80 | 0.83 |
| Full test directory (200 images) | 0.84 | 0.86 |

Full test directory, per class (precision / recall / F1, 100 images each):

| Model | `O` | `R` | Confusion matrix (rows true `O`, `R`; columns predicted `O`, `R`) |
|---|---|---|---|
| Extract features | 0.80 / 0.91 / 0.85 | 0.90 / 0.77 / 0.83 | [[91, 9], [23, 77]] |
| Fine-tuned | 0.85 / 0.88 / 0.86 | 0.88 / 0.84 / 0.86 | [[88, 12], [16, 84]] |

First 50 per class (100 images), per class (precision / recall / F1, 50 images each):

| Model | `O` | `R` |
|---|---|---|
| Extract features | 0.74 / 0.92 / 0.82 | 0.89 / 0.68 / 0.77 |
| Fine-tuned | 0.80 / 0.88 / 0.84 | 0.87 / 0.78 / 0.82 |

### Compared with the historical screenshots

| | Historical (screenshots, read by eye) | Re-run, epoch 10 | Re-run, best-val-loss checkpoint |
|---|---|---|---|
| Extract features, validation accuracy | about 0.83 | 0.849 | 0.849 (epoch 10) |
| Fine-tuned, validation accuracy | about 0.9 | 0.865 | 0.891 (epoch 7) |

The historical test-set reports were not recorded, so there is nothing to compare the test numbers against. The two models differ by 4 of 200 full-test images in this single unseeded run, which is not enough to claim that fine-tuning improves accuracy.

## Installation and Usage

### Requirements
```bash
pip install -r requirements.txt
```
The original notebook run used TensorFlow 2.17.0 and the 2026-11-03 re-run used TensorFlow 2.21.0 with Keras 3.15.1 (Python 3.11.9). `requirements.txt` allows any `tensorflow>=2.10.0`, but no other version has been tested with this script. On CPU the full script took about 12 minutes. Internet access is needed for the dataset and for Keras to fetch the ImageNet VGG16 weights.

### Running the Project
Run from the directory where the dataset and the saved models should be written (normally the repository root); all paths are relative to the current working directory.
```bash
python waste_classification.py
```

The script will:
1. Download and extract the dataset if it is not already there
2. Train both models (extract features + fine-tuned)
3. Show loss/accuracy curves (matplotlib windows; close each window to continue)
4. Evaluate both models on the 100-image test subset described above
5. Display one sample prediction per model

Importing `waste_classification` does nothing; all work happens in `main()`.

### Checks that need no data
```bash
pip install pytest
python -m pytest
```
These tests replace TensorFlow with test doubles and never touch the network. They cover the dataset download/skip logic, the choice of evaluation files, the label/threshold mapping and the absence of import side effects. They do not train or evaluate a model.

## File Structure

```
+-- waste_classification.py    # Main implementation
+-- requirements.txt           # Python dependencies
+-- README.md                  # This file
+-- LICENSE                    # MIT licence
+-- Screenshots/               # Output screenshots from the original notebook run
+-- tests/                     # Tests that need no TensorFlow, data or network
+-- (generated in the working directory when you run the script)
    +-- o-vs-r-split/                    # downloaded dataset
    +-- O_R_tlearn_vgg16.keras           # best feature-extraction checkpoint
    +-- O_R_tlearn_fine_tune_vgg16.keras # best fine-tuned checkpoint
```

## What this project covers

- Used VGG16 transfer learning for binary image classification
- Prepared image data with Keras generators (rescaling, augmentation, validation split)
- Trained a feature-extraction model and a model with the last convolutional layer unfrozen
- Produced classification reports on a 100-image test subset and visualised sample predictions

## Limitations

- Educational scope: nothing here shows the models work in a sorting facility, a smart bin or any other real setting.
- Short training (10 epochs of 5 steps), no seeds, and a single measured run, so the numbers vary between runs; no claim is made that fine-tuning improved performance.
- Only two classes, from a reduced dataset supplied by the course, whose upstream source is not documented here.
- The script's own report covers a 100-image subset of the 200 test images; the full-test numbers in the re-run section came from a separate evaluation with the same inference path. The historical screenshot figures are unverified.

## License

The code in this repository is released under the [MIT License](LICENSE). The dataset, any course-provided material and the pre-trained VGG16 ImageNet weights are not covered by it and have their own terms.

## Author

Sherwan Ali (SherwanAli0). Completed as the final project of the IBM AI Engineering Professional Certificate.
