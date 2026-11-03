# Waste Classification Using Transfer Learning

**Final Project for the IBM AI Engineering Professional Certificate (Coursera)**
**Module 3: Building Deep Learning Models with TensorFlow**

This is coursework. It uses transfer learning with VGG16 to classify waste images as organic (`O`) or recyclable (`R`). The task structure (10 required tasks), the scenario text and the dataset archive come from the course; this repository holds my implementation of those tasks. It is an educational exercise, not a deployed or validated system.

## Project Overview

Scenario (from the course brief): a company called EcoClean lacks an efficient and scalable way to automate waste sorting, and the project asks for a model that can separate recyclable from organic waste in images.

### Aim
Build a binary image classifier for organic vs recyclable waste using transfer learning, and compare a feature-extraction model with a partially fine-tuned one.

## Course Information

- **Program**: IBM AI Engineering Professional Certificate
- **Module**: Module 3 - Building Deep Learning Models with TensorFlow
- **Platform**: Coursera
- **Project Type**: Final Project with 10 Required Tasks

## Dataset

- **Source**: a course-hosted archive, `o-vs-r-split-reduced-1200.zip` (about 11 MB), downloaded from the URL in `waste_classification.py`. Its upstream origin and licence are not documented in this repository.
- **Categories**: 2 classes. Keras assigns class indices alphabetically, so Organic (`O`) = 0 and Recyclable (`R`) = 1.
- **Structure** (created in the current working directory when the script extracts the archive):
  ```
  o-vs-r-split/
  ├── train/
  │   ├── O/ (Organic waste images)
  │   └── R/ (Recyclable waste images)
  └── test/
      ├── O/ (Organic test images)
      └── R/ (Recyclable test images)
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

The project is structured around 10 tasks required by the course. The screenshots are from the original notebook run (see "Results and what is verified").

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
- **Normalization**: pixel values divided by 255 for training, validation, test and the manual evaluation, so training and inference preprocessing match. VGG16's own `preprocess_input` is not used; that is the course protocol, and changing it would require retraining.
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

- No test-set metrics are recorded in this repository.
- The screenshots come from the original Jupyter notebook run (TensorFlow 2.17.0). The accuracy curves show validation accuracy of roughly 0.83 (feature extraction) and roughly 0.9 (fine-tuned) at epoch 10, read by eye from the plots. These are historical figures; they have not been reproduced from `waste_classification.py`, and no saved models are included.
- This repository does not record an end-to-end run (download, training, plots, evaluation) of `waste_classification.py` itself, only of the original notebook. What has been checked for the script, using test doubles and no dataset, is listed under "Checks that need no data".

### Evaluation scope

The classification reports printed at the end are **not** computed on the whole test set. The script takes the first 50 images of `test/O` and the first 50 of `test/R` (images sorted by filename), 100 images in total, and prints the subset size and the number of test images available. The notebook run reported `Found 200 images belonging to 2 classes` for the test folder (see the Task 2 screenshot), so the subset is about half of it. `test_generator` is created for Task 2 but is not used for evaluation.

Earlier versions picked the "first 50" in the operating system's directory-listing order, which varies between filesystems. Sorting makes the subset reproducible, but it can select different images than the original notebook run did, so the sample image in Tasks 9 and 10 and any per-class numbers may differ from the historical screenshots.

## Installation and Usage

### Requirements
```bash
pip install -r requirements.txt
```
The original run used TensorFlow 2.17.0. `requirements.txt` allows any `tensorflow>=2.10.0`, but no other version has been tested with this script. Internet access is needed for the dataset and for Keras to fetch the ImageNet VGG16 weights.

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
├── waste_classification.py    # Main implementation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT licence
├── Screenshots/               # Output screenshots from the original notebook run
├── tests/                     # Tests that need no TensorFlow, data or network
└── (generated in the working directory when you run the script)
    ├── o-vs-r-split/                    # downloaded dataset
    ├── O_R_tlearn_vgg16.keras           # best feature-extraction checkpoint
    └── O_R_tlearn_fine_tune_vgg16.keras # best fine-tuned checkpoint
```

## What this project covers

- Used VGG16 transfer learning for binary image classification
- Prepared image data with Keras generators (rescaling, augmentation, validation split)
- Trained a feature-extraction model and a model with the last convolutional layer unfrozen
- Produced classification reports on a 100-image test subset and visualised sample predictions

## Limitations

- Educational scope: nothing here shows the models work in a sorting facility, a smart bin or any other real setting.
- Short training (10 epochs of 5 steps), no repeated runs, and no comparison of the two models on the full test set, so no claim is made that fine-tuning improved performance.
- Only two classes, from a reduced course dataset whose upstream source is not documented here.
- Evaluation covers a 100-image subset; the historical figures above are unverified.

## License

The code in this repository is released under the [MIT License](LICENSE). The dataset archive and any course-provided material are not covered by it and remain under their owners' terms.

## Author

Sherwan Ali (SherwanAli0). Developed as part of the IBM AI Engineering Professional Certificate coursework - Module 3: Building Deep Learning Models with TensorFlow.
