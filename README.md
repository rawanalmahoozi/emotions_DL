# Facial Emotion Recognition with Neural Networks

A deep learning lab project that classifies a facial image into one of
seven emotion categories. The project includes dataset exploration and
preprocessing, two baseline neural networks, test-set
evaluation and a Gradio web app with image upload and webcam input.

## Project Overview

**Task:** Multiclass facial emotion classification\
**Input:** A face image\
**Output:** Predicted scores for seven emotion classes: - angry -
disgust - fear - happy - neutral - sad - surprise

The goal is to demonstrate an end-to-end image-classification workflow,
from preparing the data and training models to evaluating them and
deploying an interactive demo.

## Repository Structure

The project is organised  as follows:

``` text
.
├── DL_emotions.ipynb
├── emotions_pred.py
├── emotions_model.keras
├── emotions/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   └── ...
│   └── test/
│       ├── angry/
│       ├── disgust/
│       └── ...
└── README.md
```


## Dataset and Preprocessing

The notebook loads images from `emotions/train` and `emotions/test`
using Keras' `image_dataset_from_directory` utility.

Preprocessing steps: 1. Resize images to **48 × 48** pixels. 2. Convert
image pixel values to the **0--1** range by dividing by 255. 3. Use the
training data for model fitting and a validation split during training.
4. Keep the test dataset separate for final evaluation.



> **Dataset note:** The notebook expects the dataset to be available
> locally in the `emotions/` directory. 

## Models

Two neural-network baselines are defined in the notebook.

### Model 1

-   Flatten layer for 48 × 48 × 3 image input
-   Dense layer with 128 units and ReLU activation
-   Output layer with 7 units and softmax activation

### Model 2

-   Flatten layer for 48 × 48 × 3 image input
-   Dense layer with 256 units and ReLU activation
-   Dense layer with 128 units and ReLU activation
-   Dense layer with 64 units and ReLU activation
-   Output layer with 7 units and softmax activation

Both models use sparse categorical cross-entropy as the loss function
and track accuracy. The notebook also defines early stopping based on
validation loss. Model2 computes class weights to help address class
imbalance. 
Model 1 gave a better accuracy score but Model 2 was used at the end because of the confusion matrix.

The notebook saves the second model as `emotions_model.keras`, which is
the filename used by the app.

## Evaluation

The notebook evaluates predictions on the held-out test set using a
classification report and confusion matrix. The report includes
per-class precision, recall and F1-score, alongside support counts.




## Run the Gradio App

### 1. Install dependencies

Use a Python environment compatible with your installed Keras/TensorFlow
version. Install the packages used by the app:

``` bash
pip install gradio numpy keras
```


``` bash
pip install tensorflow
```

### 2. Check the model file

Place `emotions_model.keras` in the same directory as `emotions_pred.py`, or
update the model path in the code.

### 3. Start the app

``` bash
python app.py
```

Gradio will print a local URL in the terminal. Open it in a browser to
use the interface.

### 4. Use the app

-   Upload a face image, or choose the webcam input.
-   Capture or select an image.
-   The app returns scores for the seven emotion classes.
-   The label output displays the top classes and their scores.

## Application Code

The app code loads the saved Keras model, converts an input image to
RGB, resizes it to 48 × 48, normalizes pixel values, and returns a score
for each emotion class.


## Limitations and Ethical Considerations

- The model predicts visual patterns associated with dataset labels, it cannot reliably
    determine what a person is actually feeling.
-   **Lighting and image quality:** Low light, blur, occlusion, camera
    quality and unusual angles may reduce prediction quality.
-   **Generalisation and bias:** Results depend on the dataset's
    composition and labels. Performance may differ across demographics,
    environments and expressions that are underrepresented in training
    data.
-   **Ambiguous expressions:** Facial expressions can overlap, and a
    single image may not clearly indicate one category.
-   **Where the model fails** The model mostly fails when it comes to anger, disgust and fear. 

## Future Improvements

Possible extensions include image augmentation, testing different
architectures, tuning hyperparameters, A convolutional
neural network (CNN) would also be a useful image-focused architecture
to compare against the dense baselines.
