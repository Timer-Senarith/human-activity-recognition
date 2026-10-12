# Human Activity Recognition with Neural Networks

This University of Bath coursework project compares three neural network configurations for classifying human activities using smartphone sensor features from the UCI Human Activity Recognition dataset.

The analysis examines how two feedforward networks and an LSTM perform on six activity classes: walking, walking upstairs, walking downstairs, sitting, standing, and lying.

**Author:** Timer Senarith  
**Course:** ME 52081 — Autonomy AI/ML, University of Bath

## Tools and data

- Python
- TensorFlow and Keras
- pandas and NumPy
- scikit-learn
- Matplotlib and seaborn

The dataset contains 10,299 observations from 30 participants, with 561 precomputed features derived from accelerometer and gyroscope measurements. The code uses these supplied features rather than extracting features from raw sensor signals.

## Project code

[CW.py](CW.py) contains data loading, preprocessing, model training, evaluation, and visualization.

## Analysis workflow

- Load the supplied training and test files and combine them.
- Create stratified random splits: 70% training, 15% validation, and 15% testing.
- Fit a standard scaler on the training data and apply it to the validation and test sets.
- Encode the six activity classes as integers from 0 to 5.
- Train three neural network configurations for 20 epochs with a batch size of 64.
- Compare validation accuracy and per-class test F1 scores.
- Examine confusion matrices for the two feedforward models.
- Explore selected feature distributions and correlations.

## Models

| Model label in code | Architecture |
|---|---|
| ANN | Dense layers with 256 and 128 units, with dropout |
| MLP | Dense layers with 512, 256, and 128 units, with dropout |
| LSTM | An LSTM layer with 128 units, dropout, and a dense layer with 64 units |

All models use a six-class softmax output, the Adam optimizer, and sparse categorical cross-entropy loss.

The ANN and MLP labels distinguish two feedforward architectures; both are multilayer perceptrons.

## Analytical focus

The comparison considers overall validation performance alongside class-level errors. Per-class F1 scores and confusion matrices help identify activities that are harder to distinguish, rather than relying on accuracy alone.

## Run the analysis

1. Obtain and extract the UCI HAR Dataset.
2. Preserve its features.txt, activity_labels.txt, train, and test structure.
3. Update the base_path argument in load_data() within CW.py to point to your extracted dataset directory.
4. Install the dependencies:

       python3 -m pip install pandas numpy scikit-learn matplotlib seaborn tensorflow

5. Run the script:

       python3 CW.py

The script trains the models and displays plots. Close each plot window to continue if execution pauses. The dataset is not included in this repository.

## Limitations

The original training and test partitions are combined before a new random split. Participants may therefore appear in multiple subsets. This evaluation does not establish performance on entirely unseen participants.

The LSTM input has a single timestep containing the extracted features. It does not model activity progression across a sequence of sensor observations.

Training randomness is not fully controlled, so results can vary between runs. The comparison is an academic experiment rather than evidence of deployment readiness.

## Skills demonstrated

Data preparation, stratified splitting, training-only feature scaling, neural network construction, multiclass evaluation, visualization, and critical interpretation of model limitations.
