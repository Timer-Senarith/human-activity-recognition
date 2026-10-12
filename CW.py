import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(base_path="/Users/timersenarith/Desktop/Bath Masters/ME 52081 Autonomy AI:ML/Coursework/Code/Dataset - humanADLs/UCI HAR Dataset"):

    features_path = os.path.join(base_path, "features.txt")
    feature_names = pd.read_csv(features_path, sep=r"\s+", header=None)[1].tolist()

    activity_labels_path = os.path.join(base_path, "activity_labels.txt")
    activity_labels = pd.read_csv(activity_labels_path,
                                  sep=r"\s+", header=None,
                                  index_col=0)[1].to_dict()

    X_train = pd.read_csv(os.path.join(base_path, "train", "X_train.txt"),
                          sep=r"\s+", header=None)
    X_train.columns = feature_names

    y_train = pd.read_csv(os.path.join(base_path, "train", "y_train.txt"),
                          sep=r"\s+", header=None)[0].map(activity_labels)

    X_test = pd.read_csv(os.path.join(base_path, "test", "X_test.txt"),
                         sep=r"\s+", header=None)
    X_test.columns = feature_names

    y_test = pd.read_csv(os.path.join(base_path, "test", "y_test.txt"),
                         sep=r"\s+", header=None)[0].map(activity_labels)

    X = pd.concat([X_train, X_test]).reset_index(drop=True)
    y = pd.concat([y_train, y_test]).reset_index(drop=True)

    data = X.copy()
    data["activity"] = y

    return data


# ============================================================
# 2. PREPROCESS DATA
# ============================================================

def preprocess_data(data):

    X = data.drop(columns=["activity"])
    y = data["activity"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # LSTM reshape (samples, timesteps, features)
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_val_lstm = X_val_s.reshape((X_val_s.shape[0], 1, X_val_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    classes = pd.Categorical(y)
    y_train_idx = pd.Categorical(y_train, categories=classes.categories).codes
    y_val_idx = pd.Categorical(y_val, categories=classes.categories).codes
    y_test_idx = pd.Categorical(y_test, categories=classes.categories).codes

    return (
        X_train_s, X_val_s, X_test_s,
        X_train_lstm, X_val_lstm, X_test_lstm,
        y_train_idx, y_val_idx, y_test_idx,
        list(classes.categories)
    )


# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

def build_ann(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation="relu"),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ============================================================
# 4. TRAIN MODELS
# ============================================================

def train_models(X_train_s, y_train, X_val_s, y_val,
                 X_train_lstm, X_val_lstm, num_classes):

    results = {}

    ann = build_ann(X_train_s.shape[1], num_classes)
    hist_ann = ann.fit(
        X_train_s, y_train, validation_data=(X_val_s, y_val),
        epochs=20, batch_size=64, verbose=1
    )
    results["ANN"] = (ann, hist_ann)

    mlp = build_mlp(X_train_s.shape[1], num_classes)
    hist_mlp = mlp.fit(
        X_train_s, y_train, validation_data=(X_val_s, y_val),
        epochs=20, batch_size=64, verbose=1
    )
    results["MLP"] = (mlp, hist_mlp)

    lstm = build_lstm((1, X_train_s.shape[1]), num_classes)
    hist_lstm = lstm.fit(
        X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
        epochs=20, batch_size=64, verbose=1
    )
    results["LSTM"] = (lstm, hist_lstm)

    return results


# ============================================================
# 5. COMBINED CONFUSION MATRIX FOR ANN + MLP
# ============================================================

def plot_confusion_matrices_ann_mlp(ann, mlp, X_test_s, y_test, class_names):

    models = {
        "ANN": (ann, X_test_s),
        "MLP": (mlp, X_test_s)
    }

    plt.figure(figsize=(14, 6))

    for i, (name, (model, X_test)) in enumerate(models.items(), 1):
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)

        plt.subplot(1, 2, i)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. COMPARATIVE PLOTS
# ============================================================

def plot_validation_accuracy_all(hist_ann, hist_mlp, hist_lstm):
    plt.figure(figsize=(8, 5))
    plt.plot(hist_ann.history["val_accuracy"], label="ANN")
    plt.plot(hist_mlp.history["val_accuracy"], label="MLP")
    plt.plot(hist_lstm.history["val_accuracy"], label="LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison Across Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_f1_all_models(ann, mlp, lstm, X_test_s, X_test_lstm, y_test, class_names):

    pred_ann = np.argmax(ann.predict(X_test_s), axis=1)
    pred_mlp = np.argmax(mlp.predict(X_test_s), axis=1)
    pred_lstm = np.argmax(lstm.predict(X_test_lstm), axis=1)

    _, _, f1_ann, _ = precision_recall_fscore_support(y_test, pred_ann, zero_division=0)
    _, _, f1_mlp, _ = precision_recall_fscore_support(y_test, pred_mlp, zero_division=0)
    _, _, f1_lstm, _ = precision_recall_fscore_support(y_test, pred_lstm, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, f1_ann, width, label="ANN")
    plt.bar(x, f1_mlp, width, label="MLP")
    plt.bar(x + width, f1_lstm, width, label="LSTM")

    plt.xticks(x, class_names, rotation=30, ha="right")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison Across ANN, MLP, and LSTM")
    plt.ylim(0.90, 1.00)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ============================================================
# FEATURE VISUALISATION
# ============================================================

def plot_feature_correlation_heatmap(data):
    plt.figure(figsize=(12, 10))
    corr = data.iloc[:, :50].corr()
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap (First 50 Features)")
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(data, features):
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(features[:9]):
        plt.subplot(3, 3, i + 1)
        plt.hist(data[feature], bins=30, color="blue")
        plt.title(feature)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():

    data = load_data()

    (
        X_train_s, X_val_s, X_test_s,
        X_train_lstm, X_val_lstm, X_test_lstm,
        y_train, y_val, y_test,
        class_names
    ) = preprocess_data(data)

    results = train_models(
        X_train_s, y_train, X_val_s, y_val,
        X_train_lstm, X_val_lstm, len(class_names)
    )

    ann, hist_ann = results["ANN"]
    mlp, hist_mlp = results["MLP"]
    lstm, hist_lstm = results["LSTM"]

    # ===== Section 4.1 =====
    plot_feature_correlation_heatmap(data)
    plot_feature_histograms(data, data.columns[:-1])

    # ===== Section 4.2 =====
    plot_validation_accuracy_all(hist_ann, hist_mlp, hist_lstm)
    compare_f1_all_models(
        ann, mlp, lstm, X_test_s, X_test_lstm, y_test, class_names
    )

    # ===== Section 4.3 (CONFUSION MATRIX FOR ANN + MLP ONLY) =====
    plot_confusion_matrices_ann_mlp(ann, mlp, X_test_s, y_test, class_names)


if __name__ == "__main__":
    main()
