import tensorflow as tf
from keras.models import load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, classification_report

import numpy as np
import os
import argparse

from lib.models import UTKModel, IMG_SIZE, CHANNEL, process_dataset, extract_features, parse_filepath


# suppress messages from TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINTS_DIR = 'checkpoints/UTKFace'
DATASET_DIR = 'datasets/UTKFace/UTKFace'
MODEL_PATH = 'models/UTKFace.h5'

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
VAL_SPLIT = 0.2

NUM_AGE_CATEGORIES = 6  # 0-17, 18-25, 26-35, 36-45, 46-55, 56+

AGE_CATEGORY_MAP = {
    0: 'Minor',
    1: 'Young Adult',
    2: 'Adult',
    3: 'Middle-Aged',
    4: 'Senior',
    5: 'Elderly'
}


def categorize_age(age):
    if age <= 17:
        return 0  # Minor
    elif age <= 25:
        return 1  # Young Adult
    elif age <= 35:
        return 2  # Adult
    elif age <= 45:
        return 3  # Middle-Aged
    elif age <= 55:
        return 4  # Senior
    else:
        return 5  # Elderly


def parse_args():
    parser = argparse.ArgumentParser(description='Train UTKFace model')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help='Path to save the model')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for training the model')
    parser.add_argument('--val-split', type=float, default=VAL_SPLIT,
                        help='Validation split for training the model')

    # For loading model and plotting predictions
    parser.add_argument('--load-model', action='store_true',
                        help='Load the model from the model-path')
    parser.add_argument('--plot-predictions', action='store_true',
                        help='Plot the predictions of the model')

    # Predict Single Image
    parser.add_argument('--predict-image', type=str,
                        help='Path to the image to predict the age')

    return parser.parse_args()


def initialize_model():
    model = UTKModel()
    model.compile(optimizer='adam',
                  loss={'age_output': 'mse',
                        'age_category_output': 'sparse_categorical_crossentropy'},
                  metrics={'age_output': 'mae', 'age_category_output': 'accuracy'})
    return model


def load_checkpoint(model, checkpoint_dir=CHECKPOINTS_DIR):
    model.load_weights(checkpoint_dir)
    return model


def load_model(model_path=MODEL_PATH):
    return load_model(model_path)


def get_callbacks(checkpoint_dir=CHECKPOINTS_DIR):
    return [
        ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]


def train_model(model, X, y_age, y_age_category, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, val_split=VAL_SPLIT):
    history = model.fit(X, {'age_output': y_age, 'age_category_output': y_age_category}, epochs=epochs, batch_size=batch_size, validation_split=val_split,
                        callbacks=get_callbacks())
    return history


def load_dataset(dataset_dir=DATASET_DIR):
    return process_dataset(dataset_dir)


def evaluate_model(model, X, y_age, y_age_category):
    results = model.evaluate(
        X, {'age_output': y_age, 'age_category_output': y_age_category})
    print("Loss (Age Output, Age Category Output):", results[0])
    print("MAE (Age Output):", results[1])
    print("Accuracy (Age Category Output):", results[2])


def plot_history(history):
    # Plotting age prediction loss (regression output)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['age_output_loss'], label='Train Age Loss')
    plt.plot(history.history['val_age_output_loss'], label='Val Age Loss')
    plt.title('Model Age Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting age prediction MAE (regression output)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_output_mae'], label='Train Age MAE')
    plt.plot(history.history['val_age_output_mae'], label='Val Age MAE')
    plt.title('Model Age MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()

    plt.savefig('outputs/age_loss_and_mae.png')
    plt.show()

    # Plotting age category accuracy (classification output)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['age_category_output_accuracy'],
             label='Train Category Accuracy')
    plt.plot(history.history['val_age_category_output_accuracy'],
             label='Val Category Accuracy')
    plt.title('Model Age Category Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting age category loss (classification output)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_category_output_loss'],
             label='Train Category Loss')
    plt.plot(
        history.history['val_age_category_output_loss'], label='Val Category Loss')
    plt.title('Model Age Category Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig('outputs/age_category_accuracy_and_loss.png')
    plt.show()


def plot_age_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, color='blue',
                label='True Ages', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, color='red',
                label='Predicted Ages', alpha=0.5)
    plt.title('Age Predictions: True vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Age')
    plt.legend()

    plt.savefig('outputs/age_predictions.png')
    plt.show()


def load_checkpoint_and_predict(model, checkpoint_dir, image_path):
    model = load_checkpoint(model, checkpoint_dir)
    X = extract_features([image_path])
    X = X / 255.0
    y_pred = model.predict(X)
    y_pred_age = y_pred['age_output']
    y_pred_category = y_pred['age_category_output']
    print(f"Predicted Age: {y_pred_age[0][0]}")
    print(
        f"Predicted Age Category: {AGE_CATEGORY_MAP[y_pred_category[0].argmax()]}")

    return y_pred


def load_model_and_predict(model_path, image_path):
    model = load_model(model_path)
    X = extract_features([image_path])
    X = X / 255.0
    y_pred = model.predict(X)
    print(f"Predicted Age: {y_pred[0][0][0]}")
    print(f"Predicted Age Category: {y_pred[1][0]}")

    return y_pred


def plot_confusion_matrix(y_true, y_pred_category):
    cm = confusion_matrix(y_true, y_pred_category)
    print(cm)
    plt.matshow(cm, cmap='Blues',)

    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig('outputs/confusion_matrix.png')
    plt.show()

    print(classification_report(y_true, y_pred_category))


def plot_accuracy(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(
        f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}")


def main():
    print('TensorFlow version:', tf.__version__)
    print('Is Executing Eagerly?', tf.executing_eagerly())

    args = parse_args()

    if args.predict_image:
        model = initialize_model()
        true_age, _, _ = parse_filepath(args.predict_image)
        print(f"True Age: {true_age}")
        y_pred = load_checkpoint_and_predict(
            model, CHECKPOINTS_DIR, args.predict_image)
        y_pred_age, y_pred_category = y_pred['age_output'], y_pred['age_category_output']

        predicted_age_int = int(y_pred_age[0][0])

        # Plot the image with the predictions
        img = plt.imread(args.predict_image)
        plt.imshow(img)
        plt.title(f"Predicted Age: {predicted_age_int}, True Age: {true_age}")
        plt.xlabel(
            f"Predicted Age Category: {AGE_CATEGORY_MAP[y_pred_category[0].argmax()]}")
        plt.show()
        return

    if args.load_model:
        model = load_model()
        if args.plot_predictions:
            # Load datasetap
            df = load_dataset()

            # Extract features
            X = extract_features(df['file'])

            # Normalize the features
            X = X / 255.0

            # Prepare the labels
            y_age = np.array(df['age'])

            # Plot the predictions
            plot_predictions(model, X, y_age)
        return

    # Load dataset
    df = load_dataset()
    print(df.head())

    # Extract features
    X = extract_features(df['file'])

    # Normalize the features
    X = X / 255.0

    # Prepare the labels
    # Prepare the labels
    y_age = np.array(df['age'])  # Continuous age
    y_age_category = np.array(df['age_category'])  # Categorical age bins

    # Initialize the model
    model = initialize_model()

    # Load weights (TODO: Optional)
    model.load_weights(CHECKPOINTS_DIR)

    # Train the model
    history = train_model(model, X, y_age, y_age_category, args.epochs,
                          args.batch_size, args.val_split)

    # try:
    #     # Save the model (TODO: Handle error)
    #     model.save(MODEL_PATH, save_format='tf')
    # except Exception as e:
    #     print("Error saving the model:", e)

    try:
        # Plot the history
        plot_history(history)
    except Exception as e:
        print("Error plotting the history:", e)

    try:
        # Evaluate the model
        evaluate_model(model, X, y_age, y_age_category)
    except Exception as e:
        print("Error evaluating the model:", e)

    try:
        # Prediction
        y_pred = model.predict(X)
        print("Y_pred", y_pred)
        y_pred_age, y_pred_category = y_pred['age_output'], y_pred['age_category_output']

        # Plot age predictions (TODO: Handle error)
        # plot_age_predictions(y_age, y_pred_age)

        # Plot the confusion matrix for age category
        plot_confusion_matrix(y_age_category, y_pred_category.argmax(axis=1))

        # Plot the accuracy
        plot_accuracy(y_age_category, y_pred_category.argmax(axis=1))
    except Exception as e:
        print("Error plotting predictions:", e)


if __name__ == '__main__':
    main()
