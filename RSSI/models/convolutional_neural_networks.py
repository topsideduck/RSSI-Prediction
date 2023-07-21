import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


class ConvolutionalNeuralNetworkModel:
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load the generated dataset from a CSV file.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        return pd.read_csv(file_path)

    def preprocess_dataset(
        self, dataset: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess the dataset by separating features and target, and scaling the features.

        Parameters:
            dataset (pd.DataFrame): The dataset containing x, y coordinates, and RSSI values.

        Returns:
            tuple[np.ndarray, np.ndarray]: The scaled features (X) and target (y).
        """
        X = dataset[["x", "y"]].values
        y = dataset["rssi"].values

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        return X, y

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> keras.Model:
        """Create and train the improved CNN model.

        Parameters:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training target.
            epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size for training.

        Returns:
            keras.Model: The trained CNN model.
        """
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(X_train.shape[1],)
                ),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2
        )

        return model

    def evaluate_model(
        self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """Evaluate the model on the test set.

        Parameters:
            model (keras.Model): The trained CNN model.
            X_test (np.ndarray): The testing features.
            y_test (np.ndarray): The testing target.

        Returns:
            float: R-squared (accuracy) of the model.
        """
        y_pred_test = model.predict(X_test).flatten()
        accuracy = r2_score(y_test, y_pred_test)
        return accuracy

    def plot_predictions(self, y_test: np.ndarray, y_pred_test: np.ndarray) -> None:
        """Create a scatter plot to compare predicted vs. actual RSSI values.

        Parameters:
            y_test (np.ndarray): The actual RSSI values.
            y_pred_test (np.ndarray): The predicted RSSI values.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, s=50, alpha=0.7)
        plt.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="gray"
        )
        plt.xlabel("Actual RSSI")
        plt.ylabel("Predicted RSSI")
        plt.title("Predicted vs. Actual RSSI")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Create the ImprovedCNN object
    convolutional_neural_network_model = ConvolutionalNeuralNetworkModel()

    # Load the dataset
    dataset = convolutional_neural_network_model.load_dataset("data/rssi_dataset_rectangle.csv")

    # Preprocess the dataset
    X, y = convolutional_neural_network_model.preprocess_dataset(dataset)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the CNN model
    trained_model = convolutional_neural_network_model.train_model(
        X_train, y_train, epochs=100, batch_size=32
    )

    # Evaluate the model
    accuracy = convolutional_neural_network_model.evaluate_model(trained_model, X_test, y_test)
    print("R-squared (Accuracy):", accuracy)

    # Plot the predictions
    convolutional_neural_network_model.plot_predictions(y_test, trained_model.predict(X_test).flatten())
