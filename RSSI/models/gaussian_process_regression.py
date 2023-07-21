import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class GaussianProcessRegressorModel:
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
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> GaussianProcessRegressor:
        """Create and train the Gaussian Process Regression (GPR) model.

        Parameters:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training target.

        Returns:
            GaussianProcessRegressor: The trained GPR model.
        """
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-10, 1e2)
        )
        gpr_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=50, random_state=42
        )
        gpr_model.fit(X_train, y_train)

        return gpr_model

    def evaluate_model(
        self, model: GaussianProcessRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """Evaluate the GPR model on the test set and calculate the R-squared (accuracy) of the model.

        Parameters:
            model (GaussianProcessRegressor): The trained GPR model.
            X_test (np.ndarray): The testing features.
            y_test (np.ndarray): The testing target.

        Returns:
            float: R-squared (accuracy) of the model.
        """
        y_pred_test, _ = model.predict(X_test, return_std=True)
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
    # Load the dataset
    gaussian_process_regressor_model = GaussianProcessRegressorModel()
    dataset = gaussian_process_regressor_model.load_dataset(
        "data/rssi_dataset_rectangle.csv"
    )

    # Preprocess the dataset
    X, y = gaussian_process_regressor_model.preprocess_dataset(dataset)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the GPR model
    trained_model = gaussian_process_regressor_model.train_model(
        X_train, y_train
    )

    # Evaluate the model
    accuracy = gaussian_process_regressor_model.evaluate_model(
        trained_model, X_test, y_test
    )
    print("R-squared (Accuracy):", accuracy)

    # Plot the predictions
    gaussian_process_regressor_model.plot_predictions(
        y_test, trained_model.predict(X_test, return_std=True)[0]
    )
