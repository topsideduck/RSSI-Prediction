import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class LinearRegressionModel:
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
        """Preprocess the dataset by separating features and target.

        Parameters:
            dataset (pd.DataFrame): The dataset containing x, y coordinates, and RSSI values.

        Returns:
            tuple[np.ndarray, np.ndarray]: The features (X) and target (y).
        """
        X = dataset[["x", "y"]].values
        y = dataset["rssi"].values
        return X, y

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train a linear regression model using the training data.

        Parameters:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training target.

        Returns:
            LinearRegression: The trained linear regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self, model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate the linear regression model on the test set and calculate the R-squared (accuracy) of the model.

        Parameters:
            model (LinearRegression): The trained linear regression model.
            X_test (np.ndarray): The testing features.
            y_test (np.ndarray): The testing target.

        Returns:
            float: R-squared (accuracy) of the model.
        """
        y_pred_test = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        # Create a dictionary to store the results
        evaluation_results = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }

        return evaluation_results

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
        plt.title("Linear Regression: Predicted vs. Actual RSSI")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    linear_regression_model = LinearRegressionModel()
    dataset = linear_regression_model.load_dataset("data/rssi_dataset_rectangle.csv")

    # Preprocess the dataset
    X, y = linear_regression_model.preprocess_dataset(dataset)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    begin_time = time.time()
    trained_model = linear_regression_model.train_model(X_train, y_train)
    end_time = time.time()

    # Evaluate the model and get evaluation metrics
    evaluation_metrics = linear_regression_model.evaluate_model(trained_model, X_test, y_test)

    # Display evaluation metrics
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    print(f"Time to train: {end_time - begin_time} seconds.")

    # Plot the predictions
    linear_regression_model.plot_predictions(y_test, trained_model.predict(X_test))
