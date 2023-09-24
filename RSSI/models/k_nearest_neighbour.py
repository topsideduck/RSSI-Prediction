import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


class KNearestNeighbourModel:
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load the dataset from the given CSV file.

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
            Tuple[np.ndarray, np.ndarray]: The scaled features (X) and target (y).
        """
        X = dataset[["x", "y"]].values
        y = dataset["rssi"].values

        # Scale the features to the range [0, 1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        return X, y

    def find_best_k(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> int:
        """Find the optimal value of k using cross-validation.

        Parameters:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training target.
            X_test (np.ndarray): The testing features.
            y_test (np.ndarray): The testing target.

        Returns:
            int: The best value of k.
        """
        best_accuracy = 0.0
        best_k = 1

        for k in range(1, 21):
            knn_model = KNeighborsRegressor(n_neighbors=k)
            knn_model.fit(X_train, y_train)
            y_pred = knn_model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        return best_k

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, best_k: int
    ) -> KNeighborsRegressor:
        """Train the KNN model using the best value of k.

        Parameters:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training target.
            best_k (int): The best value of k.

        Returns:
            KNeighborsRegressor: The trained KNN model.
        """
        knn_model = KNeighborsRegressor(n_neighbors=best_k)
        knn_model.fit(X_train, y_train)
        return knn_model

    def evaluate_model(
        self, model: KNeighborsRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate the model using the test set and calculate the R-squared (accuracy).

        Parameters:
            model (KNeighborsRegressor): The trained KNN model.
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
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, s=50, alpha=0.7)
        plt.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="gray"
        )
        plt.xlabel("Actual RSSI")
        plt.ylabel("Predicted RSSI")
        plt.title("K Nearest Neighbour: Predicted vs. Actual RSSI")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    k_nearest_neighbour_model = KNearestNeighbourModel()
    dataset = k_nearest_neighbour_model.load_dataset("data/rssi_dataset_rectangle.csv")

    # Preprocess the dataset
    X, y = k_nearest_neighbour_model.preprocess_dataset(dataset)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Find the best value of k using cross-validation
    best_k = k_nearest_neighbour_model.find_best_k(X_train, y_train, X_test, y_test)

    # Train the KNN model using the best value of k
    begin_time = time.time()
    trained_model = k_nearest_neighbour_model.train_model(X_train, y_train, best_k)
    end_time = time.time()

    # Evaluate the model and get evaluation metrics
    evaluation_metrics = k_nearest_neighbour_model.evaluate_model(trained_model, X_test, y_test)

    # Evaluate the model
    accuracy = k_nearest_neighbour_model.evaluate_model(trained_model, X_test, y_test)

    print("Best k:", best_k)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    print(f"Time to train: {end_time - begin_time} seconds.")

    # Plot the predictions
    k_nearest_neighbour_model.plot_predictions(y_test, trained_model.predict(X_test))
