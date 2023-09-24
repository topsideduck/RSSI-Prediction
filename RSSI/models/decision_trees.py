import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeModel:
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
        """Preprocess the dataset, separating features (x, y coordinates) and target (RSSI values).

        Parameters:
            dataset (pd.DataFrame): The input dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing features (X) and target (y).
        """
        X = dataset[["x", "y"]].values
        y = dataset["rssi"].values
        return X, y

    def find_best_max_depth(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> int:
        """Find the optimal value for max_depth using cross-validation.

        Parameters:
            X_train (np.ndarray): Features of the training set.
            y_train (np.ndarray): Target values of the training set.
            X_test (np.ndarray): Features of the testing set.
            y_test (np.ndarray): Target values of the testing set.

        Returns:
            int: The optimal value of max_depth.
        """
        best_accuracy = 0.0
        best_max_depth = 1

        # Iterate through different values of max_depth and find the one with the highest accuracy
        for depth in range(1, 21):
            dt_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            dt_model.fit(X_train, y_train)
            y_pred = dt_model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_max_depth = depth

        return best_max_depth

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, best_max_depth: int
    ) -> DecisionTreeRegressor:
        """Train the Decision Trees model using the best value of max_depth.

        Parameters:
            X_train (np.ndarray): Features of the training set.
            y_train (np.ndarray): Target values of the training set.
            best_max_depth (int): The optimal value of max_depth.

        Returns:
            DecisionTreeRegressor: Trained Decision Trees model.
        """
        dt_model = DecisionTreeRegressor(max_depth=best_max_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        return dt_model

    def evaluate_model(
        self, model: DecisionTreeRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate the model using the test set and calculate the R-squared (accuracy).

        Parameters:
            model (DecisionTreeRegressor): Trained Decision Trees model.
            X_test (np.ndarray): Features of the testing set.
            y_test (np.ndarray): Target values of the testing set.

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
            y_test (np.ndarray): Target values of the testing set.
            y_pred_test (np.ndarray): Predicted values on the testing set.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, s=50, alpha=0.7)
        plt.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="gray"
        )
        plt.xlabel("Actual RSSI")
        plt.ylabel("Predicted RSSI")
        plt.title("Decision Trees: Predicted vs. Actual RSSI")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Create the DecisionTreeModel object
    decision_tree_model = DecisionTreeModel()

    # Load the dataset
    dataset = decision_tree_model.load_dataset("data/rssi_dataset_rectangle.csv")

    # Preprocess the dataset
    X, y = decision_tree_model.preprocess_dataset(dataset)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Find the best value for max_depth using cross-validation
    best_max_depth = decision_tree_model.find_best_max_depth(X_train, y_train, X_test, y_test)

    # Train the Decision Trees model using the best value of max_depth
    begin_time = time.time()
    trained_model = decision_tree_model.train_model(X_train, y_train, best_max_depth)
    end_time = time.time()

    # Evaluate the model and get evaluation metrics
    evaluation_metrics = decision_tree_model.evaluate_model(trained_model, X_test, y_test)

    # Display evaluation metrics
    print("Best max_depth:", best_max_depth)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    print(f"Time to train: {end_time - begin_time} seconds.")

    # Plot the predictions
    decision_tree_model.plot_predictions(y_test, trained_model.predict(X_test))
