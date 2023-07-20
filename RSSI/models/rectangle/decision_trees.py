import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the generated dataset
dataset = pd.read_csv("data/rssi_dataset_rectangle.csv")

# Separate features (x, y coordinates) and target (RSSI values)
X = dataset[["x", "y"]].values
y = dataset["rssi"].values

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Decision Trees model
max_depth = 5  # Maximum depth of the decision tree
decision_tree_model = DecisionTreeRegressor(max_depth=max_depth)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = decision_tree_model.predict(X_test)

# Calculate the R-squared (accuracy) of the model
accuracy = r2_score(y_test, y_pred_test)

print("R-squared (Accuracy):", accuracy)

# Create a scatter plot to compare predicted vs. actual RSSI values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, s=50, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="gray")
plt.xlabel("Actual RSSI")
plt.ylabel("Predicted RSSI")
plt.title("Predicted vs. Actual RSSI")
plt.grid(True)
plt.show()
