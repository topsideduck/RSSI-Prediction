import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the generated dataset
dataset = pd.read_csv("data/rssi_dataset_rectangle.csv")

# Separate features (x, y coordinates) and target (RSSI values)
X = dataset[["x", "y"]].values
y = dataset["rssi"].values

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Create a grid of coordinates covering the entire room
x_max, y_max = 6, 4
resolution = 0.1  # Smaller resolution for more accurate predictions
xx, yy = np.meshgrid(
    np.arange(0, x_max + resolution, resolution),
    np.arange(0, y_max + resolution, resolution),
)

# Reshape the grid into a flat array for prediction
grid_points = np.column_stack((xx.ravel(), yy.ravel()))

# Make predictions on the grid
predicted_rssi_data = model.predict(grid_points).reshape(xx.shape)

# Calculate the R-squared (accuracy) of the model using the test dataset
y_pred_test = model.predict(X_test)
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
