import os

import numpy as np
import pandas as pd

# Constants for the empirical formula
REFERENCE_RSSI = -40  # Received signal strength at reference distance d0 (in dBm)
REFERENCE_DISTANCE = 1  # The reference distance from the transmitter (in m)
PATH_LOSS_EXPONENT = 2.5  # n in the empirical formula

# Room details
ROOM_SIZE = (6, 4)  # (width, height) of the rectangular room
RESOLUTION = 0.5  # Grid resolution for data generation

# Number of random samples
NUM_SAMPLES = 1000


def calculate_rssi(distance: np.ndarray, n: float) -> np.ndarray:
    """
    Calculate the Received Signal Strength Indicator (RSSI) values using the empirical formula.

    Parameters:
        distance (np.ndarray): Array of distances between the transmitter and receiver.
        n (float): Path loss exponent in the empirical formula.

    Returns:
        np.ndarray: Array of RSSI values corresponding to the given distances.
    """
    return REFERENCE_RSSI - 10 * n * np.log10(distance / REFERENCE_DISTANCE)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    Parameters:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def generate_rssi_data(
    transmitter_coord: tuple[float, float],
    room_size: tuple[float, float],
    resolution: float = 0.5,
) -> np.ndarray:
    """
    Generate the data set of RSSI values in a rectangular room for a given transmitter coordinate.

    Parameters:
        transmitter_coord (tuple): (x, y) coordinates of the transmitter.
        room_size (tuple): (width, height) of the rectangular room.
        resolution (float, optional): Grid resolution for data generation.

    Returns:
        np.ndarray: 2D array containing the RSSI values for the given transmitter coordinate.
    """
    x_transmitter, y_transmitter = transmitter_coord
    x_size, y_size = room_size

    # Create a grid of x and y coordinates in the room
    x = np.arange(0, x_size + resolution, resolution)
    y = np.arange(0, y_size + resolution, resolution)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distances between the transmitter and each point in the room
    distances = distance(x_transmitter, y_transmitter, xx, yy)

    # Calculate the RSSI values using the empirical formula
    rssi_data = calculate_rssi(distances, PATH_LOSS_EXPONENT)

    return rssi_data


def generate_dataset(
    transmitter_coord: tuple[float, float], num_samples: int
) -> pd.DataFrame:
    """
    Generate a dataset for training a model to predict RSSI at different locations.

    Parameters:
        transmitter_coord (tuple): (x, y) coordinates of the transmitter.
        num_samples (int): Number of random sample locations to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the input features (x, y coordinates) and RSSI values.
    """
    x_transmitter, y_transmitter = transmitter_coord
    x_max, y_max = ROOM_SIZE

    # Generate random sample locations within the rectangular room
    x_samples = np.random.uniform(0, x_max, num_samples)
    y_samples = np.random.uniform(0, y_max, num_samples)

    # Calculate the distances between the transmitter and each sample location
    distances = distance(x_transmitter, y_transmitter, x_samples, y_samples)

    # Calculate the RSSI values using the empirical formula
    rssi_values = calculate_rssi(distances, PATH_LOSS_EXPONENT)

    # Create a DataFrame to hold the dataset
    dataset = pd.DataFrame({"x": x_samples, "y": y_samples, "rssi": rssi_values})

    return dataset


if __name__ == "__main__":
    # Define the coordinates of the transmitter
    transmitter_coordinate = (2, 3)  # (x, y) of the transmitter

    # Generate the dataset
    dataset = generate_dataset(transmitter_coordinate, NUM_SAMPLES)

    # Create the "data" directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save the dataset to a CSV file
    dataset.to_csv("data/rssi_dataset_rectangular.csv", index=False)

    print("Dataset saved to 'data/rssi_dataset.csv'.")
