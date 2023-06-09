import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as scipy_distance_matrix
import time


def load_snake_eyes_data():
    df = pd.read_csv('snake_eyes_ones.csv.gz', compression='gzip')
    arr = np.array(df)
    return arr


def find_n_nearest_neighbours(chosen_point, points, n):
    relative_distances = scipy_distance_matrix([chosen_point], points, p=2)[0]
    indices_of_closest_n = np.argpartition(relative_distances, n)[:n]
    return [points[i] for i in indices_of_closest_n]


def create_smoothed_data(n=500):

    arr = load_snake_eyes_data()

    smoothed_points = []
    s = time.time()
    for point in arr[:100]:
        neighbourhood = find_n_nearest_neighbours(point, arr, n)
        smoothed_point = np.zeros(400)
        for neighbouring_point in neighbourhood:
            smoothed_point = smoothed_point + neighbouring_point
        smoothed_point = smoothed_point/n
        smoothed_points.append(smoothed_point)
    e = time.time()
    print(f'took {e-s:.2f} seconds')

    return smoothed_points


if __name__ == '__main__':
    smoothed_points = create_smoothed_data()
    df = pd.DataFrame(smoothed_points)
    df.to_csv('snake_eyes_ones_smooth.csv', index=False)
