import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as scipy_distance_matrix
import time

from examples.get_snake_eyes_data import load_snake_eyes_ones, load_smooth_snake_eyes_ones
from metric_space import EuclideanSubspace


def draw_dice(pixel_values):
    pixels_square = np.array(pixel_values).reshape(20, 20)
    plt.imshow(pixels_square, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()



if __name__ == '__main__':

    data = load_smooth_snake_eyes_ones()
    for d in data:
        draw_dice(d)
    ms = EuclideanSubspace(data)
