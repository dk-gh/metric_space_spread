import matplotlib.pyplot as plt
import numpy as np

from metric_space import MetricSpace


def sigma_circle(t):
    """
    This function returns the exact value of the spread of the metric space
    (S,d) where S is the unit circle, and distance function d is defined
    by the arc-length joining two points.

    This formula is due to a theorem of Willerton.
    """
    return (np.pi * t)/(1 - np.exp(-1*t*np.pi))


def circle_distance_matrix(n):
    """
    Let (X,d) be the metric space of n points evenly sampled from the
    unit circle in the Euclidean plane, where the distance d(x,y)
    is the arc-length between those two points along the circle.

    This function returns the distance matrix of this metric space
    as a numpy array.
    """
    arr = np.zeros(n**2).reshape(n, n)
    for i in range(n-1):
        for j in range(i+1, n):
            if j-i <= (n//2):
                d = j-i
            elif j-i > (n//2):
                d = (n-j)+i
            arr[i, j] = (d*2*np.pi)/n
    arr = arr + arr.T
    return arr


if __name__ == '__main__':

    dm = circle_distance_matrix(1000)
    S = MetricSpace(dm)

    S.compute_spread_dimension_profile(0, 50, 100)
    S.show_spread_dimension_profile()
    plt.show()

    S.compute_spread_profile(1, 1000, 1000)
    S.show_spread_log_growth_profile()
    plt.show()
