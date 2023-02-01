import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial


def derivative(func):
    """
    Implements a numerical approximation of the derivative
    of a function
    """
    h = 0.000000000001
    return lambda x: (func(x+h) - func(x))/h


class MetricSpace():
    """
    A metric space consists of a set X with a function
    d: XxX -> [0, infty)
    satisfying the following axioms for all x,y,z in X:
    1) d(x,y)=0 if and only if x=y
    2) d(x,y) = d(y,x)
    3) d(x,z) <= d(x,y) + d(y,z)

    The distance function is represented by the distance matrix,
    an nxn matrix D where n = |X|. The entry D[i,j]=d(x_i, x_j).

    Note that Axiom 1 means the diagonal is zero, and Axiom 2 implies
    the matrix is symmetric, i.e. D[i,j] = D[j,i] for all i,j.
    """

    def __init__(self, distance_matrix_=None):
        """
        Metric space is essentially defined by its distance matrix.
        We can create a metric space by specifying the distance
        matrix directly, or we can instantiate the object and create
        distance matrix separately.
        """
        self.distance_matrix_ = distance_matrix_

    def __repr__(self):
        """
        Returns the distance matrix as a string
        """
        return str(self.distance_matrix_)

    @staticmethod
    def spread(matrix):
        """
        The spread of a metric space as defined by Simon Willerton.
        The spread here is implemented as a function on the distance
        matrix.

        This is a vectorised implementation.
        """
        D = matrix
        J = np.ones(len(D))
        E = np.exp(-1*D)
        return np.sum(1/(E@J))

    @staticmethod
    def pseudo_spread(partial_matrix):

        list_of_pseudo_spread_values = []
        for row in partial_matrix:
            list_of_pseudo_spread_values.append(
                np.exp(-1*row).sum()
            )

        return np.array(list_of_pseudo_spread_values)

    def pseudo_spread_dimension_numerical(self, t):

        results = []
        for row in self.partial_distance_matrix_:

            def f(t): return MetricSpace.spread(t*row)
            f_prime = derivative(f)

            results.append(float((t/f(t))*(f_prime(t))))

        return results

    def spread_dimension_exact(self, t):
        """
        Returns the spread dimension of distance
        matrix when scaled by a factor of t.

        This is a vectorised implementation of the
        exact formula for spread dimension of a finite
        metric space.
        """
        D = self.distance_matrix_
        J = np.ones(len(D))
        E = np.exp(-t*D)
        return (t/MetricSpace.spread(t*D)) * np.sum(((D*E)@J)/((E@J)**2))

    def spread_dimension_numerical(self, t):
        """
        It will sometimes be more efficient to use an approximation
        of the spread dimension using a numerical approximation of
        the derivative, rather than the exact value above
        """

        def f(t): return MetricSpace.spread(t*self.distance_matrix_)

        f_prime = derivative(f)
        return float((t/f(t))*(f_prime(t)))

    def compute_spread_dimension_profile(self,
                                         min_t,
                                         max_t,
                                         no_of_vals,
                                         exact=True):
        """
        Creates a pandas DataFrame with a range of spread dimension
        calculated at different scalings.

        By default uses the exact analytical formula for computing the
        spread dimension. Including exact=False will revert to using
        numerical approximation

        spread_profile_max stores the higest dimension value attained,
        and the corresponding scale factor as a tuple
        (scale_factor, highest dim)
        """

        t_values = np.linspace(min_t, max_t, no_of_vals)

        entries = []

        if exact:
            for t in t_values:
                entries.append(
                    (
                        t,
                        self.spread_dimension_exact(t)
                    )
                )
        elif not exact:
            for t in t_values:
                entries.append(
                    (
                        t,
                        self.spread_dimension_numerical(t)
                    )
                )

        spread_dimension_profile = pd.DataFrame(entries)
        spread_dimension_profile.columns = [
            'scale factor',
            'spread dimension'
        ]

        self.spread_dimension_profile = spread_dimension_profile

        max_dimension_val = (
            spread_dimension_profile['spread dimension'].max()
        )

        max_dimension = spread_dimension_profile[
            spread_dimension_profile['spread dimension'] == max_dimension_val
        ]
        optimum_scale_factor = max_dimension['scale factor'].max()
        self.spread_profile_max = (optimum_scale_factor, max_dimension_val)

    def show_spread_dimension_profile(self, n=0):
        """
        Generates a plot of the spread dimension profile if the method
        compute_spread_dimension_profile has already been called.

        Optional parameter n includes a dashed horizontal line at n if
        chosen n>0. This is for quick visual comparison against expected
        dimension values.
        """
        self.spread_dimension_profile.sort_values('scale factor', inplace=True)
        x = self.spread_dimension_profile['scale factor']
        y = self.spread_dimension_profile['spread dimension']
        plt.plot(x, y)
        plt.grid()
        plt.xlim(
            left=0,
            right=self.spread_dimension_profile['scale factor'].max()
        )
        plt.ylim(bottom=0)
        plt.xlabel('Scale Factor')
        plt.ylabel('Spread Dimension')
        if n:
            plt.axhline(y=n, linestyle='dashed')

    def compute_spread_profile(self,
                               min_t,
                               max_t,
                               no_of_vals,
                               exact=True
                               ):
        """
        Creates a Pandas DataFrame with a range of spread
        calculated at different scalings.

        """
        t_values = np.linspace(min_t, max_t, no_of_vals+1)[1:]

        entries = []

        for t in t_values:
            spread_t = MetricSpace.spread(t*self.distance_matrix_)
            entries.append(
                (
                    t,
                    spread_t,
                    np.log(spread_t)/np.log(t)
                )
            )

        spread_profile = pd.DataFrame(entries)
        spread_profile.columns = [
            'scale factor',
            'spread',
            'log spread_t / log t'
        ]
        self.spread_profile = spread_profile

    def show_spread_profile(self, n=0):
        """
        Generates a plot of the spread profile if the method
        compute_spread_profile has already been called.

        Optional parameter n includes a dashed horizontal line at n if
        chosen n>0.
        """

        self.spread_profile.sort_values('scale factor', inplace=True)
        x = self.spread_profile['scale factor']
        y = self.spread_profile['spread']
        plt.plot(x, y)
        plt.grid()
        plt.xlim(
            left=0,
            right=self.spread_profile['scale factor'].max()
        )
        plt.ylim(bottom=0)
        plt.xlabel('Scale Factor')
        plt.ylabel('Spread')
        if n:
            plt.axhline(y=n, linestyle='dashed')

    def show_spread_log_growth_profile(self, n=0):
        """
        Generates a plot of the spread log-log growth profile if the method
        compute_spread_profile has already been called.

        Optional parameter n includes a dashed horizontal line at n if
        chosen n>0. This is for quick visual comparison against expected
        dimension values.
        """

        self.spread_profile.sort_values('scale factor', inplace=True)
        x = self.spread_profile['scale factor']
        y = self.spread_profile['log spread_t / log t']
        plt.plot(x, y)
        plt.grid()
        plt.xlim(
            left=0,
            right=self.spread_profile['scale factor'].max()
        )
        plt.ylim(bottom=0)
        plt.xlabel('Scale Factor')
        plt.ylabel('log log growth')

        if n:
            plt.axhline(y=n, linestyle='dashed')

    def from_distance_map(distance_map):
        """
        Takes a dsitance map in the form of a dict with entries
        d = {(x,x):0, (x,y):...} and generates an adjacency matrix
        from this data, and returns a metric space with this
        adjacency matrix.

        This is slow for large metric spaces, but useful for creating
        some spaces.
        """
        M = len(distance_map)
        N = int(np.sqrt(M))

        if N*N != M:
            raise ValueError(
                'Invalid distance function: number of elements in the domain'
                f' ({M}) should be a square number'
            )

        distance_matrix_ = np.zeros(N*N).reshape(N, N)

        elements = set()
        for i in distance_map:
            elements = elements.union(set(i))

        elements = list(elements)
        elements.sort()

        for i in range(N):
            for j in range(i, N):

                distance_matrix_[i, j] = distance_map[
                    (elements[i], elements[j])
                ]

        distance_matrix_ = distance_matrix_ + np.transpose(distance_matrix_)

        return MetricSpace(distance_matrix_)

    def subspace_from_indices(self, list_of_indices):
        """
        Takes a subset of the metric space in the form of a list of indices
        and returns the MetricSpace defined by that subset in terms of
        its distance matrix
        """
        rows = []
        for i in list_of_indices:
            rows.append(self.distance_matrix_[i])
        rows = np.array(rows)

        columns = []
        for i in list_of_indices:
            columns.append(rows[:, i])

        return MetricSpace(np.array(columns))


class EuclideanSubspace(MetricSpace):
    """
    A Euclidean subspace is a subset of points of R^n
    for some n.

    These are useful examples of metric spaces, with
    a variety of possible metrics.

    It will also be useful to keep track of the underlying
    points because for example we may want to implement
    noise reduction techniques on these subspaces.

    We store these points as a list of tuples, where
    the tuple represents the coordinates in n-dimensional
    Euclidean space
    """

    def __init__(self, list_of_points):
        self.points = list_of_points

    def __repr__(self):
        """
        Returns the points as a list
        """

        return str(self.points)

    def compute_metric(self, p_norm=2):
        """
        computes the pairwise distances of all points
        using the p-norm distance d(x,y) = ||x-y||p where
        for x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        default value is p=2 which is the standard Euclidean
        distance.

        uses scipy implementation of distance_matrix for speed
        """

        self.distance_matrix_ = scipy.spatial.distance_matrix(
            self.points,
            self.points,
            p=p_norm,
            threshold=100000000
        )

    def compute_partial_metric(self, p_norm=2, list_of_indices=[0]):
        """
        computes the pairwise distances between a chosen subset of points
        and the whole set using the p-norm distance d(x,y) = ||x-y||p where
        for x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        default value is p=2 which is the standard Euclidean
        distance.

        uses scipy implementation of distance_matrix for speed
        """
        chosen_points = []
        for i in list_of_indices:
            chosen_points.append(self.points[i])
        chosen_points = np.array(chosen_points)

        self.partial_distance_matrix_ = scipy.spatial.distance_matrix(
            chosen_points,
            self.points,
            p=p_norm,
            threshold=100000000
        )

    def knn_smoothing(self, k):
        """
        This method generates an entirely new set of points by
        mapping each point in the space to the average position of
        that points k-nearest neighbours

        If input k is an integer then k is that number of
        nearest neighbours.

        If input k is a number less than 1 then the number of
        nearest neighbours is taken to be that proportion
        of points overall.
        """
        if k < 1:
            k = int(len(self.points)*k)
        else:
            k = k

        smoothed_points = []
        for i in range(len(self.distance_matrix_)):
            row = self.distance_matrix_[i]
            k_nearest_indices = np.argpartition(row, k)[:k]
            k_nearest = []
            for j in k_nearest_indices:
                k_nearest.append(self.points[j])

            k_nearest = np.array(k_nearest)
            average_coordinates = k_nearest.mean(axis=0).tolist()
            smoothed_points.append(tuple(average_coordinates))

        return EuclideanSubspace(smoothed_points)


if __name__ == '__main__':

    print('Metric space with three elements a, b and c')
    dist_map = {
        ('a', 'a'): 0,
        ('a', 'b'): 1,
        ('b', 'a'): 1,
        ('b', 'b'): 0,
        ('a', 'c'): 1,
        ('c', 'a'): 1,
        ('b', 'c'): 2,
        ('c', 'b'): 2,
        ('c', 'c'): 0
    }
    print('distance function defined explicitly with distance map:')
    print(f'{dist_map}')

    print('creating MetricSpace object from distance map')
    T = MetricSpace.from_distance_map(dist_map)

    print(f'corresponding distance matrix is:\n{T.distance_matrix_}')

    print(
        'compute the instantaneous growth dimension for 100 values'
        ' over a range [0, 10] and plot the results:'
    )
    T.compute_spread_dimension_profile(0, 10, 100, exact=True)
    T.show_spread_dimension_profile()
    plt.show()

    print(f'the maximum dimension attained is {T.spread_profile_max[1]}')
    print(f'when the scale factor t={T.spread_profile_max[0]}')
    print(T.spread_profile_max)

    print(
        'compute the spread for 800 values'
        ' over a range (1, 100] and plot the log-log results:'
    )
    T.compute_spread_profile(1, 100, 800)
    T.show_spread_log_growth_profile()
    plt.show()
