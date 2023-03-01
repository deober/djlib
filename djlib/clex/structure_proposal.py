"""
Ground state detection with lower confidence bound per unit cost
Author: Chong Liu, chongliu@cs.ucsb.edu
Created on Feb 16, 2022, last modified on Feb 26, 2023
"""
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import linear_model


def lower_confidence_bound_per_unit_cost(
    corr_calculated,
    comp_calculated,
    formation_energy,
    corr_uncalculated,
    comp_uncalculated,
    scel_volumes_uncalculated,
    delta=0.01,
    sigma=0.1,
    reg=None,
) -> int:
    """Function used to select the next structure for DFT calculation.
    The algorithm aims to find the next point that is most likely to be a new ground state,
    which it returns as a structure index.

    Parameters
    ----------
    corr_calculated : np.ndarray
        The correlation matrix of the structures that have been calculated.
    comp_calculated : np.ndarray
        The composition of the structures that have been calculated.
    formation_energy : np.ndarray
        The formation energy of the structures that have been calculated.
    corr_uncalculated : np.ndarray
        The correlation matrix of the structures that have not been calculated.
    comp_uncalculated : np.ndarray
        The composition of the structures that have not been calculated.
    scel_volumes_uncalculated : np.ndarray
        The supercell volume of the structures that have not been calculated.
    delta : float, optional
        The prediction failure probability used to control the size of lower confidence bound, by default 0.01
    sigma : float, optional
        The standard deviation of Gaussian observation noise, by default 0.1
    reg : sklearn.linear_model, optional
        The regression model used to predict the formation energy of uncalculated structures, by default None

    Returns
    -------
    proposed_structure_index : int
        The index of the structure that is most likely to be a new ground state.

    """

    if reg == None:
        reg = linear_model.Ridge(alpha=0.001, fit_intercept=False)
    reg.fit(corr_calculated, formation_energy)
    x = corr_calculated
    x_inv = np.linalg.inv(x.T @ x)
    # build low convex hull on selected points
    low_hull = get_low_hull(comp_calculated, formation_energy)
    violation_list = []
    for j in range(np.shape(corr_uncalculated)[0]):
        x_new = corr_uncalculated[j, :]
        # predict energies
        pred_eng = reg.predict(np.expand_dims(x_new, 0))
        # build confidence bound
        conf_band = (
            np.sqrt(2 * np.log(1 / delta)) * sigma**2 * x_new.T @ x_inv @ x_new
        )
        new_point = (comp_uncalculated[j], pred_eng[0] - conf_band)
        violation_per_cost = below_cost_hull(
            new_point, low_hull, scel_volumes_uncalculated[j]
        )
        violation_list.append(violation_per_cost)
    # select the point with lowest confidence bound per unit cost
    proposed_structure_index = np.where(np.min(violation_list) == violation_list)
    return proposed_structure_index


def below_cost_hull(point, hull, scel):
    for i in range(np.shape(hull)[0] - 1):
        if hull[i, 0] < point[0] <= hull[i + 1, 0]:
            hull_line = (hull[i, 1] - hull[i + 1, 1]) / (
                hull[i, 0] - hull[i + 1, 0]
            ) * (point[0] - hull[i + 1, 0]) + hull[i + 1, 1]
            # lower confidence band divided by cost where cost = s ** 2 * log(s)
            return (point[1] - hull_line) / (scel**2 * np.log(scel)) ** 1


def get_low_hull(comp, eng):
    points = np.vstack((comp, eng))
    points = points.transpose()
    hull = ConvexHull(points)
    lower_hull_simplex_indices = (-hull.equations[:, -2] > 1e-14).nonzero()[0]
    lower_hull_vertex_indices = np.unique(
        np.ravel(hull.simplices[lower_hull_simplex_indices])
    )
    low_points = points[lower_hull_vertex_indices]
    low_hull = column_sort(low_points, 0)
    return low_hull


def column_sort(matrix: np.ndarray, column_index: int) -> np.ndarray:
    """Sorts a matrix by the values of a specific column. Far left column is column 0.
    Args:
        matrix(numpy_array): mxn numpy array.
        column_index(int): Index of the column to sort by.
    """
    column_index = int(column_index)
    sorted_matrix = matrix[np.argsort(matrix[:, column_index])]
    return sorted_matrix
