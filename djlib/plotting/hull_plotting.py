import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull
import thermocore.geometry.hull as thull
import djlib.djlib as dj
from sklearn.metrics import mean_squared_error


def general_binary_convex_hull_plotter(
    composition: np.ndarray, true_energies: np.ndarray, predicted_energies=[None]
) -> matplotlib.figure.Figure:
    """Plots a 2D convex hull for any 2D dataset. Can optionally include predicted energies to compare true and predicted formation energies and conved hulls.

    Parameters:
    -----------
    composition: numpy.ndarray
        Vector of composition values, same length as true_energies.
    true_energies: numpy.ndarray
        Vector of formation energies. Required.
    predicted_energies: numpy.ndarray
        None by default. If a vector of energies are provided, it must be the same length as composition. RMSE score will be reported.
    """

    predicted_color = "red"
    predicted_label = "Predicted Energies"

    plt.scatter(
        composition, true_energies, color="k", marker="x", label='"True" Energies'
    )

    dft_hull = thull.full_hull(compositions=composition, energies=true_energies)

    if any(predicted_energies):
        predicted_hull = ConvexHull(
            np.hstack(
                (composition.reshape(-1, 1), np.reshape(predicted_energies, (-1, 1)))
            )
        )

    dft_lower_hull_vertices = thull.lower_hull(dft_hull)[0]
    if any(predicted_energies):
        predicted_lower_hull_vertices = thull.lower_hull(predicted_hull)[0]

    dft_lower_hull = dj.column_sort(dft_hull.points[dft_lower_hull_vertices], 0)

    if any(predicted_energies):
        predicted_lower_hull = dj.column_sort(
            predicted_hull.points[predicted_lower_hull_vertices], 0
        )

    plt.plot(
        dft_lower_hull[:, 0], dft_lower_hull[:, 1], marker="D", markersize=15, color="k"
    )
    if any(predicted_energies):
        plt.plot(
            predicted_lower_hull[:, 0],
            predicted_lower_hull[:, 1],
            marker="D",
            markersize=10,
            color=predicted_color,
        )
    if any(predicted_energies):
        plt.scatter(
            composition,
            predicted_energies,
            color="red",
            marker="x",
            label=predicted_label,
        )

        rmse = np.sqrt(mean_squared_error(true_energies, predicted_energies))
        plt.text(
            min(composition),
            0.9 * min(np.concatenate((true_energies, predicted_energies))),
            "RMSE: " + str(rmse) + " eV",
            fontsize=21,
        )

    plt.xlabel("Composition X", fontsize=21)
    plt.ylabel("Formation Energy (eV)", fontsize=21)
    plt.legend(fontsize=21)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(19, 14)
    return fig
