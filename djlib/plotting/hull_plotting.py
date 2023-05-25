import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull
import thermocore.geometry.hull as thull
import djlib.djlib as dj
from sklearn.metrics import mean_squared_error
import djlib.clex.clex as cl


def general_binary_convex_hull_plotter(
    composition: np.ndarray,
    true_energies: np.ndarray,
    predicted_energies=[None],
    print_extra_info: bool = False,
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
        composition, true_energies, color="k", marker="1", label='"True" Energies'
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
        # Also, check the set difference between the two lower hulls
        spurious = np.setdiff1d(predicted_lower_hull_vertices, dft_lower_hull_vertices)

        if len(spurious) > 0:
            plt.scatter(
                composition[spurious],
                predicted_energies[spurious],
                color="royalblue",
                marker="s",
                label="Spurious Predictions",
                s=400,
            )
            if print_extra_info:
                print("Spurious predictions:")
                print("Index", "Composition")
                for index in spurious:
                    print(index, composition[index])
    dft_lower_hull = dj.column_sort(dft_hull.points[dft_lower_hull_vertices], 0)

    if any(predicted_energies):
        predicted_lower_hull = dj.column_sort(
            predicted_hull.points[predicted_lower_hull_vertices], 0
        )

    plt.plot(
        dft_lower_hull[:, 0], dft_lower_hull[:, 1], marker="D", markersize=15, color="k"
    )
    if print_extra_info:
        print("DFT lower hull vertices:")
        print("Index", "Composition")
        for index in dft_lower_hull_vertices:
            print(index, composition[index])
    if any(predicted_energies):
        plt.plot(
            predicted_lower_hull[:, 0],
            predicted_lower_hull[:, 1],
            marker="D",
            markersize=10,
            color=predicted_color,
        )
        if print_extra_info:
            print("Predicted lower hull vertices:")
            print("Index", "Composition")
            for index in predicted_lower_hull_vertices:
                print(index, composition[index])
    if any(predicted_energies):
        plt.scatter(
            composition,
            predicted_energies,
            color="red",
            marker="2",
            label=predicted_label,
        )

        rmse = np.sqrt(mean_squared_error(true_energies, predicted_energies))
        plt.text(
            min(composition),
            0.9 * min(np.concatenate((true_energies, predicted_energies))),
            "RMSE: " + str(rmse) + " eV",
            fontsize=19,
        )

    plt.xlabel("Composition X", fontsize=21)
    plt.ylabel("Formation Energy per Primitive Cell (eV)", fontsize=21)
    plt.legend(fontsize=19, loc="upper right")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(15, 12)
    return fig

def binary_convex_hull_plotter_dft_and_overenumeration(ax, dft_comp, dft_formation_energies, dft_corr, over_comp, over_formation_energies, over_corr, dft_names=None, over_names=None):
    '''
    Plot the convex hull for the DFT data and the overenumerated data on the same plot. Gets spurious and missing ground states. Does matching via correlation matching (can't do name matching since CASM enumeration may number configurations differently)
    - also editable cause you pass it an axis!
    Inputs
    ------
        ax: matplotlib axis
            axis to plot on
        dft_comp: np.array
            (n,1) array of DFT compositions
        dft_formation_energies: np.array
            (n,) array formation energies of DFT data
        dft_corr: np.array
            (n,k) array (n calculated configs, k correlation functions) correlation matrix of DFT configurations
        over_comp: np.array
            (m,1) array of overenumerated compositions
        over_formation_energies: np.array
            (m,) array formation energies predictions on the overenumerated data
        over_corr: np.array
            (m,k) array (m overenumerated configs, k correlation functions) correlation matrix of overenumerated configurations
    Returns
    -------
        ax: matplotlib axis
            axis of the convex hull plot
    '''
    # corr shape checks
    print(dft_corr.shape, over_corr.shape)
    assert dft_corr.shape[1] == over_corr.shape[1]
    # calculate DFT hull & overenumerated hull
    dft_hull = thull.full_hull(compositions=dft_comp, energies=dft_formation_energies)
    dft_lower_hull_vertices, _dft = thull.lower_hull(dft_hull)
    over_hull = thull.full_hull(compositions=over_comp, energies=over_formation_energies)
    over_lower_hull_vertices, _over = thull.lower_hull(over_hull)
    print('DFT hull vertices:', dft_lower_hull_vertices, dft_comp[dft_lower_hull_vertices], '\nOverenumerated hull vertices:', over_lower_hull_vertices, over_comp[over_lower_hull_vertices])
    dft_lower_hull = dj.column_sort(dft_hull.points[dft_lower_hull_vertices], 0)
    over_lower_hull = dj.column_sort(over_hull.points[over_lower_hull_vertices], 0)

    dft_hull_indices_in_overenumerated = []
    over_hull_indices_in_dft = []
    missing_indices = []
    spurious_indices = []
    spurious_and_calculated_indices = []
    correct_dft_predictions = []

    for dft_index in dft_lower_hull_vertices:
        dft_hull_indices_in_overenumerated.append(np.where(np.all(over_corr==dft_corr[dft_index],axis=1))[0][0])
    for over_index in over_lower_hull_vertices:
        # append the index of the overenumerated hull point if it's in the DFT data
        if np.where(np.all(dft_corr==over_corr[over_index],axis=1))[0].shape[0] > 0:
            over_hull_indices_in_dft.append(np.where(np.all(dft_corr==over_corr[over_index],axis=1))[0][0])
            if np.where(np.all(dft_corr==over_corr[over_index],axis=1))[0][0] not in dft_lower_hull_vertices:
                spurious_indices.append(over_index)
                spurious_and_calculated_indices.append(over_index)
        else:
            spurious_indices.append(over_index)
    for test_dft_index in dft_hull_indices_in_overenumerated:
        # check if this index is on overenumerated hull, else add to missing indices
        if test_dft_index not in over_lower_hull_vertices:
            missing_indices.append(test_dft_index)
        else:
            correct_dft_predictions.append(test_dft_index)
    
    print("There are %i / %i DFT hull points on the overenumerated hull and %i missing and %i spurious ground states" % (len(correct_dft_predictions), len(dft_lower_hull_vertices), len(missing_indices), len(spurious_indices)))

    # print the correct, missing, and spurious ground states
    if dft_names is not None and over_names is not None:
        print("Correct DFT hull points:", correct_dft_predictions, "\n", over_names[correct_dft_predictions], '\n', over_comp[correct_dft_predictions]), print("Missing DFT hull points:", missing_indices, "\n",  over_names[missing_indices], '\n',over_comp[missing_indices]), print("Spurious overenumerated hull points:", spurious_indices, "\n",  over_names[spurious_indices], '\n', over_comp[spurious_indices])
        print("Spurious predictions that have already been calculated:", spurious_and_calculated_indices, "\n", over_names[spurious_and_calculated_indices], '\n', over_comp[spurious_and_calculated_indices], '\nEquivalent DFT indice:', [np.where(np.all(dft_corr==over_corr[idx],axis=1))[0][0] for idx in spurious_and_calculated_indices])
    else:
        print("Correct DFT hull points:", correct_dft_predictions, "\n", over_comp[correct_dft_predictions]), print("Missing DFT hull points:", missing_indices, "\n", over_comp[missing_indices]), print("Spurious overenumerated hull points:", spurious_indices, "\n", over_comp[spurious_indices])
    
    # scatter plot the remaining data
    ax.scatter(dft_comp, dft_formation_energies, c='k', marker='1', label='DFT data',zorder=2)
    ax.scatter(over_comp, over_formation_energies, c='r', marker='2', label='Clex predictions',zorder=1)
    ax.plot(dft_lower_hull[:,0], dft_lower_hull[:,1], 'k--', marker='D', markersize=15,label='__nolegend__')
    ax.plot(over_lower_hull[:,0], over_lower_hull[:,1], 'r:', marker='o',markersize=15,label='__no_legend__')
    # plot spurious and missing ground states (scatter)
    ax.scatter(over_comp[spurious_indices],over_formation_energies[spurious_indices],c='royalblue',s=230,marker='s',label='Spurious predictions')
    ax.scatter(over_comp[spurious_and_calculated_indices],over_formation_energies[spurious_and_calculated_indices],c='royalblue',s=280,marker='p',label='Spurious predictions (already calculated)')
    ax.scatter(over_comp[missing_indices],over_formation_energies[missing_indices],c='limegreen',s=230,marker='s',label='Missing ground states (clex prediction)')
    
    plt.legend(fontsize=19, loc="best")
    ax.set_xlabel('Composition', fontsize=30)
    ax.set_ylabel('Formation energy (eV/atom)', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=25)

    return ax

def plot_stable_chemical_potential_windows_for_binary(
    compositions: np.ndarray,
    energies: np.ndarray,
    names: np.ndarray,
    show_legend: bool = True,
) -> matplotlib.figure.Figure:
    """Takes composit   ions, formation energies and names of elements and plots the stable chemical potential windows for a binary system.

    Parameters
    ----------
    compositions: numpy.ndarray
        Vector of compositions. Assumes a binary system, so compositions must be a 1D vector.
    energies: numpy.ndarray
        Vector of formation energies. Must be the same length as compositions.
    names: numpy.ndarray
        Optional, None by default. If provided, must be a 1D vector of the same length as compositions. Names are used to label the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot.
    """

    # Get the convex hull
    hull = thull.full_hull(compositions, energies)
    lower_hull_vertices = thull.lower_hull(hull)[0]

    slopes = cl.calculate_slopes(
        compositions[lower_hull_vertices], energies[lower_hull_vertices]
    )

    slope_windows = slope_windows = np.array(
        [[slopes[i], slopes[i + 1]] for i in range(len(slopes) - 1)]
    )

    sorting_indices = np.argsort(np.ravel(compositions[lower_hull_vertices]))[1:-1]
    lower_hull_vertices = lower_hull_vertices[sorting_indices]

    for index, element in enumerate(lower_hull_vertices):
        plt.plot(
            np.ravel([compositions[element], compositions[element]]),
            np.ravel(slope_windows[index]),
            marker="o",
            markersize=15,
            label=names[element],
        )
    if show_legend:
        plt.legend(fontsize=21)

    plt.xlim([0, 1])
    plt.xlabel("Composition X", fontsize=21)
    plt.ylabel("Chemical Potential (eV)", fontsize=21)

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    fig = plt.gcf()
    return fig
