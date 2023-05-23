import numpy as np


def sc_corr(dataframe):
    """Collects the site centric correlation vector for each environment, from every supercell.

    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe containing the site centric correlations

    Returns
    -------
    cor : numpy array
        array of all site centric correlations
    cor_remove_duplicate : numpy array
        array of all site centric correlations with duplicates removed
    """

    cor_len = len(np.array(dataframe["site_centric_correlations"][0]["value"][0]))
    cor = np.empty((0, cor_len))
    for i in range(len(dataframe.index)):
        cor = np.append(
            cor, np.array(dataframe["site_centric_correlations"][i]["value"]), axis=0
        )
    cor_remove_duplicate = np.unique(cor, axis=0)
    return cor, cor_remove_duplicate


def num_site(dataframe):
    """The primitive unit cell has some number of alloying sites.
    Supercells are composed of multiple primitive unit cells, and therefore contain multiple alloying sites.
    This function returns the number of alloying sites in each supercell.

    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe containing the site centric correlations

    Returns
    -------
    site : list
        list of the number of sites in each supercell
    """
    site = []
    for i in range(len(dataframe.index)):
        site.append(
            len(dataframe["site_centric_correlations"][i]["asymmetric_unit_indices"])
        )
    return site


def kernel_n(sc_kernel, site):
    """Takes the output of pairwise_kernel from sklearn, and sums kernel rows by supercell membership.

    Parameters
    ----------
    sc_kernel : numpy array
        kernel matrix from pairwise_kernel, shape (n_supercells * n_sites_per_supercell, n_unique_site_environments)
    site : list
        list of the number of sites in each supercell, shape (n_supercells)

    Returns
    -------
    Kernel : numpy array
        Matrix of kernels, now summed by supercell membership. Shape (n_supercells, n_unique_site_environments)

    Notes
    -----
    Kernel rows correspond to sites in calculated supercells. Some supercells may contain one or more sites.
    In order to fit the kernel to formation energy, the sites must be linked back to their corresponding supercells.
    This is achieved by summing all kernel rows that correspond to a single supercell.
    Supercell membership can be determined by the number of sites in each supercell.

    """
    Kernel = np.array([np.sum(sc_kernel[0 : site[0]], axis=0)])
    for i in range(len(site) - 1):
        Kernel = np.append(
            Kernel,
            [np.sum(sc_kernel[np.cumsum(site)[i] : np.cumsum(site)[i + 1]], axis=0)],
            axis=0,
        )
    return Kernel
