import matplotlib.pyplot as plt
import numpy as np
import os

### All functions here work with outputs from vaspkit (found at https://vaspkit.com/) ###

def plot_pdos_from_vaspkit_output():
    """
    Plot the PDOS from the VASPkit output.
    """
    pass
    # Read the PDOS from the VASPkit output
    # Plot the PDOS
    

def plot_dos_by_element(dos_by_element_file:str,emin=None,emax=None,title='',save_plot=False):
    """
    Plot the DOS by element from the VASPkit output. (usually a PDOS_ELEMENTS.dat file)

    Inputs
    ------
    dos_by_element_file: str
        The file containing the DOS with columns by element as generated.
    emin: float (optional)
        The minimum energy to plot.
    emax: float (optional)
        The maximum energy to plot.
    title: str (optional)
        String for the plot title
    save_plot: bool (optional)
        Whether to save the plot.
    
    Returns
    -------
    None
    """
    # Read the DOS by element from the VASPkit output
    with open(dos_by_element_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].split()
    if 'Energy' in header:
        energy_col = header.index('Energy')
    elif '#Energy' in header:
        energy_col = header.index('#Energy')
    print("energy in column:", energy_col)
    orbitals_list = header[(energy_col+1):]
    print(header)
    print(orbitals_list)
    data = []
    for x in lines[1:]:
        data.append(x.split())
    data = np.array(data,dtype=float)
    if emin is None:
        imin = 0
    else:
        #find corresponding index
        imin = np.searchsorted(data[:,0],emin)
    if emax is None:
        imax = -1
    else:
        #find corresponding index
        imax = np.searchsorted(data[:,0],emax)
    energy = data[imin:imax,0]
    print(type(energy[0]))
    # Plot the DOS by element
    plt.figure()
    print(energy.shape)
    dos_stack = np.vstack([data[imin:imax,i] for i in range(1,len(header)-energy_col) if header[i] != 'tot'])
    print(dos_stack.shape)
    plt.stackplot(energy,dos_stack,labels=orbitals_list)
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV)')
    plt.title(title)
    if save_plot:
        plt.savefig(title+'.png')
    plt.show()
    pass
    # Plot the DOS by element

def plot_dos_stackplot(ax,dos_by_element_file:str,orbitals_to_plot:list=None,emin=None,emax=None,title='',fermi_level=0,zero_idos_at_emin=False,hide_labels=False):
    """
    Plot the DOS by element from the VASPkit output. (usually a PDOS_ELEMENTS.dat file)

    Inputs
    ------
    ax: matplotlib.axes.Axes
        The axes to add the data to.
        TODO: Generate a new figure if ax is None
    dos_by_element_file: str
        The file containing the DOS with columns by element as generated.
    orbitals_to_plot: list
        The orbitals to plot as a list of strings following format of the PDOS/IPDOS headers (e.g. ['s','px','dxy']).
    emin: float (optional)
        The minimum energy to plot.
    emax: float (optional)
        The maximum energy to plot.
    fermi_level: float (optional)
        Fermi level energy to show as a vertical line.
    zero_idos_at_emin: bool (optional)
        Whether to zero the IDOS at the minimum energy. (This is useful for plotting the IDOS with the PDOS over a smaller range of interest)
    title: str (optional)
        Plot title.
    hide_labels: bool (optional)
        Whether to hide the labels.
    
    Returns
    -------
    pdos_plt: matplotlib.axes.Axes
        The axes with the PDOS data added.
    """
    with open(dos_by_element_file, 'r') as f:
        print("reading file:", dos_by_element_file)
        lines = f.readlines()
    header = lines[0].split()
    if 'Energy' in header:
        energy_col = header.index('Energy')
    elif '#Energy' in header:
        energy_col = header.index('#Energy')
    print("energy in column:", energy_col)
    orbitals_list = header[(energy_col+1):]
    print(header)
    print(orbitals_list)
    # If no orbitals to plot, plot all orbitals
    if orbitals_to_plot is None:
        orbitals_to_plot = orbitals_list
        print("No orbitals specified, plotting all orbitals")
    data = []
    for x in lines[1:]:
        data.append(x.split())
    data = np.array(data,dtype=float)
    #get indices of energy range
    if emin is None:
        imin = 0
    else:
        #find corresponding index
        imin = np.searchsorted(data[:,0],emin)
    if emax is None:
        imax = -1
    else:
        #find corresponding index
        imax = np.searchsorted(data[:,0],emax)
    #Get energy list
    energy = data[imin:imax,0]
    print("Energy is type:",type(energy[0]))
    # Plot the DOS by element
    print(energy.shape)
    # Stack all columns except for "energy" and "tot"
    print("Plotting data for orbitals:",orbitals_to_plot)
    dos_stack = np.vstack([data[imin:imax,i] for i in range(1,len(header)-energy_col) if header[i] in orbitals_to_plot])
    # Subtract off the value from emin if it is integrated DOS and zero_idos_at_emin is True
    if zero_idos_at_emin:
        dos_stack = np.vstack([data[imin:imax,i]-data[imin,i] for i in range(1,len(header)-energy_col) if header[i] in orbitals_to_plot])
    else:
        dos_stack = np.vstack([data[imin:imax,i] for i in range(1,len(header)-energy_col) if header[i] in orbitals_to_plot])
    print(dos_stack.shape)
    pdos_plt = ax.stackplot(energy,dos_stack,labels=orbitals_to_plot)
    ax.axvline(x=fermi_level,color='k',linestyle='--')
    #pdos_plt = ax.plot(energy,dos_stack,labels=orbitals_to_plot)
    if not hide_labels:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.legend(loc='best')
    ax.set_title(title)
    return pdos_plt

def add_pdos_data_to_axes(ax,dos_by_element_file:str,emin=None,emax=None,title='',hide_labels=False):
    """
    Plot the DOS by element from the VASPkit output. (usually a PDOS_ELEMENTS.dat file)

    Inputs
    ------
    ax: matplotlib.axes.Axes
        The axes to add the data to.
    dos_by_element_file: str
        The file containing the DOS with columns by element as generated.
    emin: float (optional)
        The minimum energy to plot.
    emax: float (optional)
        The maximum energy to plot.
    title: str (optional)
        Plot title.
    hide_labels: bool (optional)
        Whether to hide the labels.
    
    Returns
    -------
    pdos_plt: matplotlib.axes.Axes
        The axes with the PDOS data added.
    """
    # Read the DOS by element from the VASPkit output
    with open(dos_by_element_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].split()
    if 'Energy' in header:
        energy_col = header.index('Energy')
    elif '#Energy' in header:
        energy_col = header.index('#Energy')
    print("energy in column:", energy_col)
    orbitals_list = header[(energy_col+1):]
    print(header)
    print(orbitals_list)
    data = []
    for x in lines[1:]:
        data.append(x.split())
    data = np.array(data,dtype=float)
    #get indices of energy range
    if emin is None:
        imin = 0
    else:
        #find corresponding index
        imin = np.searchsorted(data[:,0],emin)
    if emax is None:
        imax = -1
    else:
        #find corresponding index
        imax = np.searchsorted(data[:,0],emax)
    #Get energy list
    energy = data[imin:imax,0]
    print("Energy is type:",type(energy[0]))
    # Plot the DOS by element
    print(energy.shape)
    # Stack all columns except for "energy" and "tot"
    dos_stack = np.vstack([data[imin:imax,i] for i in range(1,len(header)-energy_col) if header[i] != 'tot'])
    print(dos_stack.shape)
    pdos_plt = ax.stackplot(energy,dos_stack,labels=orbitals_list)
    #pdos_plt = ax.plot(energy,dos_stack,labels=orbitals_list)
    if not hide_labels:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
    elements = ''
    for i in range(len(orbitals_list)):
        elements += orbitals_list[i]
    ax.set_title(title)
    return pdos_plt
