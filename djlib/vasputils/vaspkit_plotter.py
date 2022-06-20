import matplotlib.pyplot as plt
import numpy as np

### All functions here work with outputs from vaspkit (found at https://vaspkit.com/) ###

def plot_pdos_from_vaspkit_output():
    """
    Plot the PDOS from the VASPkit output.
    """
    pass
    # Read the PDOS from the VASPkit output
    # Plot the PDOS
    

def plot_dos_by_element(dos_by_element_file:str,emin=None,emax=None,config_name=''):
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
    config_name: str (optional)
        The name of the configuration.
    
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
    element_list = header[(energy_col+1):]
    print(header)
    print(element_list)
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
    plt.stackplot(energy,dos_stack,labels=element_list)
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV)')
    elements = ''
    for i in range(len(element_list)):
        elements += element_list[i]
    #plt.title(elements+' '+config_name)
    plt.title(config_name)
    plt.show()
    pass
    # Plot the DOS by element
 
def add_pdos_data_to_axes(ax,dos_by_element_file:str,emin=None,emax=None,config_name='',hide_labels=False):
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
    config_name: str (optional)
        The name of the configuration.
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
    element_list = header[(energy_col+1):]
    print(header)
    print(element_list)
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
    print(energy.shape)
    dos_stack = np.vstack([data[imin:imax,i] for i in range(1,len(header)-energy_col) if header[i] != 'tot'])
    print(dos_stack.shape)
    pdos_plt = ax.stackplot(energy,dos_stack,labels=element_list)
    ax.legend(loc='best',fontsize='small')
    if not hide_labels:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
    elements = ''
    for i in range(len(element_list)):
        elements += element_list[i]
    #plt.title(elements+' '+config_name)
    ax.set_title(config_name)
    return pdos_plt
    # Plot the DOS by element

def main():
    """
    testing stuff
    """
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_ELEMENTS.dat',emin=-40,emax=10,config_name='SCEL4_2_2_1_1_1_0/68')
    # Plot the PDOS from the VASPkit output
    
    '''
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A5.dat',config_name='SCEL4_2_2_1_1_1_0/68-Ti-1',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A6.dat',config_name='SCEL4_2_2_1_1_1_0/68-Ti-2',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A7.dat',config_name='SCEL4_2_2_1_1_1_0/68-Ti-3',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A2.dat',config_name='SCEL4_2_2_1_1_1_0/68-O-1',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A3.dat',config_name='SCEL4_2_2_1_1_1_0/68-O-2',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A4.dat',config_name='SCEL4_2_2_1_1_1_0/68-O-3',emin=-9,emax=2)
    '''
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A1.dat',config_name='SCEL4_2_2_1_1_1_0/68-N',emin=-9,emax=2)
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_USER.dat',config_name='SCEL4_2_2_1_1_1_0/68-N_py',emin=-9,emax=2)
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_Ti.dat',config_name='SCEL3_3_1_1_0_2_1/10-Ti')
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_O.dat',config_name='SCEL3_3_1_1_0_2_1/10-O')
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_N.dat',config_name='SCEL3_3_1_1_0_2_1/10-N')

    # Plot the PDOS from the VASPkit output
    root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/'
    emin = -3
    emax = 2.2
    config_names = ['N','O1','O2','O3','Ti1','Ti2','Ti3']
    fig = plt.figure()
    subfigs = fig.subfigures(nrows=1,ncols=2,wspace=0.1,hspace=0.30)
    TiAxs=subfigs[0].subplots(3,1,sharex=True)
    NOAxs=subfigs[1].subplots(4,1,sharex=True)
    subfigs[0].suptitle('Ti')
    subfigs[1].suptitle('N & O')
    #plot Ti PDOS's
    for i,ax in enumerate(TiAxs):
        fname = root_loc+'PDOS_A%s'%(i+5)+'.dat'
        config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_names[i+4]
        print(fname,config_name)
        add_pdos_data_to_axes(ax,fname,config_name=config_name,emin=emin,emax=emax,hide_labels=False)
    #plot NO PDOS's
    for i,ax in enumerate(NOAxs):
        fname = root_loc+'PDOS_A%s'%(i+1)+'.dat'
        config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_names[i]
        print(fname,config_name)
        add_pdos_data_to_axes(ax,fname,config_name=config_name,emin=emin,emax=emax,hide_labels=False)
    plt.show()

if __name__ == "__main__":
    main()