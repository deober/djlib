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

def main():
    """
    testing stuff
    """
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_ELEMENTS.dat',emin=-40,emax=10,title='SCEL4_2_2_1_1_1_0/68')
    # Plot the PDOS from the VASPkit output
    
    '''
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A5.dat',title='SCEL4_2_2_1_1_1_0/68-Ti-1',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A6.dat',title='SCEL4_2_2_1_1_1_0/68-Ti-2',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A7.dat',title='SCEL4_2_2_1_1_1_0/68-Ti-3',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A2.dat',title='SCEL4_2_2_1_1_1_0/68-O-1',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A3.dat',title='SCEL4_2_2_1_1_1_0/68-O-2',emin=-9,emax=2)
    plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A4.dat',title='SCEL4_2_2_1_1_1_0/68-O-3',emin=-9,emax=2)
    '''
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_A1.dat',title='SCEL4_2_2_1_1_1_0/68-N',emin=-9,emax=2)
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_1/PDOS_USER.dat',title='SCEL4_2_2_1_1_1_0/68-N_py',emin=-9,emax=2)
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_Ti.dat',title='SCEL3_3_1_1_0_2_1/10-Ti')
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_O.dat',title='SCEL3_3_1_1_0_2_1/10-O')
    #plot_dos_by_element('/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/PDOS_N.dat',title='SCEL3_3_1_1_0_2_1/10-N')

    # Plot the PDOS from the VASPkit output
    root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/filled_Ti_vac/dos/'
    emin = -11
    emax = 3
    ### Plot and save individual orbital PDOSes ###
    '''
    dat_file_list = ['Ti1_dx2.dat','Ti1_dxy.dat','Ti1_dxz.dat','Ti1_dz2.dat','Ti1_dyz.dat','Ti1_pz.dat','Ti1_py.dat','Ti1_px.dat','Ti2_dx2.dat','Ti2_dxy.dat','Ti2_dxz.dat','Ti2_dz2.dat','Ti2_dyz.dat','Ti2_pz.dat','Ti2_py.dat','Ti2_px.dat','Ti3_dx2.dat','Ti3_dxy.dat','Ti3_dxz.dat','Ti3_dz2.dat','Ti3_dyz.dat','Ti3_pz.dat','Ti3_py.dat','Ti3_px.dat','O3_pz.dat','O3_py.dat','O3_px.dat','O2_pz.dat','O2_py.dat','O2_px.dat','O1_pz.dat','O1_py.dat','O1_px.dat','N_pz.dat','N_py.dat','N_px.dat',]
    #root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/near_hull/FCC/SCEL3_3_1_1_0_2_1/10/calctype.SCAN/dos_1/'
 
    for dat_file in dat_file_list:
        plot_dos_by_element(os.path.join(root_loc,'PDOS_'+dat_file),emin=emin,emax=emax,title='SCEL4_2_2_1_1_1_0-68-'+dat_file,save_plot=True)
    '''
    dat_file = 'O1_pz.dat'
    #plot_dos_by_element(os.path.join(root_loc,'PDOS_'+dat_file),emin=emin,emax=emax,title='SCEL4_2_2_1_1_1_0-68-'+dat_file,save_plot=True)
    
    #config_names = ['N','O1','O2','O3','Ti1','Ti2','Ti3']

    #fig = plt.figure()
    '''
    subfigs = fig.subfigures(nrows=1,ncols=2,wspace=0.1,hspace=0.30)
    TiAxs=subfigs[0].subplots(3,1,sharex=True)
    NOAxs=subfigs[1].subplots(4,1,sharex=True)
    subfigs[0].suptitle('Ti')
    subfigs[1].suptitle('N & O')
    #plot Ti PDOS's
    for i,ax in enumerate(TiAxs):
        fname = root_loc+'IPDOS_A%s'%(i+5)+'.dat' #+5 for Ti3NO3 + 4 for Ti2NO2
        config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_names[i+4]
        #config_name = 'SCEL3_3_1_1_0_2_1-10-'+config_names[i+3]
        print(fname,config_name)
        add_pdos_data_to_axes(ax,fname,config_name=config_name,emin=emin,emax=emax,hide_labels=False)
    #plot NO PDOS's
    for i,ax in enumerate(NOAxs):
        fname = root_loc+'IPDOS_A%s'%(i+1)+'.dat'
        config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_names[i]
        #config_name = 'SCEL3_3_1_1_0_2_1-10-'+config_names[i]
        print(fname,config_name)
        add_pdos_data_to_axes(ax,fname,config_name=config_name,emin=emin,emax=emax,hide_labels=False)
    '''
    
    #Compare N different PDOS's using add_pdos_data_to_axes
    '''
    config_1='Ti1_dz2'
    config_2='Ti1_dx2'
    config_3='Ti4_dxz'
    config_4='Ti3_dyz'
    config_5='Ti3_dxz'
    config_6='Ti2_dz2'
    config_7='Ti2_dx2'
    config_8='Ti2_dxy'
    config_9='Ti2_dyz'
    config_10='Ti2_dxz'
    config_11='Ti3_dz2'
    config_12='Ti3_dx2'
    config_13='Ti3_dxy'
    config_14='Ti3_dyz'
    config_15='Ti3_dxz'
    #NxM grid
    N=2
    #make overall figure
    plots = fig.subplots(N,1,sharex=True)
    #fill in individual subplots
    for i,ax in enumerate(plots):
        print(i,ax)
        if i==0:
            fname = root_loc+'PDOS_'+config_1+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_1
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==1:
            fname = root_loc+'PDOS_'+config_2+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_2
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==2:
            fname = root_loc+'PDOS_'+config_3+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_3
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==3:
            fname = root_loc+'PDOS_'+config_4+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_4
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==4:
            fname = root_loc+'PDOS_'+config_5+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_5
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,
            emax=emax,hide_labels=False)
        if i==5:
            fname = root_loc+'PDOS_'+config_6+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_6
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==6:
            fname = root_loc+'PDOS_'+config_7+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_7
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==7:
            fname = root_loc+'PDOS_'+config_8+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_8
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==8:
            fname = root_loc+'PDOS_'+config_9+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_9
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==9:
            fname = root_loc+'PDOS_'+config_10+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_10
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==10:
            fname = root_loc+'PDOS_'+config_11+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_11
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==11:
            fname = root_loc+'PDOS_'+config_12+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_12
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==12:
            fname = root_loc+'PDOS_'+config_13+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_13
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==13:
            fname = root_loc+'PDOS_'+config_14+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_14
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        if i==14:
            fname = root_loc+'PDOS_'+config_15+'.dat'
            config_name = 'SCEL4_2_2_1_1_1_0-68-'+config_15
            print(fname,config_name)
            add_pdos_data_to_axes(ax,fname,title=config_name,emin=emin,emax=emax,hide_labels=False)
        '''
    
    # Now using upgraded plot_dos_stackplot
    title = 'Ti4NO3 - Ti3NO3 structure with Ti vacancy filled'
    # Plot Ti DOS
    for n in range(1,5):
        fig = plt.figure()
        plt.suptitle(title,fontsize=20)
        plots = fig.subplots(nrows=3,ncols=1,sharex=True)
        ax1 = plots[0]
        ax2 = plots[1]
        ax3 = plots[2]
        plot_dos_stackplot(ax1,os.path.join(root_loc,'IPDOS_Ti%s.dat'% (n)),orbitals_to_plot=['dxy','dyz','dz2','dxz','dx2'],emin=emin,emax=emax,zero_idos_at_emin=True,title='Integrated DOS Ti%s' % n,hide_labels=False)
        plot_dos_stackplot(ax2,os.path.join(root_loc,'PDOS_Ti%s.dat' % n),orbitals_to_plot=['dxy','dyz', 'dxz'],emin=emin,emax=emax,title='DOS Ti%s' % n,hide_labels=False)
        #plot_dos_stackplot(ax3,os.path.join(root_loc,'PDOS_Ti%s.dat' % n),orbitals_to_plot=['dz2','dx2'],emin=emin,emax=emax,title='eg DOS Ti%s' % n,hide_labels=False)
        plot_dos_stackplot(ax3,os.path.join(root_loc,'PDOS_O%s.dat' % n),orbitals_to_plot=['dz2','dx2'],emin=emin,emax=emax,title='DOS O%s' % n,hide_labels=False)
        plt.show()
        plt.close()


    #Plot O DOS
    for n in range(1,4):
        fig = plt.figure()
        plt.suptitle(title,fontsize=20)
        plots = fig.subplots(nrows=2,ncols=1,sharex=True)
        ax1 = plots[0]
        ax2 = plots[1]
        #ax3 = plots[2]
        plot_dos_stackplot(ax1,os.path.join(root_loc,'IPDOS_O%s.dat'% (n)),orbitals_to_plot=['px','py','pz'],emin=emin,emax=emax,zero_idos_at_emin=True,title='Integrated DOS O%s' % n,hide_labels=False)
        plot_dos_stackplot(ax2,os.path.join(root_loc,'PDOS_O%s.dat' % n),orbitals_to_plot=['px','py', 'pz'],emin=emin,emax=emax,title='p DOS O%s' % n,hide_labels=False)
        #plot_dos_stackplot(ax3,os.path.join(root_loc,'PDOS_O%s.dat' % n),orbitals_to_plot=['dz2','dx2'],emin=emin,emax=emax,title='eg DOS Ti%s' % n,hide_labels=False)
        plt.show()
        plt.close()

    #Plot N DOS
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=2,ncols=1,sharex=True)
    ax1 = plots[0]
    ax2 = plots[1]
    #ax3 = plots[2]
    plot_dos_stackplot(ax1,os.path.join(root_loc,'IPDOS_N.dat'),orbitals_to_plot=['px','py','pz'],emin=emin,emax=emax,zero_idos_at_emin=True,title='Integrated DOS N',hide_labels=False)
    plot_dos_stackplot(ax2,os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['px','py', 'pz'],emin=emin,emax=emax,title='p DOS N',hide_labels=False)        
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()