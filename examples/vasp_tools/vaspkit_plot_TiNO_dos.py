from logging import root
from djlib.vasputils.vaspkit_plotter import *
import matplotlib.pyplot as plt
import os

def main():
    '''
    testing stuff
    '''
    # Plot the PDOS from the VASPkit output
    root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/dos_4/'
    #root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/filled_Ti_vac-fixedsym/dos/'
    #root_loc = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/no_N-fixedsym/dos/'
    system = 'Ti3NO3 (left) Ti3O4 (right)' # or Ti3NO3 or Ti3O4 (replaced N) or Ti4NO3 - filled Ti vac
    emin = -11
    emax = 3
 
    
    # Now using upgraded plot_dos_stackplot
    
    # Symmetry Checks
    # Ti eg's
    '''
    title = system + ' - Ti eg symmetry checks'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=3,ncols=3,sharex=True,sharey=True)
    ax1 = plots[0,0]
    ax2 = plots[1,0]
    ax3 = plots[2,0]
    ax4 = plots[0,1]
    ax5 = plots[1,1]
    ax6 = plots[2,1]
    ax7 = plots[0,2]
    ax8 = plots[1,2]
    ax9 = plots[2,2]
    #ax10 = plots[3,0]
    #ax11 = plots[3,1]
    #ax12 = plots[3,2]

    plot_dos_stackplot(ax1,os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti1 dxy',hide_labels=False)
    plot_dos_stackplot(ax2,os.path.join(root_loc,'PDOS_Ti2.dat'),orbitals_to_plot=['dyz'],emin=emin,emax=emax,title='DOS Ti2 dyz',hide_labels=False)
    plot_dos_stackplot(ax3,os.path.join(root_loc,'PDOS_Ti3.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,title='DOS Ti3 dxz',hide_labels=False)
    plot_dos_stackplot(ax4,os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti1 dxz',hide_labels=False)
    plot_dos_stackplot(ax5,os.path.join(root_loc,'PDOS_Ti2.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,title='DOS Ti2 dxy',hide_labels=False)
    plot_dos_stackplot(ax6,os.path.join(root_loc,'PDOS_Ti3.dat'),orbitals_to_plot=['dyz'],emin=emin,emax=emax,title='DOS Ti3 dyz',hide_labels=False)
    plot_dos_stackplot(ax7,os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dyz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti1 dyz',hide_labels=False)
    plot_dos_stackplot(ax8,os.path.join(root_loc,'PDOS_Ti2.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,title='DOS Ti2 dxz',hide_labels=False)
    plot_dos_stackplot(ax9,os.path.join(root_loc,'PDOS_Ti3.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,title='DOS Ti3 dxy',hide_labels=False)
    #plot_dos_stackplot(ax10,os.path.join(root_loc,'PDOS_Ti4.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti4 dxy',hide_labels=False)
    #plot_dos_stackplot(ax11,os.path.join(root_loc,'PDOS_Ti4.dat'),orbitals_to_plot=['dyz'],emin=emin,emax=emax,title='DOS Ti4 dyz',hide_labels=False)
    #plot_dos_stackplot(ax12,os.path.join(root_loc,'PDOS_Ti4.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,title='DOS Ti4 dxz',hide_labels=False)
    plt.show()
    plt.close()
    '''
    # Symmetry Checks
    # Ti t2g's
    '''
    title = system + ' - Ti t2g symmetry checks'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
    ax1 = plots[0,0]
    ax2 = plots[1,0]
    ax3 = plots[2,0]
    ax4 = plots[0,1]
    ax5 = plots[1,1]
    ax6 = plots[2,1]
    #ax7 = plots[3,0]
    #ax8 = plots[3,1]

    plot_dos_stackplot(ax1,os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti1 dz2',hide_labels=False)
    plot_dos_stackplot(ax2,os.path.join(root_loc,'PDOS_Ti2.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,title='DOS Ti2 dx2',hide_labels=False)
    plot_dos_stackplot(ax3,os.path.join(root_loc,'PDOS_Ti3.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,title='DOS Ti3 dx2',hide_labels=False)
    plot_dos_stackplot(ax4,os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti1 dx2',hide_labels=False)
    plot_dos_stackplot(ax5,os.path.join(root_loc,'PDOS_Ti2.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,title='DOS Ti2 dz2',hide_labels=False)
    plot_dos_stackplot(ax6,os.path.join(root_loc,'PDOS_Ti3.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,title='DOS Ti3 dz2',hide_labels=False)
    #plot_dos_stackplot(ax7,os.path.join(root_loc,'PDOS_Ti4.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS Ti4 dz2',hide_labels=False)
    #plot_dos_stackplot(ax8,os.path.join(root_loc,'PDOS_Ti4.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,title='DOS Ti4 dx2',hide_labels=False)
    plt.show()
    plt.close()
    '''

    # Symmetry Checks
    # Anion p's
    '''
    title = system + ' - anion symmetry checks'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=4,ncols=3,sharex=True,sharey=True)
    r1 = plots[0,:]
    r2 = plots[1,:]
    r3 = plots[2,:]
    r4 = plots[3,:]
    plot_dos_stackplot(r1[0],os.path.join(root_loc,'PDOS_O1.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O1 px',hide_labels=False)
    plot_dos_stackplot(r1[1],os.path.join(root_loc,'PDOS_O1.dat'),orbitals_to_plot=['py'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O1 py',hide_labels=False)
    plot_dos_stackplot(r1[2],os.path.join(root_loc,'PDOS_O1.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O1 pz',hide_labels=False)
    plot_dos_stackplot(r2[0],os.path.join(root_loc,'PDOS_O2.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O2 px',hide_labels=False)
    plot_dos_stackplot(r2[1],os.path.join(root_loc,'PDOS_O2.dat'),orbitals_to_plot=['py'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O2 py',hide_labels=False)
    plot_dos_stackplot(r2[2],os.path.join(root_loc,'PDOS_O2.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O2 pz',hide_labels=False)
    plot_dos_stackplot(r3[0],os.path.join(root_loc,'PDOS_O3.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O3 px',hide_labels=False)
    plot_dos_stackplot(r3[1],os.path.join(root_loc,'PDOS_O3.dat'),orbitals_to_plot=['py'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O3 py',hide_labels=False)
    plot_dos_stackplot(r3[2],os.path.join(root_loc,'PDOS_O3.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O3 pz',hide_labels=False)
    #N
    #plot_dos_stackplot(r4[0],os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS N px',hide_labels=False)
    #plot_dos_stackplot(r4[1],os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['py'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS N py',hide_labels=False)
    #plot_dos_stackplot(r4[2],os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS N pz',hide_labels=False)
    #if replaced w/ O4 instead of N
    plot_dos_stackplot(r4[0],os.path.join(root_loc,'PDOS_O4.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O4 px',hide_labels=False)
    plot_dos_stackplot(r4[1],os.path.join(root_loc,'PDOS_O4.dat'),orbitals_to_plot=['py'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O4 py',hide_labels=False)
    plot_dos_stackplot(r4[2],os.path.join(root_loc,'PDOS_O4.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='DOS O4 pz',hide_labels=False)
    plt.show()
    plt.close()
    '''

    # With/without Ti vacancy
    '''
    root_loc_2 = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/filled_Ti_vac-fixedsym/dos/'
    # Metal-Metal bonds (Ti1 dxz-Ti2dxz)
    title = system + ' - Ti-Ti dxz-pz bonding (pi)'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    plot_dos_stackplot(plots[0,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dxz',hide_labels=False)
    plot_dos_stackplot(plots[0,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti4NO3 Ti1-dxz',hide_labels=False)
    plot_dos_stackplot(plots[1,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dxy',hide_labels=False)
    plot_dos_stackplot(plots[1,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti4NO3 Ti1-dxy',hide_labels=False)
    #plot_dos_stackplot(plots[1,0],os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 N-px',hide_labels=False)
    #plot_dos_stackplot(plots[1,1],os.path.join(root_loc_2,'PDOS_N.dat'),orbitals_to_plot=['px'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti4NO3 N-px',hide_labels=False)
    #plot_dos_stackplot(plots[2,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dxy',hide_labels=False)
    #plot_dos_stackplot(plots[2,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dxy'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti4NO3 Ti1-dxy',hide_labels=False)
    plt.show()
    plt.close()
    '''

    # N vs O (Ti3NO3 vs Ti3O4)
    root_loc_2 = '/home/jonnyli/Desktop/CASM/experiments/TiNO_full/ground_states/FCC/SCEL4_2_2_1_1_1_0/68/calctype.SCAN/no_N-fixedsym/dos/'
    # Metal-t2g (Ti1 dz2, dx2-y2)
    '''
    title = system + ' - Ti t2g'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    plot_dos_stackplot(plots[0,0],os.path.join(root_loc,'IPDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[0,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[0,1],os.path.join(root_loc_2,'IPDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[0,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[1,0],os.path.join(root_loc,'IPDOS_Ti1.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dx2',hide_labels=False)
    plot_dos_stackplot(plots[1,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dx2',hide_labels=False)
    plot_dos_stackplot(plots[1,1],os.path.join(root_loc_2,'IPDOS_Ti1.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dx2',hide_labels=False)
    plot_dos_stackplot(plots[1,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dx2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dx2',hide_labels=False)
    plt.show()
    plt.close()
    '''

    # Metal-anion sigma bond (Ti1 dz2, anion (N/O1) pz)
    title = system + ' - Ti-anion bonding'
    fig = plt.figure()
    plt.suptitle(title,fontsize=20)
    plots = fig.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
    plot_dos_stackplot(plots[0,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[0,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dz2'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dz2',hide_labels=False)
    plot_dos_stackplot(plots[1,0],os.path.join(root_loc,'PDOS_N.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 N-pz',hide_labels=False)
    plot_dos_stackplot(plots[1,1],os.path.join(root_loc_2,'PDOS_O1.dat'),orbitals_to_plot=['pz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 O1-pz',hide_labels=False)
    plot_dos_stackplot(plots[2,0],os.path.join(root_loc,'IPDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dxz',hide_labels=False)
    plot_dos_stackplot(plots[2,1],os.path.join(root_loc_2,'IPDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dxz',hide_labels=False)
    plot_dos_stackplot(plots[2,0],os.path.join(root_loc,'PDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3NO3 Ti1-dxz',hide_labels=False)
    plot_dos_stackplot(plots[2,1],os.path.join(root_loc_2,'PDOS_Ti1.dat'),orbitals_to_plot=['dxz'],emin=emin,emax=emax,zero_idos_at_emin=False,title='Ti3O4 Ti1-dxz',hide_labels=False)
    plt.show()
    plt.close()

    '''
    # Plot Ti DOS
    for n in range(1,4):
        fig = plt.figure()
        plt.suptitle(title,fontsize=20)
        plots = fig.subplots(nrows=3,ncols=1,sharex=True)
        ax1 = plots[0]
        ax2 = plots[1]
        ax3 = plots[2]
        plot_dos_stackplot(ax1,os.path.join(root_loc,'PDOS_Ti%s.dat'% (n)),orbitals_to_plot=['dxy','dyz','dz2','dxz','dx2'],emin=emin,emax=emax,zero_idos_at_emin=True,title='Integrated DOS Ti%s' % n,hide_labels=False)
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
    '''
if __name__ == "__main__":
    main()


def extra_stuff():
    # This is just a bunch of extra stuff that I was testing out

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

    return