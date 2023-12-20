from djlib.plotting.mc_plotting import sgcmc_full_project_diagnostic_plots
import djlib.propagation.propagate_gcmc as pg
import sys

def main():
    # directory should be the casm root of the Monte Carlo project (ie. /path/to/sample_index_0)
    directory = sys.argv[1]
    data = pg.propagation_project_parser(directory,incomplete_override=True)
    figure = sgcmc_full_project_diagnostic_plots(data,show_legends=False)
    figure.savefig(directory+'/diagnostic_plots.png',dpi=300)
    return

if __name__ == '__main__':
    main()