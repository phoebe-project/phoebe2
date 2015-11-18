import numpy as np
import matplotlib.pyplot as plt

def get_data(f):
    """
    """
    ifile = open(f, 'r')
    header = (ifile.readlines()[0]).split()
    ifile.close()
    
    if header[0] == '#':
	header = header[1:]
    header = [x.lower() for x in header]
    
    d = np.loadtxt(f)
    
    data = {}
    for i,rec in enumerate(header):
	if rec == 'vphase':
	    d[:,i] = d[:,i]%np.pi
	data[rec] = d[:,i]
	
    return data

def plot_comparison(d1, d2, target, ax=None):
    """
    """
    b1 = np.sqrt(d1['ucoord']**2 + d1['vcoord']**2)/d1['eff_wave']*1e10
    b2 = np.sqrt(d2['ucoord']**2 + d2['vcoord']**2)/d2['eff_wave']*1e10
    ax.plot(b1, d1[target], 'ro')
    ax.plot(b2, d2[target], 'bo')
    ax.set_xlabel('Spatial frequency')
    ax.set_ylabel(target)

def plot_residual(d1, d2, target, ax=None):
    """
    """
    b1 = np.sqrt(d1['ucoord']**2 + d1['vcoord']**2)/d1['eff_wave']*1e10
    b2 = np.sqrt(d2['ucoord']**2 + d2['vcoord']**2)/d2['eff_wave']*1e10
    ax.plot(b1, d1[target]-d2[target], 'ro')
    ax.set_xlabel('Spatial frequency')
    ax.set_ylabel("Residual: "+target)  

def main():
    """
    """
    p2 = get_data('if01.p2.syn')
    ac = get_data('if01.analytic.syn')
    #print p2['vphase'], ac['vphase']
    plot_obs = ['vis2', 'vphase']
    
    nrow = len(plot_obs)
    ncol = 2
    plt.figure(figsize=(5*ncol,5*nrow), dpi=100)
    for i,rec in enumerate(plot_obs):
        # visibility
        ax = plt.subplot(nrow, ncol, i*ncol+1)
        plot_comparison(p2, ac, rec, ax=ax)
        ax = plt.subplot(nrow, ncol, i*ncol+2)
        plot_residual(p2, ac, rec, ax=ax)

    plt.savefig('comparison_uniform_disk.png')

if __name__ == '__main__':
    main()