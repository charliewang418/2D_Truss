import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def ConfigPlot(x, y, f_unit, mark_print = 0, fig_name = ''):
    Nf = f_unit.shape[0]
    cmap = plt.get_cmap('turbo', Nf)
    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})
    cidx = np.random.permutation(Nf)

    # plot each triangle with a random color
    for nf in np.arange(Nf): 
        f_sub = f_unit[nf, :]
        xv = x[f_sub]
        yv = y[f_sub]
        ## verts = [list(zip(xv, yv))]
        verts = np.stack((xv, yv), axis = 1)
        tri = Polygon(verts, color = cmap.colors[cidx[nf]])
        ax.add_patch(tri)

    # plot vertex connections in black lines
    for f_sub in f_unit:
        xv = x[f_sub]
        yv = y[f_sub]
        ax.plot([xv[0], xv[1]], [yv[0], yv[1]], c = 'black', linewidth = 2)
        ax.plot([xv[1], xv[2]], [yv[1], yv[2]], c = 'black', linewidth = 2)
        ax.plot([xv[2], xv[0]], [yv[2], yv[0]], c = 'black', linewidth = 2)

    # plot vertices in blue dots
    for xv, yv in zip(x, y):
        ax.scatter(xv, yv, c = 'blue', s = 30)

    # whether or not to save the figure
    if (mark_print == 1) and (len(fig_name) != 0):
        fig.savefig(fig_name, dpi = 300)

    plt.show()