# %%
import sys
import numpy as np

# from MD_functions import EnergyMinimization
from PlotFunctions import ConfigPlot
import MD_functions as MD
# %%
rand_id = int(sys.argv[1]) # random number seed
N_trap = int(sys.argv[2]) # number of trapezoids to form the ring
point_type = sys.argv[3] # where to put an extra point in the trapezoid, currently can be either 'edge' or 'center'
# %%
KA = 1.0 # spring constant for area energy term
R1 = 1.0 # radius of inner ring
R2 = 2.0 # radius of outer ring

# L1, L2, and L3: lengths for one trapezoid
L1 = R1 * 2.0 * np.sin(np.pi / N_trap)
L2 = R2 * 2.0 * np.sin(np.pi / N_trap)
L3 = R2 - R1
theta = np.arange(N_trap) / N_trap * 2 * np.pi # evenly space vertices for the trapezoids on the inner and outer rings

# indices for next (ift_temp) and previous (jft_temp) vertices along a ring, in a ciruclar way
ift_temp = np.concatenate((np.arange(1, N_trap), np.arange(1)))
jft_temp = np.concatenate((np.arange(N_trap - 1, N_trap), np.arange(N_trap - 1)))

# x1, y1: x and y coordinates of trapezoid vertices on the inner ring
# x2, y2: x and y coordinates of trapezoid vertices on the outer ring
x1 = R1 * np.cos(theta)
y1 = R1 * np.sin(theta)
x2 = R2 * np.cos(theta)
y2 = R2 * np.sin(theta)

xv_all = np.concatenate((x1, x2)) # x coordinates for all vertices in one row
yv_all = np.concatenate((y1, y2)) # y coordinates for all vertices in one row


if point_type == 'center': # add another point in the middle of the trapezoid
    Nv_cell = 2 * N_trap # number of vertices for all trapezoids per row
    Nv_temp = Nv_cell # total number of vertices before adding vertices in the center of each trapezoid and rectangle
    Nf = 4 * N_trap # number of triangles after adding vertices at the center of each trapezoid and rectangle
    
    f_unit = np.empty((0, 3), dtype = 'int16') # list of vertex indices for all triangles, will be Nf x 3 array
    # add one vertex per each trapezoid in the middle of the trapezoid
    for nt in np.arange(N_trap):
        Nv_temp = Nv_temp + 1
        v1 = nt
        v2 = ift_temp[nt]
        v3 = v2 + N_trap
        v4 = v1 + N_trap
        f_unit = np.concatenate((f_unit, np.array([[Nv_temp - 1, v2, v1], [Nv_temp - 1, v3, v2], [Nv_temp - 1, v4, v3], [Nv_temp - 1, v1, v4]])))
        xv_temp = np.mean(xv_all[[v1, v2, v3, v4]])
        yv_temp = np.mean(yv_all[[v1, v2, v3, v4]])
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))
elif point_type == 'edge': # add another point in the middle of the outer edge of the trapezoid
    Nv_cell = 2 * N_trap # number of vertices for all trapezoids per row
    Nv_temp = Nv_cell # total number of vertices before adding vertices in the center of each trapezoid and rectangle
    Nf = 3 * N_trap # number of triangles after adding vertices at the center of each trapezoid and rectangle
    
    f_unit = np.empty((0, 3), dtype = 'int16') # list of vertex indices for all triangles, will be Nf x 3 array
    # add one vertex per each trapezoid in the middle of the trapezoid
    for nt in np.arange(N_trap):
        Nv_temp = Nv_temp + 1
        v1 = nt
        v2 = ift_temp[nt]
        v3 = v2 + N_trap
        v4 = v1 + N_trap
        f_unit = np.concatenate((f_unit, np.array([[Nv_temp - 1, v2, v1], [Nv_temp - 1, v1, v4], [Nv_temp - 1, v3, v2]])))
        xv_temp = np.mean(xv_all[[v3, v4]])
        yv_temp = np.mean(yv_all[[v3, v4]])
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))

xv_ori = xv_all
yv_ori = yv_all
Nv = Nv_temp

A0 = np.zeros((Nf, ), dtype = 'float64') # area for all triangles, follow the same order of f_unit

for nf in np.arange(Nf):
    v1 = f_unit[nf, 0]
    v2 = f_unit[nf, 1]
    v3 = f_unit[nf, 2]
    r1 = np.array([xv_all[v1], yv_all[v1]])
    r2 = np.array([xv_all[v2], yv_all[v2]])
    r3 = np.array([xv_all[v3], yv_all[v3]])
    r12 = r2 - r1 # vector from site 1 to 2
    r13 = r3 - r1 # vector from site 1 to 3
    A0[nf] = 0.5 * (r12[0] * r13[1] - r12[1] * r13[0])

# %%
xv_pert = xv_ori + np.random.rand(Nv) * 0.001
yv_pert = yv_ori + np.random.rand(Nv) * 0.001
xv_equ, yv_equ = MD.FIRE(xv_pert, yv_pert, A0, Nv, Nf, f_unit, KA)

# Fx, Fy = MD_functions.Force(xv_equ, yv_equ, A0, Nv, Nf, f_unit, KA)
# print(np.amax(np.absolute(np.concatenate((Fx, Fy)))))
# %%
H1, eigD1 = MD.Hessian(xv_equ, yv_equ, A0, Nv, Nf, f_unit, KA)
V1, eigDv1 = MD.Stiffness(xv_equ, yv_equ, Nv, Nf, f_unit, KA)
H0, eigD0 = MD.Hessian(xv_ori, yv_ori, A0, Nv, Nf, f_unit, KA)
V0, eigDv0 = MD.Stiffness(xv_ori, yv_ori, Nv, Nf, f_unit, KA)
# %%
print(eigD1)
print(eigDv1)
# %%
fp = open('./Eigenvalues.txt', 'w')
for data in eigD0:
    fp.write('%.16e\n' % data)
for data in eigDv0:
    fp.write('%.16e\n' % data)
for data in eigD1:
    fp.write('%.16e\n' % data)
for data in eigDv1:
    fp.write('%.16e\n' % data)
fp.close()

# %%
ConfigPlot(xv_all, yv_all, f_unit, 1, './Config.png')