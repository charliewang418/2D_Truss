#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
#from numba import jit

def Force(xv, yv, A0, Nv, Nf, f_unit, KA):
    Fx = np.zeros(Nv)
    Fy = np.zeros(Nv)

    ift = np.array([1, 2, 0], dtype = 'int16')
    jft = np.array([2, 0, 1], dtype = 'int16')

    for nf in range(Nf):
        f_sub = f_unit[nf, :]
        xf = xv[f_sub]
        yf = yv[f_sub]
        A_vert = xf * yf[ift] - yf * xf[ift]
        Af = np.sum(A_vert) / 2.0
        dA = KA * (Af - A0[nf])
        # Eval = Eval + 0.5 * KA * dA**2
        
        dx_A = xf[ift] - xf[jft]
        dy_A = yf[jft] - yf[ift]
        Fx[f_sub] += 0.5 * dA * dy_A
        Fy[f_sub] += 0.5 * dA * dx_A

    return Fx, Fy


def FIRE(xv, yv, A0, Nv, Nf, f_unit, KA):
    # FIRE parameters
    Fthresh = 1e-10
    dt_md = 0.01 * np.sqrt(KA)
    Nt = 100000000 # maximum fire md steps
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    vx = np.zeros(Nv)
    vy = np.zeros(Nv)

    Fx, Fy = Force(xv, yv, A0, Nv, Nf, f_unit, KA)
    F_tot = np.mean(np.sqrt(Fx**2 + Fy**2))
    # putting a threshold on total force
    if F_tot < Fthresh:
        return xv, yv
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy)
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                xv -= vx * dt_half
                yv -= vy * dt_half
                vx = np.zeros(Nv)
                vy = np.zeros(Nv)

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2)) / np.sqrt(np.sum(Fx**2 + Fy**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        xv += vx * dt
        yv += vy * dt

        Fx, Fy = Force(xv, yv, A0, Nv, Nf, f_unit, KA)

        F_tot = np.mean(np.sqrt(Fx**2 + Fy**2))
        # putting a threshold on total force
        if F_tot < Fthresh:
            break

        vx += Fx * dt_half
        vy += Fy * dt_half

    print("Total Time Step: %d" % nt)
    print("Mean Force: %.3e" % F_tot)

    return xv, yv



def Hessian(xv, yv, A0, Nv, Nf, f_unit, KA):
    ift = np.array([1, 2, 0], dtype = 'int16')
    jft = np.array([2, 0, 1], dtype = 'int16')

    d2Adr1dr2 = np.zeros((2 * Nv, 2 * Nv), dtype = 'float64')

    for nf in np.arange(Nf):
        f_sub = f_unit[nf, :]
        xf = xv[f_sub]
        yf = yv[f_sub]
        A_vert = xf * yf[ift] - yf * xf[ift]
        Af = np.sum(A_vert) / 2
        dA = Af - A0[nf]
        dx_A = xf[jft] - xf[ift]
        dy_A = yf[ift] - yf[jft]
        for sidx1 in np.arange(3):
            ix = f_sub[sidx1]
            iy = ix + Nv
            iy0 = f_sub[jft[sidx1]] + Nv
            iy2 = f_sub[ift[sidx1]] + Nv
            
            d2Adr1dr2[ix, iy2] = d2Adr1dr2[ix, iy2] + dA
            d2Adr1dr2[ix, iy0] = d2Adr1dr2[ix, iy0] - dA
            
            d2Adr1dr2[ix, ix] = d2Adr1dr2[ix, ix] + dy_A[sidx1] * dy_A[sidx1] / 2
            d2Adr1dr2[ix, iy] = d2Adr1dr2[ix, iy] + dy_A[sidx1] * dx_A[sidx1]
            d2Adr1dr2[iy, iy] = d2Adr1dr2[iy, iy] + dx_A[sidx1] * dx_A[sidx1] / 2
            for sidx2 in np.arange(sidx1+1, 3):
                ix2 = f_sub[sidx2]
                iy2 = ix2 + Nv
                d2Adr1dr2[ix, ix2] = d2Adr1dr2[ix, ix2] + dy_A[sidx1] * dy_A[sidx2]
                d2Adr1dr2[ix, iy2] = d2Adr1dr2[ix, iy2] + dy_A[sidx1] * dx_A[sidx2]
                d2Adr1dr2[ix2, iy] = d2Adr1dr2[ix2, iy] + dx_A[sidx1] * dy_A[sidx2]
                d2Adr1dr2[iy, iy2] = d2Adr1dr2[iy, iy2] + dx_A[sidx1] * dx_A[sidx2]
 
    d2Adr1dr2 = d2Adr1dr2 * KA / 4

    H = d2Adr1dr2 + d2Adr1dr2.T
    eigD, _ = np.linalg.eigh(H)
    idx = eigD.argsort()
    eigD = eigD[idx]

    return H, eigD


def Stiffness(xv, yv, Nv, Nf, f_unit, KA):
    ift = np.array([1, 2, 0], dtype = 'int16')
    jft = np.array([2, 0, 1], dtype = 'int16')

    d2Adr1dr2 = np.zeros((2 * Nv, 2 * Nv), dtype = 'float64')

    for nf in np.arange(Nf):
        f_sub = f_unit[nf, :]
        xf = xv[f_sub]
        yf = yv[f_sub]
        dAdx = 0.5 * (yf[ift] - yf[jft])
        dAdy = 0.5 * (xf[jft] - xf[ift])
        for sidx1 in np.arange(3):
            ix = f_sub[sidx1]
            iy = ix + Nv

            d2Adr1dr2[ix, ix] = d2Adr1dr2[ix, ix] + dAdx[sidx1] * dAdx[sidx1]
            d2Adr1dr2[ix, iy] = d2Adr1dr2[ix, iy] + dAdx[sidx1] * dAdy[sidx1]
            d2Adr1dr2[iy, ix] = d2Adr1dr2[iy, ix] + dAdy[sidx1] * dAdx[sidx1]
            d2Adr1dr2[iy, iy] = d2Adr1dr2[iy, iy] + dAdy[sidx1] * dAdy[sidx1]

            for sidx2 in np.arange(sidx1+1, 3):
                ix2 = f_sub[sidx2]
                iy2 = ix2 + Nv
                d2Adr1dr2[ix, ix2] = d2Adr1dr2[ix, ix2] + dAdx[sidx1] * dAdx[sidx2]
                d2Adr1dr2[ix, iy2] = d2Adr1dr2[ix, iy2] + dAdx[sidx1] * dAdy[sidx2]
                d2Adr1dr2[iy, ix2] = d2Adr1dr2[iy, ix2] + dAdy[sidx1] * dAdx[sidx2]
                d2Adr1dr2[iy, iy2] = d2Adr1dr2[iy, iy2] + dAdy[sidx1] * dAdy[sidx2]
                d2Adr1dr2[ix2, ix] = d2Adr1dr2[ix2, ix] + dAdx[sidx2] * dAdx[sidx1]
                d2Adr1dr2[ix2, iy] = d2Adr1dr2[ix2, iy] + dAdx[sidx2] * dAdy[sidx1]
                d2Adr1dr2[iy2, ix] = d2Adr1dr2[iy2, ix] + dAdy[sidx2] * dAdx[sidx1]
                d2Adr1dr2[iy2, iy] = d2Adr1dr2[iy2, iy] + dAdy[sidx2] * dAdy[sidx1]

    V = KA * (d2Adr1dr2 + d2Adr1dr2.T) / 2

    eigDv, _ = np.linalg.eigh(V)
    idx = eigDv.argsort()
    eigDv = eigDv[idx]

    return V, eigDv