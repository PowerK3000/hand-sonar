"""Numba-compiled forward kinematics. Same maths as skeleton.local_joints_reference (which the
tests compare against), written as scalar loops: the NumPy version spent ~1.4 ms per call in
per-op overhead on 21-joint arrays, which made live fitting too slow."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def _one(palm, fbones, splay, tbones, pose, out):
    out[0, 0] = 0.0; out[0, 1] = 0.0; out[0, 2] = 0.0
    for c in range(3):
        out[1, c] = palm[0, c]
    for i in range(4):
        root = 5 + 4 * i
        abd = pose[6 + 4 * i]
        lat = splay[i] + abd
        sl, cl = math.sin(lat), math.cos(lat)
        px, py, pz = palm[i + 1, 0], palm[i + 1, 1], palm[i + 1, 2]
        out[root, 0] = px; out[root, 1] = py; out[root, 2] = pz
        phi = 0.0
        for k in range(3):
            phi += pose[7 + 4 * i + k]
            cp, sp = math.cos(phi), math.sin(phi)
            L = fbones[i, k]
            px += L * sl * cp
            py += L * cl * cp
            pz -= L * sp
            out[root + 1 + k, 0] = px; out[root + 1 + k, 1] = py; out[root + 1 + k, 2] = pz
    az, el, roll, fm, fi = pose[22], pose[23], pose[24], pose[25], pose[26]
    ux = math.sin(az) * math.cos(el); uy = math.cos(az) * math.cos(el); uz = -math.sin(el)
    # a0 = u x (0,0,1) normalised; b0 = a0 x u
    ax, ay, az0 = uy, -ux, 0.0
    n = math.sqrt(ax * ax + ay * ay)
    if n < 1e-9:
        n = 1e-9
    ax /= n; ay /= n
    bx = ay * uz - az0 * uy; by = az0 * ux - ax * uz; bz = ax * uy - ay * ux
    cr, sr = math.cos(roll), math.sin(roll)
    kx = cr * ax + sr * bx; ky = cr * ay + sr * by; kz = cr * az0 + sr * bz
    px = palm[0, 0] + tbones[0] * ux; py = palm[0, 1] + tbones[0] * uy; pz = palm[0, 2] + tbones[0] * uz
    out[2, 0] = px; out[2, 1] = py; out[2, 2] = pz
    dx, dy, dz = ux, uy, uz
    for k in range(2):
        ang = fm if k == 0 else fi
        c, s = math.cos(ang), math.sin(ang)
        cx = ky * dz - kz * dy; cy = kz * dx - kx * dz; cz = kx * dy - ky * dx
        dot = kx * dx + ky * dy + kz * dz
        dx, dy, dz = (dx * c + cx * s + kx * dot * (1 - c), dy * c + cy * s + ky * dot * (1 - c),
                      dz * c + cz * s + kz * dot * (1 - c))
        L = tbones[1 + k]
        px += L * dx; py += L * dy; pz += L * dz
        out[3 + k, 0] = px; out[3 + k, 1] = py; out[3 + k, 2] = pz


@njit(cache=True)
def _rot(r, R):
    th = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if th < 1e-12:
        for i in range(3):
            for j in range(3):
                R[i, j] = 1.0 if i == j else 0.0
        return
    kx, ky, kz = r[0] / th, r[1] / th, r[2] / th
    c, s = math.cos(th), math.sin(th)
    v = 1 - c
    R[0, 0] = c + kx * kx * v; R[0, 1] = kx * ky * v - kz * s; R[0, 2] = kx * kz * v + ky * s
    R[1, 0] = ky * kx * v + kz * s; R[1, 1] = c + ky * ky * v; R[1, 2] = ky * kz * v - kx * s
    R[2, 0] = kz * kx * v - ky * s; R[2, 1] = kz * ky * v + kx * s; R[2, 2] = c + kz * kz * v


@njit(cache=True, parallel=False)
def local_batch(palm, fbones, splay, tbones, poses):
    B = poses.shape[0]
    out = np.empty((B, 21, 3))
    for b in prange(B):
        _one(palm, fbones, splay, tbones, poses[b], out[b])
    return out


@njit(cache=True, parallel=False)
def camera_batch(palm, fbones, splay, tbones, poses):
    B = poses.shape[0]
    out = np.empty((B, 21, 3))
    loc = np.empty((21, 3))
    R = np.empty((3, 3))
    for b in range(B):
        _one(palm, fbones, splay, tbones, poses[b], loc)
        _rot(poses[b, 0:3], R)
        for j in range(21):
            for i in range(3):
                out[b, j, i] = R[i, 0] * loc[j, 0] + R[i, 1] * loc[j, 1] + R[i, 2] * loc[j, 2] + poses[b, 3 + i]
    return out
