import numba as nb
import time
import numpy as np

@nb.njit(parallel=True)
def solve_incompressibility(u, v, p, s, density, h, dt, numIters, overRelaxation):
    cp = density * h / dt
    n, m = u.shape[0] - 1, u.shape[1] - 1

    # Precompute 1/s_sum for all cells
    inv_s_sum = np.zeros_like(s)
    for i in nb.prange(1, n):
        for j in range(1, m):
            s_sum = s[i - 1, j] + s[i + 1, j] + s[i, j - 1] + s[i, j + 1]
            if s[i, j] != 0 and s_sum != 0:
                inv_s_sum[i, j] = 1 / s_sum

    for _ in range(numIters):
        for i in nb.prange(1, n):
            for j in range(1, m):
                if s[i, j] != 0:
                    # Compute divergence
                    div = (u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j]) * inv_s_sum[i, j]

                    # Compute pressure change
                    p_change = -div * overRelaxation

                    # Update pressure
                    p[i, j] += cp * p_change

                    # Update velocities
                    u[i, j] -= s[i - 1, j] * p_change
                    u[i + 1, j] += s[i + 1, j] * p_change
                    v[i, j] -= s[i, j - 1] * p_change
                    v[i, j + 1] += s[i, j + 1] * p_change

    return u, v, p

@nb.jit(nopython=True)
def extrapolate(u, v):
    for i in range(u.shape[0]):
        u[i, 0] = u[i, 1]
        u[i, -1] = u[i, -2]
    for j in range(v.shape[1]):
        v[0, j] = v[1, j]
        v[-1, j] = v[-2, j]
    return u, v

@nb.jit(nopython=True)
def advect_vel(u, v, s, h, dt):
    newU = u.copy()
    newV = v.copy()
    for i in range(1, u.shape[0]):
        for j in range(1, u.shape[1]):
            if s[i, j] != 0 and s[i-1, j] != 0 and j < u.shape[1] - 1:
                x = i * h
                y = j * h + 0.5 * h
                u_val = u[i, j]
                v_val = (v[i, j] + v[i, j+1] + v[i-1, j] + v[i-1, j+1]) * 0.25
                x = x - dt * u_val
                y = y - dt * v_val
                newU[i, j] = sample_field(x, y, u, h, 'u')
            if s[i, j] != 0 and s[i, j-1] != 0 and i < u.shape[0] - 1:
                x = i * h + 0.5 * h
                y = j * h
                u_val = (u[i, j] + u[i+1, j] + u[i, j-1] + u[i+1, j-1]) * 0.25
                v_val = v[i, j]
                x = x - dt * u_val
                y = y - dt * v_val
                newV[i, j] = sample_field(x, y, v, h, 'v')
    return newU, newV

@nb.jit(nopython=True)
def advect_smoke(m, u, v, s, h, dt):
    newM = m.copy()
    for i in range(1, m.shape[0] - 1):
        for j in range(1, m.shape[1] - 1):
            if s[i, j] != 0:
                u_val = (u[i, j] + u[i+1, j]) * 0.5
                v_val = (v[i, j] + v[i, j+1]) * 0.5
                x = i * h + 0.5 * h - dt * u_val
                y = j * h + 0.5 * h - dt * v_val
                newM[i, j] = sample_field(x, y, m, h, 'm')
    return newM

@nb.jit(nopython=True)
def sample_field(x, y, field, h, field_type):
    h1 = 1.0 / h
    h2 = 0.5 * h
    dx, dy = 0, 0
    if field_type == 'u':
        dy = h2
    elif field_type == 'v':
        dx = h2
    elif field_type == 'm':
        dx, dy = h2, h2
    x = max(min(x, field.shape[0] * h), h)
    y = max(min(y, field.shape[1] * h), h)
    x0 = min(int((x - dx) * h1), field.shape[0] - 2)
    tx = ((x - dx) - x0 * h) * h1
    x1 = min(x0 + 1, field.shape[0] - 1)
    y0 = min(int((y - dy) * h1), field.shape[1] - 2)
    ty = ((y - dy) - y0 * h) * h1
    y1 = min(y0 + 1, field.shape[1] - 1)
    sx, sy = 1.0 - tx, 1.0 - ty
    return (sx * sy * field[x0, y0] +
            tx * sy * field[x1, y0] +
            tx * ty * field[x1, y1] +
            sx * ty * field[x0, y1])

class Fluid:
    def __init__(self, density, numX, numY, h):
        self.density = density
        self.numX = numX + 2
        self.numY = numY + 2
        self.numCells = self.numX * self.numY
        self.h = h
        self.u = np.zeros((self.numX, self.numY))
        self.v = np.zeros((self.numX, self.numY))
        self.newU = np.zeros((self.numX, self.numY))
        self.newV = np.zeros((self.numX, self.numY))
        self.p = np.zeros((self.numX, self.numY))
        self.s = np.zeros((self.numX, self.numY))
        self.m = np.zeros((self.numX, self.numY))
        self.newM = np.zeros((self.numX, self.numY))


        self.object_center = None  # Add this line


    def simulate(self, dt, numIters):
        rect_height = self.numY // 8
        start_y = self.numY // 2 - rect_height // 2
        end_y = start_y + rect_height
        self.m[1, start_y:end_y] = 1.0 # Add smoke
        self.p.fill(1000) # Default / atmospheric pressure at 1000
        self.u, self.v, self.p = solve_incompressibility(self.u, self.v, self.p, self.s, self.density, self.h, dt, numIters, 1.9)
        self.u, self.v = extrapolate(self.u, self.v)
        self.u, self.v = advect_vel(self.u, self.v, self.s, self.h, dt)
        self.m = advect_smoke(self.m, self.u, self.v, self.s, self.h, dt)

    def fill(self):
        # Fill scenario with fluid
        self.u[:, :] = 0
        self.v[:, :] = 0
        self.m[:, :] = 0
        self.s[:, :] = 1  # Fluid everywhere

    def set_object_center(self, center_x, center_y):
        self.object_center = (center_x, center_y)

    def calculate_force(self):
        force = np.zeros(2)  # [Fx, Fy]

        for i in range(2, self.numX - 1):
            for j in range(2, self.numY - 1):
                if self.s[i, j] == 0:  # This is an obstacle cell
                    # Check all neighboring cells
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ni, nj = i + di, j + dj
                        if self.numX > ni >= 0 != self.s[ni, nj] and 0 <= nj < self.numY:  # This is a fluid cell

                            pressure_force = self.p[ni, nj] * self.h
                            pressure_force_x = -1 * pressure_force * di
                            pressure_force_y = -1 * pressure_force * dj
                            force += pressure_force_x, pressure_force_y


        # Calculate magnitude and direction
        magnitude = np.linalg.norm(force)
        direction = np.arctan2(force[1], force[0])  # in radians

        # Convert direction to degrees
        direction_deg = np.degrees(direction)

        return {
            'force_vector': force,
            'magnitude': magnitude,
            'direction_rad': direction,
            'direction_deg': direction_deg
        }