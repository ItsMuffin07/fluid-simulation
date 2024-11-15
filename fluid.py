import numba as nb
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set number of threads for parallel processing
NUM_THREADS = multiprocessing.cpu_count()

@nb.njit(nopython=True, parallel=True)
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
                if s[i, j] != 0: # This cell is fluid
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


@nb.jit(nopython=True, parallel = True)
def extrapolate(u, v):
    for i in range(u.shape[0]):
        u[i, 0] = u[i, 1]
        u[i, -1] = u[i, -2]
    for j in range(v.shape[1]):
        v[0, j] = v[1, j]
        v[-1, j] = v[-2, j]
    return u, v


@nb.jit(nopython=True, parallel=True)
def advect_vel(u, v, s, h, dt):
    newU = u.copy()
    newV = v.copy()
    for i in range(1, u.shape[0]):
        for j in range(1, u.shape[1]):
            if s[i, j] != 0 and s[i - 1, j] != 0 and j < u.shape[1] - 1:
                x = i * h
                y = j * h + 0.5 * h
                u_val = u[i, j]
                v_val = (v[i, j] + v[i, j + 1] + v[i - 1, j] + v[i - 1, j + 1]) * 0.25
                x = x - dt * u_val
                y = y - dt * v_val
                newU[i, j] = sample_field(x, y, u, h, 'u')
            if s[i, j] != 0 and s[i, j - 1] != 0 and i < u.shape[0] - 1:
                x = i * h + 0.5 * h
                y = j * h
                u_val = (u[i, j] + u[i + 1, j] + u[i, j - 1] + u[i + 1, j - 1]) * 0.25
                v_val = v[i, j]
                x = x - dt * u_val
                y = y - dt * v_val
                newV[i, j] = sample_field(x, y, v, h, 'v')
    return newU, newV


@nb.jit(nopython=True, parallel=True)
def advect_smoke(m, u, v, s, h, dt):
    newM = m.copy()
    for i in range(1, m.shape[0] - 1):
        for j in range(1, m.shape[1] - 1):
            if s[i, j] != 0:
                u_val = (u[i, j] + u[i + 1, j]) * 0.5
                v_val = (v[i, j] + v[i, j + 1]) * 0.5
                x = i * h + 0.5 * h - dt * u_val
                y = j * h + 0.5 * h - dt * v_val
                newM[i, j] = sample_field(x, y, m, h, 'm')
    return newM


@nb.jit(nopython=True, parallel=True)
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


@nb.njit(nopython=True, parallel=True)
def calculate_force_optimized(s, p, h, numX, numY):
    force = np.zeros(2, dtype=np.float64)

    # Pre-compute neighbor offsets
    neighbors = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int32)

    for i in nb.prange(2, numX - 1):
        force_thread = np.zeros(2, dtype=np.float64)
        for j in range(2, numY - 1):
            if s[i, j] == 0:  # Obstacle cell
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < numX and 0 <= nj < numY and s[ni, nj] != 0:
                        pressure_force = p[ni, nj] * h
                        force_thread[0] += -pressure_force * di
                        force_thread[1] += -pressure_force * dj
        force += force_thread

    magnitude = np.sqrt(force[0] ** 2 + force[1] ** 2)
    direction = np.arctan2(force[1], force[0])

    return force, magnitude, direction, np.degrees(direction)


import numba as nb
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


# Previous functions remain the same...

class Fluid:
    def __init__(self, density: float, numX: int, numY: int, h: float, atmosphere: float = 1000):
        self.density = density
        self.numX = numX
        self.numY = numY
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
        self.atmosphere = atmosphere
        self.executor = ThreadPoolExecutor(max_workers=NUM_THREADS)

        rect_height = self.numY // 8
        self.start_y = self.numY // 2 - rect_height // 2
        self.end_y = self.start_y + rect_height

    def simulate(self, dt: float, numIters: int) -> None:
        start = time.time()

        # Use thread pool for smoke addition and pressure initialization
        def init_simulation():
            self.m[1, self.start_y:self.end_y] = 1.0
            self.p.fill(self.atmosphere)

        self.executor.submit(init_simulation)

        # Main simulation steps
        futures = []

        # Solve incompressibility
        futures.append(self.executor.submit(
            lambda: solve_incompressibility(self.u, self.v, self.p, self.s,
                                            self.density, self.h, dt, numIters, 1.9)))

        # Wait for incompressibility to complete before continuing
        self.u, self.v, self.p = futures[-1].result()

        # Extrapolate and advect in parallel
        futures = []
        futures.append(self.executor.submit(lambda: extrapolate(self.u, self.v)))
        futures.append(self.executor.submit(lambda: advect_vel(self.u, self.v, self.s, self.h, dt)))
        futures.append(self.executor.submit(lambda: advect_smoke(self.m, self.u, self.v, self.s, self.h, dt)))

        # Collect results
        self.u, self.v = futures[0].result()
        self.u, self.v = futures[1].result()
        self.m = futures[2].result()

    def fill(self, fluidSpeed: float) -> None:
        # Fill scenario with fluid
        self.u[:, :] = 0
        self.v[:, :] = 0
        self.m[:, :] = 0
        self.s[:, :] = 1  # Fluid everywhere

        self.s[0, :] = 0  # Left wall (required)

        self.u[1, :] = fluidSpeed  # Intake fluid speed

    def calculate_force(self) -> dict:
        force, magnitude, direction, direction_deg = calculate_force_optimized(
            self.s, self.p, self.h, self.numX, self.numY
        )

        return {
            'force_vector': force,
            'magnitude': magnitude,
            'direction_rad': direction,
            'direction_deg': direction_deg
        }

    def __del__(self):
        self.executor.shutdown()