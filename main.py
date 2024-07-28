import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import numpy as np
from scipy import ndimage
import time

from fluid import Fluid
from shapes import Shapes

matplotlib.use('TkAgg')

"""
Simple fluid simulation by Jayden Fung

---------------------------------------

This is an Eulerian fluid simulation which utilises a 2D grid-based plane to simulate the flow of an
incompressible, inviscid fluid. To use this simulation, modify parameters within the scene dictionary
and the setup_scene() function.

---------------------------------------

Keybindings:
p : Show pressure field
m : Show smoke field
v : Show velocity field
+ or = : Increase amount of arrows in velocity field
- : Decrease amount of arrows in velocity field
[ : Decrease arrow size in velocity field
] : Increase arrow size in velocity field
"""

# Modify this to change features of simulation
scene = {
    'dt': 1.0 / 60.0,  # Duration of each timestep
    'numIters': 100,  # Higher increases accuracy at the cost of performance
    'overRelaxation': 1.90,
    'paused': False,
    'showObstacle': True,
    'showPressure': False,
    'showSmoke': True,
    'showVelocity': False,
    'fluidSpeed': 1,  # Intake fluid velocity
    'resolution': 500,  # How many vertical pixels
    'screenWidth': 16,
    'screenHeight': 9,
    'density': 1000,  # Fluid density, affects pressure
    'frames': 500,  # Amount of frames to run, set to None if you do not want to finish program
    'arrowSpacing': 23,
    'arrowScale': 95,
    'record': False,  # When False, view mode enabled
    'fileName': 'naca_4412_fluid_simulation_smoke_force.mp4'  # File stored in videos directory
}

# Do not modify these - used for interla purposes
simulation_metrics = {
    'frameNr': 0,
    'totalTime': 0.0,
}


def setup_scene():
    global fluid, arrow_location

    res = scene['resolution']

    domain_height = 1.0
    domain_width = domain_height * scene['screenWidth'] / scene['screenHeight']

    h = domain_height / res

    numX = int(domain_width / h)
    numY = int(domain_height / h)

    density = scene['density']

    fluid = Fluid(density, numX, numY, h)

    fluid.fill(fluidSpeed=scene['fluidSpeed'])  # Fill up everywhere with fluid

    # Set a circular obstacle
    # set_obstacle(Shapes.circle(0.5, 0.5, 0.15))

    # Add NACA airfoil obstacle
    center_x = 0.1
    center_y = 0.5  # Center vertically
    chord = 0.9
    angle_of_attack = np.radians(-5)  # positive is anticlockwise
    naca_number = '4412'  # NACA airfoil number

    set_obstacle(Shapes.naca_airfoil(center_x, center_y, chord, angle_of_attack, naca_number))
    # Note that the centre is actually the leftmost point of the airfoil

    # Set location of force arrow
    arrow_location = (center_x + chord / 2, center_y)


def set_obstacle(shape_func):
    for i in range(1, fluid.numX - 1):
        for j in range(1, fluid.numY - 1):
            x = (i + 0.5) * fluid.h
            y = (j + 0.5) * fluid.h

            if shape_func(x, y):
                fluid.s[i, j] = 0.0
                fluid.m[i, j] = 0.0
                fluid.u[i, j] = fluid.u[i + 1, j] = 0
                fluid.v[i, j] = fluid.v[i, j + 1] = 0


def update(frame):
    if not scene['paused']:
        fluid.simulate(scene['dt'], scene['numIters'])
        simulation_metrics['frameNr'] += 1

    plt.clf()

    if scene['showSmoke']:
        plt.imshow(fluid.m.T, origin='lower', cmap='viridis', vmin=0, vmax=1,
                   extent=(0.0, fluid.numX * fluid.h, 0.0, fluid.numY * fluid.h))
        plt.colorbar(label='Smoke Density')
        plt.title(f"Frame: {simulation_metrics['frameNr']} - Smoke")
    elif scene['showPressure']:
        plt.imshow(fluid.p.T, origin='lower', cmap='RdBu_r',
                   extent=(0.0, fluid.numX * fluid.h, 0.0, fluid.numY * fluid.h))
        plt.colorbar(label='Pressure')
        plt.title(f"Frame: {simulation_metrics['frameNr']} - Pressure")
    elif scene['showVelocity']:
        # Calculate velocity magnitude
        velocity_mag = np.sqrt(fluid.u ** 2 + fluid.v ** 2)

        # Create a mesh grid for quiver plot
        x = np.arange(0, fluid.numX) * fluid.h
        y = np.arange(0, fluid.numY) * fluid.h
        X, Y = np.meshgrid(x, y)

        # Plot velocity magnitude
        plt.imshow(velocity_mag.T, origin='lower', cmap='viridis',
                   extent=(0.0, fluid.numX * fluid.h, 0.0, fluid.numY * fluid.h))
        plt.colorbar(label='Velocity Magnitude')

        # Plot velocity vectors with adjustable spacing and size
        spacing = scene['arrowSpacing']
        skip = (slice(None, None, spacing), slice(None, None, spacing))
        plt.quiver(X[skip], Y[skip], fluid.u.T[skip], fluid.v.T[skip],
                   scale=scene['arrowScale'], color='white', alpha=0.8,
                   width=0.002, headwidth=3, headlength=4)

        plt.title(f"Frame: {simulation_metrics['frameNr']} - Velocity")

    if scene['showObstacle']:
        # Create a mask of the obstacle
        obstacle_mask = 1 - fluid.s

        # Find the outline of the obstacle
        outline = obstacle_mask - ndimage.binary_erosion(obstacle_mask)

        # Get the coordinates of the outline
        outline_coords = np.where(outline.T)  # Transpose to match imshow orientation

        # Scale the coordinates to match the extent of the imshow plot
        y_coords = outline_coords[0] * fluid.h
        x_coords = outline_coords[1] * fluid.h

        # Plot the outline
        plt.scatter(x_coords, y_coords, color='red', s=1, alpha=1)

        # Calculate and plot force vector
        force_info = fluid.calculate_force()
        force_magnitude = force_info['magnitude']
        force_direction = force_info['direction_rad']

        # Scale the arrow length based on force magnitude
        max_arrow_length = 0.2  # Adjust this value to change the maximum arrow length
        arrow_length = min(force_magnitude / 1000, max_arrow_length)  # Adjust scaling factor as needed

        # Calculate arrow end point
        dx = arrow_length * np.cos(force_direction)
        dy = arrow_length * np.sin(force_direction)

        # Plot the arrow
        center_x, center_y = arrow_location
        plt.arrow(center_x, center_y, dx, dy, color='red', width=0.005,
                  head_width=0.02, head_length=0.02, zorder=5)

        # Calculate angle in degrees, with 0 degrees being upwards
        angle_deg = (-np.degrees(force_direction) + 90) % 360

        # Add force magnitude and angle labels
        plt.text(center_x - 0.05, center_y - 0.05, f'F = {force_magnitude:.2f} \n{angle_deg:.1f}°',
                 color='red', fontsize=10, ha='right', va='bottom')


def on_key_press(event):
    global scene
    if event.key == 'p':
        scene['showPressure'] = True
        scene['showSmoke'] = False
        scene['showVelocity'] = False
    elif event.key == 'm':
        scene['showPressure'] = False
        scene['showSmoke'] = True
        scene['showVelocity'] = False
    elif event.key == 'v':
        scene['showPressure'] = False
        scene['showSmoke'] = False
        scene['showVelocity'] = True
    elif event.key == '+' or event.key == '=':
        scene['arrowSpacing'] = max(2, scene['arrowSpacing'] - 1)
    elif event.key == '-':
        scene['arrowSpacing'] += 1
    elif event.key == '[':
        scene['arrowScale'] += 5
    elif event.key == ']':
        scene['arrowScale'] = max(10, scene['arrowScale'] - 5)
    elif event.key == ' ':
        scene['paused'] = not scene['paused']


def animate(frame):
    start = time.time()
    ax.clear()
    update(frame)
    end = time.time()
    timestep = end - start
    simulation_metrics['totalTime'] += timestep
    # print(f"Timestep: {end-start}s")
    # print(f"Average timestep: {simulation_metrics['totalTime']  / simulation_metrics['frameNr']}s")
    return ax,


setup_scene()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(scene['screenWidth'], scene['screenHeight']))
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Create the animation
anim = FuncAnimation(fig, animate, frames=scene['frames'], interval=20, blit=False)

if scene['record']:
    # Set up the writer
    writer = FFMpegWriter(fps=1 / scene['dt'], metadata=dict(artist='Me'), bitrate=1800)

    # Save the animation
    anim.save(str('/Users/jaydenfung/PycharmProjects/fluid_simulation/videos/' + scene['fileName']), writer=writer)
else:
    plt.show()
