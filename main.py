import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import pygame
import colorsys
import math
from pygame import gfxdraw

import numpy as np
from scipy import ndimage
import time

from fluid import Fluid
from shapes import Shapes

import logging
import time


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
    'fileName': 'naca_4412_fluid_simulation_smoke_force.mp4',  # File stored in videos directory
    'pyGame': True # When false, use matplotlib.pyplot animation instead
}

# Do not modify these - used for interla purposes
simulation_metrics = {
    'frameNr': 0,
    'totalTime': 0.0,
}

performance_metrics = {
    'event_handling': 0,
    'calculations': 0,
    'visualization': 0,
    'total_time': 0
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


class ObstacleOutline:
    def __init__(self, fluid, window_width, window_height):
        # Calculate outline coordinates once
        obstacle_mask = 1 - fluid.s
        outline = obstacle_mask - ndimage.binary_erosion(obstacle_mask)
        outline_coords = np.where(outline.T)

        # Vectorized conversion to pixel coordinates
        x_pixels = (outline_coords[1] * window_width / fluid.numX).astype(int)
        y_pixels = ((fluid.numY - outline_coords[0]) * window_height / fluid.numY).astype(int)

        # Create a surface for the outline
        self.outline_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        self.outline_surface.fill((0, 0, 0, 0))  # Fill with transparent background

        # Draw points with small rectangles for better visibility
        for x, y in zip(x_pixels, y_pixels):
            pygame.draw.rect(self.outline_surface, (255, 0, 0), (x, y, 2, 2))

    def draw(self, screen):
        # Blit the pre-rendered outline surface
        screen.blit(self.outline_surface, (0, 0))

setup_scene()

if scene['pyGame']:
    # Initialize Pygame
    pygame.init()

    # Calculate window size while maintaining aspect ratio
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = int(WINDOW_HEIGHT * scene['screenWidth'] / scene['screenHeight'])

    # Initialize obstacle outline class
    obstacle_outline = ObstacleOutline(fluid, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Set up display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Fluid Simulation")
    clock = pygame.time.Clock()


    class ColorMapper:
        def __init__(self):
            # Pre-compute color lookup tables
            self.viridis_lut = self._create_viridis_lut()
            self.rdbu_lut = self._create_rdbu_lut()

        @staticmethod
        def _create_viridis_lut(size=256):
            # Pre-compute viridis lookup table
            indices = np.linspace(0, 1, size)
            hue = 0.67 - (0.12 * indices)
            saturation = np.ones_like(indices)
            value = 0.2 + (0.8 * indices)

            return np.array([colorsys.hsv_to_rgb(h, s, v)
                             for h, s, v in zip(hue, saturation, value)]) * 255

        @staticmethod
        def _create_rdbu_lut(size=256):
            # Pre-compute RdBu lookup table
            indices = np.linspace(0, 1, size)
            lut = np.zeros((size, 3))

            mid = size // 2
            # Blue to white
            lut[:mid] = np.column_stack([
                np.linspace(0, 1, mid),
                np.linspace(0, 1, mid),
                np.ones(mid)
            ])
            # White to red
            lut[mid:] = np.column_stack([
                np.ones(size - mid),
                np.linspace(1, 0, size - mid),
                np.linspace(1, 0, size - mid)
            ])

            return lut * 255


    _color_mapper = None


    # Optimised
    def create_color_surface(data, colormap='viridis', vmin=None, vmax=None):
        global _color_mapper

        # Lazy initialization of ColorMapper
        if _color_mapper is None:
            _color_mapper = ColorMapper()

        data = np.flip(data, axis=1)

        # Normalize data using vectorized operations
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        normalized = np.clip((data - vmin) / (vmax - vmin if vmax != vmin else 1), 0, 1)

        # Convert normalized values to lookup table indices
        indices = (normalized * 255).astype(np.uint8)

        # Use pre-computed lookup tables
        if colormap == 'viridis':
            rgb_array = _color_mapper.viridis_lut[indices]
        else:  # 'RdBu_r'
            rgb_array = _color_mapper.rdbu_lut[indices]

        # Reshape to match input dimensions
        rgb_array = rgb_array.reshape(data.shape + (3,))

        # Create surface directly from uint8 array
        surface = pygame.surfarray.make_surface(rgb_array.astype(np.uint8))
        return pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))

        # Convert to surface (single conversion)
        rgb_array = (rgb_array * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(rgb_array)
        return pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))


    def draw_velocity_arrows(screen, fluid, spacing, scale):
        arrow_color = (255, 255, 255)
        for i in range(0, fluid.numX, spacing):
            for j in range(0, fluid.numY, spacing):
                x = int(i * WINDOW_WIDTH / fluid.numX)
                y = int((fluid.numY - j) * WINDOW_HEIGHT / fluid.numY)  # Modified this line

                u_val = fluid.u[i, j]
                v_val = -fluid.v[i, j]  # Negate v component to match coordinate system

                # Scale velocity for arrow length
                dx = int(u_val * scale)
                dy = int(v_val * scale)

                if abs(dx) > 1 or abs(dy) > 1:
                    end_x = x + dx
                    end_y = y + dy
                    pygame.draw.line(screen, arrow_color, (x, y), (end_x, end_y), 1)

                    # Draw arrowhead
                    if dx != 0 or dy != 0:
                        angle = math.atan2(dy, dx)
                        arr_len = math.sqrt(dx * dx + dy * dy)
                        arr_size = min(5, arr_len / 3)

                        pygame.draw.line(screen, arrow_color,
                                         (end_x, end_y),
                                         (end_x - arr_size * math.cos(angle + math.pi / 6),
                                          end_y - arr_size * math.sin(angle + math.pi / 6)), 1)
                        pygame.draw.line(screen, arrow_color,
                                         (end_x, end_y),
                                         (end_x - arr_size * math.cos(angle - math.pi / 6),
                                          end_y - arr_size * math.sin(angle - math.pi / 6)), 1)


    def draw_force_vector(screen, fluid, arrow_location):
        # Cache constants and perform calculations once
        RED = (255, 0, 0)
        ARROW_WIDTH = 2
        FONT_SIZE = 24
        TEXT_OFFSET_X = 60
        TEXT_OFFSET_Y = 30

        # Get force info (single calculation)
        force_info = fluid.calculate_force()
        force_magnitude = force_info['magnitude']
        force_direction = force_info['direction_rad']

        # Calculate position (convert coordinates only once)
        center_x = int(arrow_location[0] * WINDOW_WIDTH)
        center_y = int((1 - arrow_location[1]) * WINDOW_HEIGHT)

        # Calculate arrow length and direction
        max_arrow_length = WINDOW_WIDTH * 0.2
        arrow_length = min(force_magnitude / 1000, max_arrow_length)

        # Calculate end point (single trig calculation)
        cos_dir = math.cos(force_direction)
        sin_dir = math.sin(force_direction)
        dx = int(arrow_length * cos_dir)
        dy = int(-arrow_length * sin_dir)  # Negated for coordinate system
        end_x = center_x + dx
        end_y = center_y + dy

        # Draw arrow (single line draw)
        pygame.draw.line(screen, RED, (center_x, center_y), (end_x, end_y), ARROW_WIDTH)

        # Calculate angle text (single calculation)
        angle_deg = f"{(-np.degrees(force_direction) + 90) % 360:.1f}°"
        magnitude_text = f"F = {force_magnitude:.2f}"

        # Create and render text once
        if not hasattr(draw_force_vector, 'font'):
            draw_force_vector.font = pygame.font.Font(None, FONT_SIZE)

        text = draw_force_vector.font.render(f"{magnitude_text} {angle_deg}", True, RED)
        screen.blit(text, (center_x - TEXT_OFFSET_X, center_y - TEXT_OFFSET_Y))


    def render_performance_metrics(screen):
        if not hasattr(render_performance_metrics, 'font'):
            render_performance_metrics.font = pygame.font.Font(None, 20)

        # Prevent division by zero for FPS calculation
        fps = 1000 / performance_metrics['total_time'] if performance_metrics['total_time'] > 0 else 0

        metrics_text = [
            f"Event handling: {performance_metrics['event_handling']:.1f}ms",
            f"Calculations: {performance_metrics['calculations']:.1f}ms",
            f"Visualization: {performance_metrics['visualization']:.1f}ms",
            f"Total time: {performance_metrics['total_time']:.1f}ms",
            f"FPS: {fps:.1f}"
        ]

        # Create background rectangle
        padding = 10
        line_height = 20
        text_width = 200
        text_height = (len(metrics_text) * line_height) + (padding * 2)

        # Draw semi-transparent background
        background_surface = pygame.Surface((text_width, text_height))
        background_surface.fill((0, 0, 0))
        background_surface.set_alpha(128)
        screen.blit(background_surface, (WINDOW_WIDTH - text_width - padding, padding))

        # Render text
        for i, text in enumerate(metrics_text):
            text_surface = render_performance_metrics.font.render(text, True, (255, 255, 255))
            screen.blit(text_surface,
                        (WINDOW_WIDTH - text_width,
                         padding + (i * line_height) + padding))


    running = True
    while running and (scene['frames'] is None or simulation_metrics['frameNr'] < scene['frames']):
        # Start timing the frame
        start = fstart = time.time()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    scene['showPressure'] = True
                    scene['showSmoke'] = False
                    scene['showVelocity'] = False
                elif event.key == pygame.K_m:
                    scene['showPressure'] = False
                    scene['showSmoke'] = True
                    scene['showVelocity'] = False
                elif event.key == pygame.K_v:
                    scene['showPressure'] = False
                    scene['showSmoke'] = False
                    scene['showVelocity'] = True
                elif event.key == pygame.K_EQUALS:
                    scene['arrowSpacing'] = max(2, scene['arrowSpacing'] - 1)
                elif event.key == pygame.K_MINUS:
                    scene['arrowSpacing'] += 1
                elif event.key == pygame.K_LEFTBRACKET:
                    scene['arrowScale'] += 5
                elif event.key == pygame.K_RIGHTBRACKET:
                    scene['arrowScale'] = max(10, scene['arrowScale'] - 5)
                elif event.key == pygame.K_SPACE:
                    scene['paused'] = not scene['paused']

        performance_metrics['event_handling'] = (time.time() - start) * 1000
        start = time.time()

        # Fluid simulation calculations
        if not scene['paused']:
            fluid.simulate(scene['dt'], scene['numIters'])
            simulation_metrics['frameNr'] += 1

        performance_metrics['calculations'] = (time.time() - start) * 1000
        start = time.time()

        # Visualization
        screen.fill((0, 0, 0))  # Clear screen

        # Draw the appropriate visualization
        if scene['showSmoke']:
            surface = create_color_surface(fluid.m, 'viridis', 0, 1)
            screen.blit(surface, (0, 0))
        elif scene['showPressure']:
            surface = create_color_surface(fluid.p, 'RdBu_r')
            screen.blit(surface, (0, 0))
        elif scene['showVelocity']:
            velocity_mag = np.sqrt(fluid.u ** 2 + fluid.v ** 2)
            surface = create_color_surface(velocity_mag, 'viridis')
            screen.blit(surface, (0, 0))
            draw_velocity_arrows(screen, fluid, scene['arrowSpacing'], scene['arrowScale'])

        # Draw obstacles and force vectors if enabled
        if scene['showObstacle']:
            obstacle_outline.draw(screen)
            draw_force_vector(screen, fluid, arrow_location)

        # Draw performance metrics
        render_performance_metrics(screen)

        performance_metrics['visualization'] = (time.time() - start) * 1000
        performance_metrics['total_time'] = (time.time() - fstart) * 1000

        # Update display
        pygame.display.flip()
        clock.tick(int(1 / scene['dt']))

    pygame.quit()

else:
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
