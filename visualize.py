import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Plotter:
    def __init__(self, l1=1.0, l2=1.0, theta1_init=np.pi / 4, theta2_init=np.pi / 2, dt=0.1, t_max=5):
        """
        Initialize the double pendulum plotter.

        Parameters:
        - l1: Length of the first pendulum arm.
        - l2: Length of the second pendulum arm.
        - theta1_init: Initial angle of the first pendulum (in radians).
        - theta2_init: Initial angle of the second pendulum (in radians).
        - dt: Time step for the simulation.
        - t_max: Total simulation time.
        """
        self.l1 = l1
        self.l2 = l2
        self.theta1_init = theta1_init
        self.theta2_init = theta2_init
        self.dt = dt
        self.t_max = t_max

        # Time vector
        self.time = np.arange(0, t_max, dt)

        # Generate example angles (replace these with actual simulation data if needed)
        self.theta1 = self.theta1_init 
        self.theta2 = self.theta2_init 

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid()
        

        # Plot elements
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.bob1, = self.ax.plot([], [], 'ro', markersize=8)
        self.bob2, = self.ax.plot([], [], 'ro', markersize=8)

    def get_positions(self, theta1, theta2):
        """
        Compute the Cartesian positions of the pendulum bobs.
        
        Parameters:
        - theta1: Angle of the first pendulum (in radians).
        - theta2: Angle of the second pendulum (in radians).

        Returns:
        - x1, y1: Coordinates of the first pendulum bob.
        - x2, y2: Coordinates of the second pendulum bob.
        """
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)
        return x1, y1, x2, y2

    def init_plot(self):
        """Initialize the plot for animation."""
        self.line.set_data([], [])
        self.bob1.set_data([], [])
        self.bob2.set_data([], [])
        return self.line, self.bob1, self.bob2

    def update_plot(self, frame):
        """Update the plot for each animation frame."""
        theta1_t, theta2_t = self.theta1[frame], self.theta2[frame]
        x1, y1, x2, y2 = self.get_positions(theta1_t, theta2_t)

        # Update line and bobs
        self.line.set_data([0, x1, x2], [0, y1, y2])
        self.bob1.set_data(x1, y1)
        self.bob2.set_data(x2, y2)
        self.fig.suptitle(f'Time: {frame * self.dt:.1f} s')
        return self.line, self.bob1, self.bob2

    def animate(self):
        """Run the animation."""
        self.anim = FuncAnimation(
            self.fig, self.update_plot, frames=len(self.time), init_func=self.init_plot,
            blit=True, interval= self.dt * 1000, repeat=False
        )
        
        plt.show()
