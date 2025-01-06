import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- System setup ---
g = 9.81  # acceleration due to gravity (m/s^2)
l1 = 1.0  # length of pendulum 1 (m)
l2 = 1.0  # length of pendulum 2 (m)
m1 = 1.0  # mass of pendulum 1 (kg)
m2 = 1.0  # mass of pendulum 2 (kg)

a = g * (m1 + m2) / (m1 * l1)
b = (m2 * g) / (m1 * l1) 
d = 1 / (m2* l2)
c = g / l2

dt = 0.1
A = np.array([[0,1,0,0],
              [-a,0,b,0],
              [0,0,0,1],
              [0,0,-c,0]])

B = np.array([[0],
              [0],
              [0],
              [d]])
nx = A.shape[0]  # number of states = 2
nu = 1           # number of inputs = 1

# MPC horizon
N = 10

# Constraints
u_max = 5.0
u_min = -5.0


# Cost weights
Q = np.diag([3.0, 1.0,3.0,1.0])  # State tracking cost
R = 0.01                 # Input cost
# Reference
r = [np.pi/6,0,np.pi/6,0]                  # Desired position

# Initial state
x0 = np.array([0, 0, 0, 0])    # Start at position=0, velocity=0

# Simulation parameters
sim_time = 5.0  # seconds
steps = int(sim_time / dt)

# For storing simulation data
x_history = []
u_history = []

x = x0.copy()

for t in range(steps):
    # Create optimization variables
    X = cp.Variable((nx, N+1))  # predicted states
    U = cp.Variable((nu, N))    # predicted inputs
    
    # Define the cost function and constraints
    cost = 0
    constraints = []
    
    # Initial condition constraint
    constraints += [X[:, 0] == x]
    
    for k in range(N):
        # Cost: state deviation from ref & input cost
        # We only care about position deviation from r for x1,
        # but we can penalize velocity as well to help settle
        cost += cp.quad_form(X[:, k] - np.array([r]), Q)
        cost += R * cp.square(U[:, k])
        
        # System dynamics constraints
        constraints += [X[:, k+1] == A @ X[:, k] + B @ U[:, k]]
        
        # Input constraints
        constraints += [U[:, k] <= u_max, U[:, k] >= u_min]
    
    # Terminal cost (optional, can help if you define P matrix)
    # cost += cp.quad_form(X[:, N] - np.array([r, 0]), P) # if you have a terminal weight P
    
    # Solve the optimization
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, warm_start=True)
    
    # Obtain the first control input and apply to the system
    u_opt = U.value[0, 0]  # the optimal input at time k=0
    x = A @ x + B @ np.array([u_opt])
    
    # Store data
    x_history.append(x.copy())
    u_history.append(u_opt)

x_history = np.array(x_history)
tgrid = np.arange(steps) * dt

# --- Plot results ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(tgrid, x_history[:,0], label='Position')
plt.axhline(r, color='r', linestyle='--', label='Ref position = 5')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Tracking')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(tgrid, u_history, label='Control Input (accel)')
plt.axhline(u_max, color='k', linestyle='--', label='u_max')
plt.axhline(u_min, color='k', linestyle='--', label='u_min')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Control Input')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Example simulation data (replace with actual data)
time = np.linspace(0, 5, 51)        # Time vector (0 to 5 seconds, 51 points)
reference = 5.0 * np.ones_like(time)  # Constant reference position
position = 5 - 3 * np.exp(-time)     # Simulated position (replace with actual simulation output)
control = np.clip(2 * np.exp(-time), -5, 5)  # Simulated control input (replace with actual control)

# Define evaluation functions
def tracking_error(position, reference):
    """Compute tracking error (absolute)."""
    error = reference - position
    return error

def control_effort(control):
    """Compute total control effort (sum of absolute control values)."""
    effort = np.sum(np.abs(control))
    return effort

def settling_time(position, reference, time, tolerance=0.05):
    """
    Calculate settling time.
    Time required for position to remain within a given tolerance around the reference.
    """
    target_range = reference[0] * (1 - tolerance), reference[0] * (1 + tolerance)
    within_tolerance = np.where((position >= target_range[0]) & (position <= target_range[1]))[0]
    if within_tolerance.size > 0:
        return time[within_tolerance[0]]
    else:
        return np.nan  # Settling time not reached

def overshoot(position, reference):
    """
    Calculate overshoot.
    Maximum deviation above the reference as a percentage of the reference.
    """
    max_pos = np.max(position)
    os = (max_pos - reference[0]) / reference[0] * 100 if max_pos > reference[0] else 0
    return os

def steady_state_error(position, reference):
    """Calculate steady-state error at the end of the simulation."""
    return reference[-1] - position[-1]

def check_constraints(control, control_min=-5, control_max=5):
    """Check for constraint violations."""
    violations = {
        "below_min": np.sum(control < control_min),
        "above_max": np.sum(control > control_max)
    }
    return violations

# Evaluate performance
error = tracking_error(position, reference)
effort = control_effort(control)
settling = settling_time(position, reference, time)
os = overshoot(position, reference)
ss_error = steady_state_error(position, reference)
violations = check_constraints(control)

# Display results
print("Performance Metrics:")
print(f"Tracking Error (Final): {error[-1]:.4f}")
print(f"Total Control Effort: {effort:.4f}")
print(f"Settling Time: {settling:.4f} seconds")
print(f"Overshoot: {os:.2f}%")
print(f"Steady-State Error: {ss_error:.4f}")
print(f"Constraint Violations: {violations}")

