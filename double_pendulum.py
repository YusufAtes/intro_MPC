import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from visualize import Plotter
from double_pendulum_equations import theta1_update, theta2_update
from double_pendulum_equations import pos_update, deriv
from double_pendulum_equations import derivs
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
u_max = 10.0
u_min = -10.0


# Cost weights
Q = np.diag([1.0,0.0,1.0,0.0])  # State tracking cost
R = 0.01             # Input cost
# Reference
r = np.array([np.pi/8,0.0,np.pi/6,0.0] )
# Initial state
x0 = np.array([0, 0, 0, 0])    # Start at position=0, velocity=0

# Simulation parameters
sim_time = 5.0  # seconds
steps = int(sim_time / dt)

time = np.arange(0, sim_time, dt)
# For storing simulation data
x_history = []
u_history = []
cost_history = []
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
        cost += cp.quad_form(X[:, k] - r, Q)
        cost += R * cp.square(U[:, k])
        
        # System dynamics constraints
        constraints += [X[:, k+1] == A @ X[:, k] + B @ U[:, k]]
        
        # Input constraints
        constraints += [U[:, k] <= u_max, U[:, k] >= u_min]
    
    # Terminal cost (optional, can help if you define P matrix)
    # cost += cp.quad_form(X[:, N] - np.array([r, 0]), P) # if you have a terminal weight P
    
    # Solve the optimization
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True)
    
    # Obtain the first control input and apply to the system
    # if U.value is None:
    #     u_opt = 0.0  # no control
    # else:
    u_opt = U.value[0, 0]  # the optimal input at time k=0
    #x = A @ x + B @ np.array([u_opt])
    # theta1 = x[0]
    # theta2 = x[2]
    # theta1_velocity = x[1]
    # theta2_velocity = x[3]
    # theta1, theta1_velocity = theta1_update(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,t,u_opt)
    # theta2, theta2_velocity = theta2_update(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,t,u_opt)
    # x = pos_update(deriv,x,time,l1,l2,m1,m2,g,u_opt)
    x = x + derivs(x,m1,m2,l1,l2,g,u_opt) * dt

    # Store data
    x_history.append(x.copy())
    u_history.append(u_opt)
    cost_history.append(problem.value)
    
x_history = np.array(x_history)
tgrid = np.arange(steps) * dt

# Define evaluation functions
def tracking_error(position, reference):
    """Compute tracking error (absolute)."""
    error = reference - position
    return np.mean(np.abs(error))

def control_effort(control):
    """Compute total control effort (sum of absolute control values)."""
    effort = np.sum(np.abs(control))
    return effort

def settling_time(position, reference, time, tolerance=0.05):
    """
    Calculate settling time.
    Time required for position to remain within a given tolerance around the reference.
    """
    target_range = reference * (1 - tolerance), reference * (1 + tolerance)
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
    os = ((max_pos - reference) / (reference + 0.000001)) * 100 if max_pos > reference else 0
    return os

def steady_state_error(position, reference):
    """Calculate steady-state error at the end of the simulation."""
    return reference - position[-1]

def check_constraints(control, control_min=-4.999, control_max=5.001):
    """Check for constraint violations."""
    violations = {
        "below_min": np.sum(control < control_min),
        "above_max": np.sum(control > control_max)
    }
    return violations

# Evaluate performance
pos_error1 = tracking_error(x_history[:,0], r[0])
effort = control_effort(np.array(u_history))
settling1 = settling_time(x_history[:,0], r[0], time)
os1 = overshoot(x_history[:,0], r[0])
ss_error1 = steady_state_error(x_history[:,0], r[0])
violations = check_constraints(np.array(u_history), u_min, u_max)

pos_error2 = tracking_error(x_history[:,2], r[2])
settling2 = settling_time(x_history[:,2], r[2], time)
ss_error2 = steady_state_error(x_history[:,2], r[2])
os2 = overshoot(x_history[:,2], r[2])

# Display results
print("Performance Metrics:")
print(f"Tracking Error theta 1: {pos_error1:.4f}")
print(f"Settling Time theta 1: {settling1:.4f} seconds")
print(f"Overshoot theta 1: {os1:.2f}%")
print(f"Steady-State Error theta 1: {ss_error1:.4f}")

print(f"Tracking Error theta 2: {pos_error2:.4f}")
print(f"Settling Time theta 2: {settling2:.4f} seconds")
print(f"Overshoot theta 2: {os2:.2f}%")
print(f"Steady-State Error theta 2: {ss_error2:.4f}")

print(f"Constraint Violations: {violations}")
print(f"Total Control Effort: {effort:.4f}")
print(f"Total Cost: {np.sum(cost_history):.4f}")

# --- Plot results ---
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(tgrid, x_history[:,0], label='Position')
plt.axhline(r[0], color='r', linestyle='--', label='Ref position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Tracking theta1')
plt.legend()
plt.grid(True)


plt.subplot(1,3,2)
plt.plot(tgrid, x_history[:,2], label='Position')
plt.axhline(r[2], color='r', linestyle='--', label='Ref position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Tracking theta2')
plt.legend()
plt.grid(True)


plt.subplot(1,3,3)
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

plotter = Plotter(l1=l1, l2=l2, theta1_init=x_history[:,0], theta2_init=x_history[:,2], dt=0.1, t_max=5)
plotter.animate()