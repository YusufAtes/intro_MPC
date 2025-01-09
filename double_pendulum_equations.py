import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#Get theta1 acceleration 
import numpy as np

def theta1_acceleration(m1, m2, l1, l2,
                        theta1, theta2,
                        theta1_velocity, theta2_velocity,
                        g, F):
    """
    Returns the second derivative (angular acceleration) of theta1,
    including an external horizontal force F acting on the second mass.
    """
    # Original terms (standard double-pendulum):
    mass1 = -g * (2*m1 + m2) * np.sin(theta1)
    mass2 = -m2 * g * np.sin(theta1 - 2*theta2)
    interaction = -2 * np.sin(theta1 - theta2) * m2 * (
        theta2_velocity**2 * l2
        + theta1_velocity**2 * l1 * np.cos(theta1 - theta2)
    )
    normalization = l1 * (2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2))

    # ---- NEW: Add external generalized force on mass2 for theta1 ----
    # Q_theta1 = F * l1 * cos(theta1)
    external_term = F * l1 * np.cos(theta1) * np.cos(theta2)

    # Combine them into the numerator:
    numerator = (mass1 + mass2 + interaction) + external_term

    theta1_ddot = numerator / normalization
    return theta1_ddot


def theta2_acceleration(m1, m2, l1, l2,
                        theta1, theta2,
                        theta1_velocity, theta2_velocity,
                        g, F):
    """
    Returns the second derivative (angular acceleration) of theta2,
    including an external horizontal force F acting on the second mass.
    """
    # Original terms (standard double-pendulum):
    system = 2 * np.sin(theta1 - theta2) * (
        theta1_velocity**2 * l1 * (m1 + m2)
        + g * (m1 + m2) * np.cos(theta1)
        + theta2_velocity**2 * l2 * m2 * np.cos(theta1 - theta2)
    )
    normalization = l2 * (2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2))
    # NOTE: check carefully if you want the same normalizing factor
    # as in your original code. Sometimes one uses l1*(...) for eqn1
    # and l2*(...) for eqn2. Make sure it matches your formula.

    # ---- NEW: Add external generalized force on mass2 for theta2 ----
    # Q_theta2 = F * l2 * cos(theta2)
    external_term = F * l2 * np.cos(theta2)

    # Combine them into the numerator:
    numerator = system + external_term

    theta2_ddot = numerator / (l2*(2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2)))
    return theta2_ddot


def theta1_update(m1, m2, l1, l2,
                  theta1, theta2,
                  theta1_velocity, theta2_velocity,
                  g, time_step, F):
    # Update theta1_velocity using the new acceleration
    theta1_velocity += time_step * theta1_acceleration(
        m1, m2, l1, l2,
        theta1, theta2,
        theta1_velocity, theta2_velocity,
        g, F
    )
    # Then update theta1
    theta1 += time_step * theta1_velocity
    return theta1, theta1_velocity


def theta2_update(m1, m2, l1, l2,
                  theta1, theta2,
                  theta1_velocity, theta2_velocity,
                  g, time_step, F):
    # Update theta2_velocity using the new acceleration
    theta2_velocity += time_step * theta2_acceleration(
        m1, m2, l1, l2,
        theta1, theta2,
        theta1_velocity, theta2_velocity,
        g, F
    )
    # Then update theta2
    theta2 += time_step * theta2_velocity
    return theta2, theta2_velocity

def deriv(y, t, L1, L2, m1, m2, g, F):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)+ F * m2 * np.cos(theta2+theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c +F * m2 * np.cos(theta2)) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def pos_update(deriv,y0,t,L1,L2,m1,m2,g,F):
    """Update the position of the pendulum."""
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, g, F))
    return y

def derivs(state,M1,M2,L1,L2,G,F):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] + state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    dydx[1] = (( M2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
                + M2 * G * np.sin(state[2]) * np.cos(delta)
                + M2 * L2 * state[3] * state[3] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[0])
                + F * M2 * np.cos(state[0] + state[2]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
                + (M1+M2) * G * np.sin(state[0]) * np.cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[2])
                + F * (M1+M2) * np.cos(state[0] + state[2]))
               / den2)

    return dydx

# #Run full double pendulum
# def double_pendulum(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,time_step,time_span):
#     theta1_list = [theta1]
#     theta2_list = [theta2]
    
#     for t in time_span:
#         theta1, theta1_velocity = theta1_update(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,time_step)
#         theta2, theta2_velocity = theta2_update(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,time_step)

#         theta1_list.append(theta1)
#         theta2_list.append(theta2)
    
#     x1 = l1*np.sin(theta1_list) #Pendulum 1 x
#     y1 = -l1*np.cos(theta1_list) #Pendulum 1 y

#     x2 = l1*np.sin(theta1_list) + l2*np.sin(theta2_list) #Pendulum 2 x
#     y2 = -l1*np.cos(theta1_list) - l2*np.cos(theta2_list) #Pendulum 2 y
    
#     return x1,y1,x2,y2


# #Define system parameters
# g = 9.8 #m/s^2

# m1 = 1 #kg
# m2 = 1 #kg

# l1 = 1 #m
# l2 = 1 #m

# theta1 = np.radians(0)
# theta2 = np.radians(0)

# theta1_velocity = 0 #m/s
# theta2_velocity = 0 #m/s

# theta1_list = [theta1]
# theta2_list = [theta2]

# time_step = 20/300

# time_span = np.linspace(0,20,300)
# x1,y1,x2,y2 = double_pendulum(m1,m2,l1,l2,theta1,theta2,theta1_velocity,theta2_velocity,g,time_step,time_span)

    
# counter=0
# images = []
# for i in range(0,len(x1)):
#     plt.figure(figsize = (6,6))

#     plt.figure(figsize = (6,6))
#     plt.plot([0,x1[i]],[0,y1[i]], "o-", color = "b", markersize = 7, linewidth=.7 )
#     plt.plot([x1[i],x2[i]],[y1[i],y2[i]], "o-", color = "b", markersize = 7, linewidth=.7 )
#     plt.title("Double Pendulum")
#     plt.xlim(-2.1,2.1)
#     plt.ylim(-2.1,2.1)
#     plt.savefig("frames/" + str(counter)+ ".png")
#     images.append(imageio.imread("frames/" + str(counter)+ ".png"))
#     counter += 1
#     plt.close()

# imageio.mimsave("double_pendulum.gif", images)
