import numpy as np
import matplotlib.pyplot as plt

# Constants
m_e_keV = 511  # Mass of electron (keV)
epsilon_0_per_nm = 8.854e-3  # Permittivity of free space (F/nm)
c = 3e8  # Speed of light in vacuum (m/s)
angstrom_to_m = 1e-10  # Conversion factor from Angstrom to meters
m_e_10eminus31 = 9.10938356  # Mass of electron
e_in_10eminus19 = 1.60217663 # Elementary charge

# Given parameters
E_keV = 60  # Energy in keV
d_angstrom = 10  # Initial distance from proton in Angstrom
detector_distance_mm = 150   # Distance to detector in cm

# Convert units
detector_distance_A = detector_distance_mm * 1e7  # Distance to detector in A

# Calculate initial velocity of the electron
v_m = np.sqrt(2 * E_keV / m_e_keV) * c  # Initial velocity of the electron

# Time it takes for the electron to reach the detector in nano-seconds
t_ns = detector_distance_A / v_m /10 #divided by 10 to convert to ns (A is 1e-10 m)

# Electron accelartion due to proton
def electric_acceleration_in_1e22(r_in_A):
    # F = m * a 
    # a = F / m
    # F = e^2 / (4 * pi * epsilon_0 * r^2)
    # 1e-19**2 / (1e-9 * 1e-10**2 * 1e-31) = 1e22  
    return -e_in_10eminus19**2 / (4 * np.pi * epsilon_0_per_nm * r_in_A**2 * m_e_10eminus31)


# Integrate the electron's trajectory
def electron_trajectory(d, dt=1):
    """timesteps are in 1e-20 seconds, d in A"""
    x_in_A = d
    initial_z_in_A = -10
    z_in_A = initial_z_in_A
    vx_in_m = 0
    vz = v_m
    trajectory = []
    
    
    while (z_in_A < -initial_z_in_A):
        r = np.sqrt(x_in_A**2 + z_in_A**2)
        a = electric_acceleration_in_1e22(r)
        ax = a * x_in_A / r
        vx_in_m += ax * dt * 1e4
        x_in_A += vx_in_m * dt * 1e-10 #1e-18 but A is 1e-10 m
        z_in_A += vz * dt *1e-10
        trajectory.append((x_in_A, z_in_A))
    
    #rest distance to detector
    rest_distance_A = detector_distance_A - z_in_A
    rest_time_1eminus10 = rest_distance_A / vz 
    z_in_A += vz * rest_time_1eminus10
    x_in_A += vx_in_m * rest_time_1eminus10
    trajectory.append((x_in_A, z_in_A))
    
    return np.array(trajectory)

trajectory = electron_trajectory(d_angstrom)
x_final, y_final = trajectory[-1]

# Calculate the 2D distance between proton and electron's landing position
distance = x_final


print(f"The 2D distance between the electron's start and the electron's landing position is {distance:.2e} angstrom(s)")

d_angstrom = 1
minimum_measurable_distance = 10*1e-6 # 10 micrometers
while True:
    d_angstrom += 1
    trajectory = electron_trajectory(d_angstrom)
    x_final, y_final = trajectory[-1]
    distance = x_final - d_angstrom
    if np.abs(distance*1e-10) < np.abs(minimum_measurable_distance):
        print(f"The electron with a starting x distance in A of {d_angstrom} will land within the minimum measurable distance of {minimum_measurable_distance:.2e} meters")
        break



# Plot the trajectory
plt.scatter(trajectory[:,0][:-1], trajectory[:,1][:-1]/1e10)
plt.xlabel('x (A)')
plt.ylabel('z (meters)')
plt.title('Trajectory of Electron Near a Proton')
plt.grid(True)
plt.savefig('electron_trajectory.png')
