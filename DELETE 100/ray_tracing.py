import numpy as np
import matplotlib.pyplot as plt

def reflect(ray_direction, normal):
    """Reflects a ray using the law of reflection."""
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    reflected = ray_direction - 2 * np.dot(ray_direction, normal) * normal
    return reflected / np.linalg.norm(reflected)  # Return unit vector

# Define the incident ray and surface normal
incident_ray = np.array([1, -1])  # Incident ray direction
normal = np.array([0, 1])         # Surface normal pointing upward

# Compute the reflected ray
reflected_ray = reflect(incident_ray, normal)

# Plot the incident and reflected rays
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, *incident_ray, angles='xy', scale_units='xy', scale=1, color='r', label='Incident Ray')
plt.quiver(0, 0, *reflected_ray, angles='xy', scale_units='xy', scale=1, color='b', label='Reflected Ray')
plt.quiver(0, 0, *normal, angles='xy', scale_units='xy', scale=1, color='g', label='Normal')

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='k', linestyle='--')  # Surface
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title('Ray Reflection at a Surface')
plt.show()