import gdstk
# from stl import Mesh
import numpy as np
# import numpy-stl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Load GDS file
gds_file = "gdsFileName.gds"
lib = gdstk.read_gds(gds_file)
top_cell = lib.top_level()[0]  # Assuming there's only one top cell

# Parameters for extrusion
layer_height = 1.0  # Set the height for extrusion
z_offset = 0.0  # Z offset if needed

# Collect vertices and faces for the 3D model
vertices = []
faces = []
ii = 0
print(len(top_cell.polygons))
for polygon in top_cell.polygons:
    print(ii)
    ii += 1
    if ii > 10001:
        break
    points = polygon.points
    base_index = len(vertices)

    # Add base vertices
    for point in points:
        vertices.append([point[0], point[1], z_offset])

    # Add top vertices (extrusion)
    for point in points:
        vertices.append([point[0], point[1], z_offset + layer_height])

    # Create faces (side walls)
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        faces.append([base_index + i, base_index + next_i, base_index + i + len(points)])
        faces.append([base_index + next_i, base_index + next_i + len(points), base_index + i + len(points)])

    # Create faces (top and bottom)
    bottom_face = [base_index + i for i in range(len(points))]
    top_face = [base_index + i + len(points) for i in range(len(points))]
    faces.append(bottom_face)
    faces.append(top_face[::-1])  # Reverse to maintain proper orientation

# Convert to numpy arrays
vertices = np.array(vertices)[:10000]
facets = np.array(faces, dtype=object)[:10000]

print(vertices, facets)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a list of polygons from the facets and vertices
polygons = [[vertices[vertex] for vertex in facet] for facet in facets]

# Create a Poly3DCollection object
poly3d = Poly3DCollection(polygons, alpha=0.5, edgecolor='r')

# Add the collection to the Axes
ax.add_collection3d(poly3d)

# Set the limits of the axes
ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])

# Optionally, you can set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
# Create the mesh
# gds_mesh = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
# for i, face in enumerate(faces):
#     for j in range(3):
#         gds_mesh.vectors[i][j] = vertices[face[j]]
#
# # Write to STL file
# stl_file = "output.stl"
# gds_mesh.save(stl_file)