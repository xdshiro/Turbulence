"""
GDSII to STL Conversion Script
===============================

Created by: Dmitrii Tsvetkov (Duke ECE)
Date: 8/30/2024

Description:
-------------
This script converts GDSII files into 3D STL files, which are commonly used for 3D printing and CAD applications.
It reads in a list of GDSII files, extracts the polygon data, and extrudes them into 3D shapes with a specified
height (in millimeters). The resulting 3D models are then saved as STL files.

Key Features:
--------------
- Supports multiple GDSII files, each with a custom extrusion height.
- Option to visualize the 2D layout of the GDSII files before extrusion.
- Automatically calculates and prints the bounds of the generated 3D mesh.
- Saves each 3D mesh as an STL file with the same name as the original GDSII file.

Usage:
------
1. Modify the `gds_file_names` list to include the names of the GDSII files you want to convert.
2. Set the corresponding heights in the `heights_each_file` list.
3. Run the script. The STL files will be saved in the same directory as the GDSII files.

Dependencies:
--------------
- gdstk: For reading and processing GDSII files.
- matplotlib: For optional 2D visualization of the GDSII layout.
- numpy: For array manipulation and mathematical operations.
- numpy-stl: For creating and saving the STL files.

Notes:
------
- Ensure that the GDSII files are in the same directory as the script, or provide the full path to the files.
- The extrusion height should be specified in millimeters for consistency with 3D printing standards.

"""

from stl import mesh
import gdstk
import matplotlib.pyplot as plt
import numpy as np

gds_file_names = ["For_Dima.gds", ]
heights_each_file = [1700e-6, ]  # in mm

plot_2D = True  # if we want to see the structure ourselves


# Function to create a 3D mesh for a polygon
def polygon_to_3d(polygon, height):
    points = polygon.points
    num_points = len(points)
    
    # Create the top and bottom faces
    top_face = np.hstack([points, np.ones((num_points, 1)) * height])
    bottom_face = np.hstack([points, np.zeros((num_points, 1))])
    
    # Initialize an empty list to hold the vertices and faces for the STL
    vertices = []
    faces = []
    
    # Add the top and bottom faces to vertices
    vertices.extend(top_face)
    vertices.extend(bottom_face)
    
    # Create faces for the top and bottom polygons
    for i in range(1, num_points - 1):
        faces.append([0, i, i + 1])  # Top face
        faces.append([num_points, num_points + i, num_points + i + 1])  # Bottom face
    
    # Create side faces
    for i in range(num_points):
        next_i = (i + 1) % num_points
        faces.append([i, next_i, num_points + next_i])  # First triangle
        faces.append([i, num_points + next_i, num_points + i])  # Second triangle
    
    return np.array(vertices), np.array(faces)


# Read the GDSII file
for gds_file, height in zip(gds_file_names, heights_each_file):
    print(f'Working on file:    {gds_file[:-4]}')
    lib = gdstk.read_gds(gds_file)

    # Prepare the vertices and faces for the STL
    stl_vertices = []
    stl_faces = []
    
    # Iterate over each cell and convert polygons to 3D meshes
    for cell in lib.cells:
        print(f"Cell name: {cell.name}")
        for polygon in cell.polygons:
            vertices, faces = polygon_to_3d(polygon, height)
            # Update the faces indices relative to the current length of stl_vertices
            faces += len(stl_vertices)
            # Append to the STL vertices and faces lists
            stl_vertices.extend(vertices)
            stl_faces.extend(faces)
    
    # Plot the first cell
    cell = lib.cells[0]  # Assuming you want to plot the first cell
    if plot_2D:
        for polygon in cell.polygons:
            points = polygon.points
            plt.fill(points[:, 0], points[:, 1], alpha=0.6)  # Plot the polygon
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    # Convert to numpy arrays
    stl_vertices = np.array(stl_vertices)
    stl_faces = np.array(stl_faces)
    
    # Create the mesh object
    gds_mesh = mesh.Mesh(np.zeros(stl_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(stl_faces):
        for j in range(3):
            gds_mesh.vectors[i][j] = stl_vertices[f[j]]
    # Check the bounds of the mesh (min and max coordinates)
    min_coords = np.min(gds_mesh.points, axis=0)
    max_coords = np.max(gds_mesh.points, axis=0)
    print(f"Mesh bounds for {gds_file}:")
    print("X range:", min_coords[0], "-", max_coords[0])
    print("Y range:", min_coords[1], "-", max_coords[1])
    print("Z range:", min_coords[2], "-", max_coords[2])
    
    
    # Save the mesh to an STL file
    gds_mesh.save(f'{gds_file[:-4]}.stl')
    
    print(f"STL file {gds_file[:-4]} created successfully!")
