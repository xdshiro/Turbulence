import numpy as np
import sympy as sp

x = sp.symbols('x')
c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18 = sp.symbols(
	'c1 c2 c3 c4 c5 c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18')
coefficients = sp.Matrix([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18])
# Define the Huckel matrix for benzene
N = 18
# x = sp.symbols('x')
huckel_matrix = sp.zeros(N, N)
for i in range(N):
	huckel_matrix[i, i] = x
connections = [
	(0, 1), (0, 5),
	(1, 2), (1, 6),
	(2, 3), (2, 9),
	(3, 4),
	(4, 5),
	(6, 7), (6, 10),
	(7, 8), (7, 13),
	(8, 9),
	(10, 11), (10, 14),
	(11, 12), (11, 17),
	(12, 13),
	(14, 15),
	(15, 16),
	(16, 17)
]
for i, j in connections:
	huckel_matrix[i, j] = 1
	huckel_matrix[j, i] = 1

determinant = huckel_matrix.det()
roots_analytic = sp.roots(determinant, x)
remaining_poly = sp.Poly(determinant, x)
for root, multiplicity in roots_analytic.items():
	for i in range(multiplicity):
		remaining_poly = remaining_poly / (x - root)
remaining_poly = sp.simplify(remaining_poly)
print("Roots and their multiplicities:")
for root, multiplicity in roots_analytic.items():
	print(f"{root}: multiplicity {multiplicity}")

print(f'Remaining polynomial to solve: {remaining_poly}')
polynomial_coefficients = [8, 0, -35, 0, 35, 0, -11, 0, 1]
roots = np.roots(polynomial_coefficients)
print("The roots of the polynomial are:", roots)

# array of x values
expanded_roots_list = []
for root, multiplicity in roots_analytic.items():
	expanded_roots_list.extend([root] * multiplicity)
expanded_roots_array = np.array(expanded_roots_list)
# print(f'expanded roots array: {expanded_roots_array}')

x_set = np.concatenate([expanded_roots_array, roots])
# print(f'x_set = {x_set}')

# Huckel matrix with alpha, beta, E
alpha, beta, E = sp.symbols('alpha, beta, E')
E_set = [alpha - beta * x for x in x_set]

# make a new Huckel matrix to find coefficients
huckel_matrix_real = sp.zeros(N, N)
connections_2 = [(0, 1), (0, 5),
                 (1, 2), (1, 6),
                 (2, 3), (2, 9),
                 (3, 4),
                 (4, 5),
                 (6, 7), (6, 10),
                 (7, 8), (7, 13),
                 (8, 9),
                 (10, 11), (10, 14),
                 (11, 12), (11, 17),
                 (12, 13),
                 (14, 15),
                 (15, 16),
                 (16, 17)]
for i in range(N):
	huckel_matrix_real[i, i] = alpha - E
for i, j in connections_2:
	huckel_matrix_real[i, j] = beta
	huckel_matrix_real[j, i] = beta

print(f'Energy levels:')
for E_j in E_set:
	print(E_j)
	substituted_matrix = huckel_matrix_real.subs(E, E_j)
	results = substituted_matrix * coefficients
	equations = [sp.Eq(r, 0) for r in results]
	normalization_equation = sp.Eq(sum(c ** 2 for c in coefficients), 1)
	equations.append(normalization_equation)
	coefficients_solutions = sp.solve(equations, coefficients, dict=True)
	print("Corresponding coefficients:")
	for solution in coefficients_solutions:
		simplified_solution = {var: sp.simplify(expr) for var, expr in solution.items()}
		print(simplified_solution)
	print("\n")
