from knots_propagation.paper_plots_knots_ML_shaping.plots_functions_general import *


sigma = 0.25
foils_paths = [
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_14.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_16.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30both.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_optimized.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30oneX.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_15oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_trefoil_standard_12.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_trefoil_optimized.npy',

]
foils_paths = [
	# f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_14.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_16.npy',
	# f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_standard_18.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30both.npy',
	# f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30oneZ.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_optimized.npy',
	# f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_pm_03_z.npy',
	f'data_all_hopfs_basis_turb\\hopf_spectr_{sigma}_30oneX.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_15oneZ.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_standard_12.npy',
	# f'data_all_hopfs_basis_turb\\hopf_field_{sigma}_trefoil_optimized.npy',
]
# foils_paths = [
# 	'data_all_hopfs_basis\\hopf_spectr__1e-40_standard_14.npy',
# ]
for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)

	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -6, 6, 0, 6, every_ticks=True)
# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
