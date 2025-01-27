from knots_propagation.paper_plots_knots_ML_shaping.plots_functions_general import *
foils_paths = [
	'data_trefoils_turb\\hopf_spectr_before_1e-40_trefoil_standard_12.npy',
]

# foils_paths = [
# 	'foil4_spectr_XY_noturb_[2, 1, 2, 0].npy',
# ]


for path in foils_paths:
	foil4_spectrum_sorted = np.load(path)

	plot_shifted_paper_grid_spectrum(foil4_spectrum_sorted, -6, 6, 0, 6, every_ticks=False)
# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
