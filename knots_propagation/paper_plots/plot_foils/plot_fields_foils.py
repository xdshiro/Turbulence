
from knots_propagation.paper_plots.plots_functions_general import *

foils_paths = [
	'foil4_field_XY_noturb_[2, 2, 2, 2].npy',
	'foil4_field_XY_noturb_[2, 2, 2, 0].npy',
	'foil4_field_XY_noturb_[2, 1, 2, 0].npy',
]
foils_paths = [
	'foil4_field_XY_noturb_[2, 2, 2, 2].npy',
]
for path in foils_paths:
	foil4_field = np.load(path)
	foil4_field = foil4_field / np.max(np.abs(foil4_field))
	XY_max = 30e-3 * 185 / 300 * 1e3
	X = [-XY_max, XY_max]
	Y = [-XY_max, XY_max]
	plot_field_both_paper(foil4_field, extend=[*X, *Y])