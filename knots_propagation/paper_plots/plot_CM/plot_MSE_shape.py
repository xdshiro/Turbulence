from knots_propagation.paper_plots.plots_functions_general import *
import seaborn as sns





data = np.load("MSE_005.npz")
cm = data['cm']
class_labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
                'T1', 'T2']
label = ('MSE_005')
plot_confusion_matrix(cm, class_labels, label=label)