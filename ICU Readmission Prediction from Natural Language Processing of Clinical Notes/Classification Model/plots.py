import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product
import numpy as np


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    
	epochs = [epoch for epoch in range(len(train_losses))]

	# set all the font sizes for plotting
	SMALL_SIZE = 16
	MEDIUM_SIZE = 18
	BIGGER_SIZE = 20

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
	axes[0].plot(epochs, train_losses, 'b', label='Training Loss')
	axes[0].plot(epochs, valid_losses, 'r', label='Validation Loss')
	axes[0].set_title('Loss Curve')
	axes[0].set(xlabel='epoch', ylabel='Loss')
	axes[0].legend(loc="best")

	axes[1].plot(epochs, train_accuracies, 'b', label='Training Accuracy')
	axes[1].plot(epochs, valid_accuracies, 'r', label='Validation Accuracy')
	axes[1].set_title('Accuracy Curve')
	axes[1].set(xlabel='epoch', ylabel='Accuracy')
	axes[1].legend(loc="best")

	fig.tight_layout()
	plt.show()


def plot_confusion_matrix(results, class_names):
    
	class ConfusionMatrixDisplay:
		def __init__(self, confusion_matrix, display_labels):
			self.confusion_matrix = confusion_matrix
			self.display_labels = display_labels

		def plot(self, include_values=True, cmap='viridis',
					xticks_rotation='horizontal', values_format=None, ax=None):

			if ax is None:
				fig, ax = plt.subplots()
			else:
				fig = ax.figure

			cm = self.confusion_matrix
			n_classes = cm.shape[0]
			self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
			self.text_ = None

			cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

			if include_values:
				self.text_ = np.empty_like(cm, dtype=object)
				if values_format is None:
					values_format = '.2g'

				# print text with appropriate color depending on background
				thresh = (cm.max() - cm.min()) / 2.
				for i, j in product(range(n_classes), range(n_classes)):
					color = cmap_max if cm[i, j] < thresh else cmap_min
					self.text_[i, j] = ax.text(j, i,
												format(cm[i, j], values_format),
												ha="center", va="center",
												color=color)

			fig.colorbar(self.im_, ax=ax)
			ax.set(xticks=np.arange(n_classes),
					yticks=np.arange(n_classes),
					xticklabels=self.display_labels,
					yticklabels=self.display_labels,
					ylabel="TRUE",
					xlabel="PREDICTED",
					title="Normalized Confusion Matrix")

			ax.set_ylim((n_classes - 0.5, -0.5))
			plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
			fig.tight_layout()
			plt.show()

			self.figure_ = fig
			self.ax_ = ax
			return self

	def plot_cm(results, labels,
								sample_weight=None, normalize='true',
								display_labels=None, include_values=True,
								xticks_rotation=45,
								values_format=None,
								cmap='Blues', ax=None):
		
		label_dict = {}
		for key, value in enumerate(labels):
			label_dict[key] = value

		y_true = np.asarray([label_dict[x[0]] for x in results])
		y_pred = np.asarray([label_dict[x[1]] for x in results])

		cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
								labels=labels, normalize=normalize)

		if display_labels is None:
			display_labels = labels

		disp = ConfusionMatrixDisplay(confusion_matrix=cm,
										display_labels=display_labels)
		return disp.plot(include_values=include_values,
							cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)


	plot_cm(results=results, labels=class_names)

def plot_roc_curve(fpr_t, tpr_t, test_auc):
	fig, ax = plt.subplots()
	fig.set_size_inches(10,6)
	lw = 2
	ax.plot(fpr_t, tpr_t, color='darkorange', lw=lw, label='Test (AUC = %0.2f)' % test_auc)
	ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.02])
	ax.set_xlabel('False Positive Rate', fontsize=16)
	ax.set_ylabel('True Positive Rate', fontsize=16)
	ax.set_title('ROC curve')
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.legend(loc="lower right", prop={'size': 14})
	plt.show()