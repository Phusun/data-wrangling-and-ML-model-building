import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pickle

from utils import train, evaluate, EarlyStopping
from plots import plot_learning_curves, plot_confusion_matrix, plot_roc_curve
from mymodels import BERTCNN

#TODO: https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection

PATH_DATA = "../../Data/processed/pretrained_BERT/CNN_selectedCols_no_note_processing"
# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = os.path.join(PATH_DATA, "seqs.train")
PATH_TRAIN_LABELS = os.path.join(PATH_DATA, "labels.train")
PATH_VALID_SEQS = os.path.join(PATH_DATA, "seqs.validation")
PATH_VALID_LABELS = os.path.join(PATH_DATA, "labels.validation")
PATH_TEST_SEQS = os.path.join(PATH_DATA, "seqs.test")
PATH_TEST_LABELS = os.path.join(PATH_DATA, "labels.test")

# Path for saving model
PATH_OUTPUT = "../../Data/output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# Some parameters
NUM_EPOCHS = 50 #1
BATCH_SIZE = 64
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0 #0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Data loading
print('===> Loading entire datasets')
X_train = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
y_train = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
X_valid = pickle.load(open(PATH_VALID_SEQS, 'rb'))
y_valid = pickle.load(open(PATH_VALID_LABELS, 'rb'))
X_test = pickle.load(open(PATH_TEST_SEQS, 'rb'))
y_test = pickle.load(open(PATH_TEST_LABELS, 'rb'))

def load_CNN_data(X, y):
	# For conv1d, input data shape should (N, C, L): https://pytorch.org/docs/stable/nn.html#conv1d
	data = torch.from_numpy(X.astype('float32')).unsqueeze(1) # (N, L) --> (N, 1, L)
	target = torch.from_numpy(y).long()
	dataset = TensorDataset(data, target)
	return dataset

train_dataset = load_CNN_data(X_train, y_train)
valid_dataset = load_CNN_data(X_valid, y_valid)
test_dataset = load_CNN_data(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = BERTCNN()
save_file = 'MyCNN.pth'


criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 8.0]))
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

# _______________________________________________________________________________________________
# initialize the early_stopping object
early_stopping = EarlyStopping(path_output=PATH_OUTPUT, save_file=save_file, patience=7)

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	# early_stopping needs the validation loss to check if it has decresed, 
	# and if it has, it will make a checkpoint of the current model
	early_stopping(valid_loss, model)
	
	if early_stopping.early_stop:
		print("Early stopping")
		break
# __________________________________________________________________________________________________

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, save_file))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
_, _, test_results = evaluate(best_model, device, test_loader, criterion)

class_names = ['No ICU revisit', 'ICU revisit']
plot_confusion_matrix(test_results, class_names)


def predict_mortality(model, device, data_loader):
	model.eval()
	# TODO: Evaluate the data (from data_loader) using model,
	# TODO: return a List of probabilities
	probas = []
	with torch.no_grad():
		for input, _ in data_loader:

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)

			output = model(input)

			probas.append(torch.softmax(output, 1)[0][1].item())
	
	return probas


test_prob = predict_mortality(best_model, device, test_loader)

# # Performance metrics and plotting of results
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
import numpy as np

y_true = np.asarray([x[0] for x in test_results])
y_pred = np.asarray([x[1] for x in test_results])
y_pred_prob = np.asarray(test_prob)

test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)
test_auc = roc_auc_score(y_true, y_pred_prob)
fpr_t, tpr_t, _ = roc_curve(y_true, y_pred_prob)

print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test AUC: {test_auc}')

plot_roc_curve(fpr_t, tpr_t, test_auc)