import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate, EarlyStopping
from plots import plot_learning_curves, plot_confusion_matrix, plot_roc_curve
from mydatasets import VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyVariableRNN, BERTVariableRNN

PATH_OUTPUT = "../../Data/processed/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NLP_technique = 'BERT'        # 'TF-IDF' or 'BERT'

if NLP_technique == 'TF-IDF':
	PATH_DATA = os.path.join(PATH_OUTPUT, 'TF-IDF/')
	num_features = 1000
	model = MyVariableRNN(num_features)
elif NLP_technique == 'BERT':
	PATH_DATA = os.path.join(PATH_OUTPUT, 'pretrained_BERT/Mean_selectedCols_no_note_processing/')
	num_features = 768
	model = BERTVariableRNN(num_features)
else:
	raise AssertionError("Wrong NLP technique!") 

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = os.path.join(PATH_DATA, "seqs.train")
PATH_TRAIN_LABELS = os.path.join(PATH_DATA, "labels.train")
PATH_VALID_SEQS = os.path.join(PATH_DATA, "seqs.validation")
PATH_VALID_LABELS = os.path.join(PATH_DATA, "labels.validation")
PATH_TEST_SEQS = os.path.join(PATH_DATA, "seqs.test")
PATH_TEST_LABELS = os.path.join(PATH_DATA, "labels.test")


PATH_OUTPUT = "../../Data/output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 50
BATCH_SIZE = 32	  # has to be 1 for any model other than variable RNN
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))


train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)


criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 8.0]))
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

# _______________________________________________________________________________________________
# initialize the early_stopping object
early_stopping = EarlyStopping(path_output=PATH_OUTPUT, save_file="MyVariableRNN.pth", patience=7)


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

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

class_names = ['No ICU revisit', 'ICU revisit']
_, _, test_results = evaluate(best_model, device, test_loader, criterion)

plot_confusion_matrix(test_results, class_names)


# predict_mortality
def predict_mortality(model, device, data_loader):
	model.eval()
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