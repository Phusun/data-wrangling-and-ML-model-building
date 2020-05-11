import torch
import torch.nn as nn

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.dim_red = 32              
		self.hidden = 16               
		self.drop = nn.Dropout(0.2)
		self.tanh = nn.Tanh()
		self.fc1 = nn.Linear(in_features=dim_input, out_features=self.dim_red)
		self.rnn = nn.GRU(input_size=self.dim_red, hidden_size=self.hidden, num_layers=1, batch_first=True)
		self.fc2 = nn.Linear(in_features=self.hidden, out_features=2)

	def forward(self, input_tuple):
		
		seqs, lengths = input_tuple

		batch_size, seq_len, _ = seqs.size()

		seqs = self.tanh(self.drop(self.fc1(seqs)))

		_, seqs = self.rnn(seqs)     # h_n

		seqs = seqs[-1, :, :]     # h_n is of shape (num_layers, batch, hidden_size)

		seqs = self.drop(seqs)

		seqs = self.fc2(seqs)

		return seqs


class BERTVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(BERTVariableRNN, self).__init__()
        self.dim_red = 64               
        self.hidden = 16                
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=dim_input, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.dim_red)
        self.rnn = nn.GRU(input_size=self.dim_red, hidden_size=self.hidden, num_layers=3, batch_first=True)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=2)

    def forward(self, input_tuple):
        
        seqs, lengths = input_tuple

        batch_size, seq_len, _ = seqs.size()

        seqs = self.tanh(self.fc3(self.fc1(seqs)))

        _, seqs = self.rnn(seqs)     # h_n

        seqs = seqs[-1, :, :]     # h_n is of shape (num_layers, batch, hidden_size)

        seqs = self.fc2(seqs)

        return seqs


class BERTCNN(nn.Module):
    def __init__(self):
        super(BERTCNN, self).__init__()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.1)        # 0.3
        self.drop2 = nn.Dropout(0.0)        #0.1
        self.relu = nn.ReLU()        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5)
        self.ln1 = nn.LayerNorm([8, 764])
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(in_features=32*92, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        
        x = self.maxpool1(self.ln1(self.relu(self.drop1(self.conv1(x)))))
        x = self.maxpool1(self.relu(self.drop1(self.conv2(x))))
        x = self.maxpool1(self.relu(self.drop1(self.conv3(x))))

        x = x.view(-1, 32*92)

        x = self.relu(self.drop1(self.fc1(x)))
        x = self.relu(self.drop1(self.fc2(x)))
        x = self.relu(self.drop2(self.fc3(x)))
        x = self.fc4(x)

        return x