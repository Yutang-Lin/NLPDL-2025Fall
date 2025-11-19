import torch.nn
lstm = torch.nn.LSTMCell(input_size=10, hidden_size=20)
print(lstm.state_dict().keys())