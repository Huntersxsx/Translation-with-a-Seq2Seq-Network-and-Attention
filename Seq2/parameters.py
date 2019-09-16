import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

DROPOUT = 0.1

teacher_forcing_ratio = 0.5
Learning_Rate = 0.01

Print_Interval = 1000
Plot_Interval = 100

HIDDEN_SIZE = 256

