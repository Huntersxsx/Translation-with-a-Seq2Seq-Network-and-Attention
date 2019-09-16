from parameters import *
from prepare import *
from model import *
from train import *
from evaluate import *


encoder1 = EncoderRNN(input_size=input_lang.n_words, hidden_size=HIDDEN_SIZE).to(DEVICE)
attn_decoder1 = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_lang.n_words, dropout_p=DROPOUT).to(DEVICE)

trainIters(encoder1, attn_decoder1, n_iters=75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)
