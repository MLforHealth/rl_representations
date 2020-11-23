import os
import itertools

learning_rates = ['1e-4']#,'5e-4','1e-3']
h_sizes = [4,8,16,32,64,128,256]
rand_seeds = [25, 32, 1234, 2020, 53]

test_array = list(itertools.product(learning_rates,rand_seeds,h_sizes))
with open('rnn_exp_params.txt', 'w') as f:
    for lr, rand_seed, sz in test_array:
        name = 'rnn_s{0}_l{1}_rand{2}_sepsis'.format(sz,lr,rand_seed)
        f.writelines('--autoencoder RNN --domain sepsis -o resume True -o folder_name {0} -o hidden_size {1} -o autoencoder_lr {2} -o random_seed {3} \n'.format(name,sz,lr,rand_seed))
