import os
import itertools

latent_dims = [4,8,16,32,64,128,256]
rand_seeds = [25, 32, 1234, 2020, 53]
include_demos = [True, False]
corr_coeff_params = [0, 1]
        
test_array = list(itertools.product(latent_dims, rand_seeds, include_demos, corr_coeff_params))
with open('odernn_exp_params.txt', 'w') as f:
    for hidden_size, random_seed, context_input, corr_coeff_param in test_array:
        name = 'odernn_sz{0}_rand{1}_corr{2}_context{3}_sepsis_training'.format(hidden_size,random_seed, corr_coeff_param, str(context_input))
        f.writelines('--autoencoder ODERNN --domain sepsis -o folder_name {0} -o hidden_size {1} -o random_seed {2} -o context_input {3} -o corr_coeff_param {4}\n'.format(name, hidden_size, random_seed, context_input, corr_coeff_param))         
