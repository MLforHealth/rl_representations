import os
import itertools

learning_rates = ['1e-5', '1e-4', '1e-3']
demographics = [0, 1]
num_nodes = [64, 128, 256]
weight_decays = [0, 0.15, 0.25]
optim_type = ['adam','sgd']

test_array = list(itertools.product(learning_rates, num_nodes, weight_decays, optim_type, demographics))
with open('BC_exp_params.txt', 'w') as f:
    for lr, nodes, wd, optim, demo in test_array:
        if demo:
            name = 'BC_l{0}_n{1}_w{2}_{3}_withDemo'.format(lr, nodes, wd, optim)
            f.writelines('--storage_folder {0} --learning_rate {1} --num_nodes {2} --weight_decay {3} --optimizer_type {4} --demographics \n'.format(name,lr,nodes,wd,optim))
        else:
            name = 'BC_l{0}_n{1}_w{2}_{3}'.format(lr, nodes, wd, optim)
            f.writelines('--storage_folder {0} --learning_rate {1} --num_nodes {2} --weight_decay {3} --optimizer_type {4} \n'.format(name,lr,nodes,wd,optim))