grep -oP "\s*\K(num_agents = |timesteps|samples|freq|batch_size =|dist_coeff|lr).*$" tune.py
