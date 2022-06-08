grep -oP "\s*\K(num_agents = |timesteps|samples|freq|batch_size =|dist_coeff|lr|local_dir|name=).*$" tune.py
