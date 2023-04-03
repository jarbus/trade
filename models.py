
extrasmall = {
    "conv_filters": [[64, [1, 1], 1], [32, [1, 1], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [256, 256],
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_cell_size": 512,
    "lstm_use_prev_action": False,
    "max_seq_len": 50,
}

small = {
    "conv_filters": [[32, [3, 3], 1], [32, [3, 3], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [64, 64],
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_cell_size": 128,
    "lstm_use_prev_action": False,
    "max_seq_len": 50,
}

medium = {
    "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [256, 256],
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_cell_size": 256,
    "lstm_use_prev_action": False,
    "max_seq_len": 50,
}

large = {
    "conv_filters": [[128, [3, 3], 1], [128, [3, 3], 1], [128, [3, 3], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [256, 256],
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_cell_size": 512,
    "lstm_use_prev_action": False,
    "max_seq_len": 50,
}

