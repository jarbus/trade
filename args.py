import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Execute Trading Environment.')
    parser.add_argument('--ip', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--punish-coeff', type=float, default=2)
    parser.add_argument('--random-start', action="store_true")
    parser.add_argument('--respawn', action="store_true")
    parser.add_argument('--punish', action="store_true")
    return parser.parse_args()
