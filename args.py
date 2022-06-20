import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Execute Trading Environment.')
    parser.add_argument('--ip', type=str)
    parser.add_argument('--name', type=str, default="test")
    parser.add_argument('--class-name', type=str, default="class")
    parser.add_argument('--spawn-agents', type=str, help="[center, corner]")
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--day-night-cycle', action="store_true")
    parser.add_argument('--day-steps', type=int, default=20)
    parser.add_argument('--night-time-death-prob', type=float, default=0.1)
    parser.add_argument('--checkpoint-interval', type=int, default=500)
    parser.add_argument('--punish-coeff', type=float, default=2.0)
    parser.add_argument('--dist-coeff', type=float, default=1.0)
    parser.add_argument('--gx', type=int, default=5)
    parser.add_argument('--gy', type=int, default=5)
    parser.add_argument('--twonn-coeff', type=float, default=1.0)
    parser.add_argument('--num-steps', type=float, default=1_000_000)
    parser.add_argument('--move-coeff', type=float, default=0.0)
    parser.add_argument('--death-prob', type=float, default=0.1)
    parser.add_argument('--survival-bonus', type=float, default=0.0)
    parser.add_argument('--self-other-frames', action="store_true")
    parser.add_argument('--health-baseline', action="store_true")
    parser.add_argument('--parameter-sharing', action="store_true")
    parser.add_argument('--respawn', action="store_true")
    parser.add_argument('--punish', action="store_true")
    parser.add_argument('--pbt', action="store_true")
    return parser.parse_args()
