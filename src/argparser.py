import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='Number of epochs to train. Default is 50.')

    parser.add_argument('--lr',
                        type=float,
                        default=5e-4,
                        help='Default learning rate is 0.001.')

    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='Default patience is 5.')

    parser.add_argument('--upsample',
                        type=str,
                        default='1v1',
                        choices=['1v1', '1v10', '1v50'],
                        help='Default task type is E1')


    return parser.parse_args()