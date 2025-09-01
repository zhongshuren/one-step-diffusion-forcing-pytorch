import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--num_eval', default=8, type=int)
parser.add_argument('--lr', default='3e-4', type=float)
parser.add_argument('--flow_steps', default='1', type=int)
parser.add_argument('--seed', default='7', type=int)