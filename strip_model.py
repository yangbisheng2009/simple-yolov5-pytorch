import argparse
import torch

parser = argparse.ArgumentParser(description='strip model')
parser.add_argument('--input', type=str, default='checkpoints/helmet-m/best.pt', help='input checkpoint.')
parser.add_argument('--output', type=str, default='checkpoints/helmet-m/strip-best.pth', help='output checkpoint.')
args = parser.parse_args()


def main():
    x = torch.load(args.input, map_location=torch.device('cpu'))
    torch.save(x['model'].half().state_dict(), args.output)

if __name__ == '__main__':
    main()
