import argparse
import torch

parser = argparse.ArgumentParser(description='strip model')
parser.add_argument('--input-checkpoint', type=str, default='checkpoints/helmet-m/best.pt', help='input checkpoint.')
parser.add_argument('--output-checkpoint', type=str, default='checkpoints/helmet-m/strip-best.pth', help='output checkpoint.')
args = parser.parse_args()


def main():
    x = torch.load(args.input_checkpoint, map_location=torch.device('cpu'))
    print(x.keys())
    #x['model'].half()
    torch.save(x['model'].half().state_dict(), args.output_checkpoint)
    #torch.save(x['model'].half(), args.output_checkpoint)


if __name__ == '__main__':
    main()
