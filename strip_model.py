import argparse
import torch
#from models.yolo import Model

parser = argparse.ArgumentParser(description='strip model')
parser.add_argument('--input-checkpoint', type=str, default='checkpoints/helmet-m/best.pt', help='input checkpoint.')
parser.add_argument('--output-checkpoint', type=str, default='checkpoints/helmet-m/strip-best.pth', help='output checkpoint.')
args = parser.parse_args()


def strip_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.input_checkpoint, map_location=device)
    torch.save(model['model'].state_dict(), args.output_checkpoint)

if __name__ == '__main__':
    strip_model()
