import yaml
import torch

from models.yolo import Model

m = torch.load('checkpoints/driver-m/best.pt', map_location=torch.device('cpu'))
print(m['model'].names)


with open('configs/driver/driver-m.yaml') as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)

model = Model(data_dict).to('cpu')
model.names = []
print(model.names)
#model.load_state_dict(m['model'].state_dict())
