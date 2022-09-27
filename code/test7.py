#save&load model
import torch
import torchvision.models as models

#save&load model weights: torch stores params in state_dict -> persisted via torch.save:
model=models.vgg16(pretrained=True)
torch.save(model.state_dict(),"model_weights.pth")
#load my weights: make instance(model) -> load params=load_state_dict():
model=models.vgg16()#not load def weights.
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()#important

#save&load models w/shapes:
torch.save(model,"model.pth")
#load it
model=torch.load("model.pth")
