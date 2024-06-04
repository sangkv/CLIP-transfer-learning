import torch
import clip

def VisionTransformerCLIP(model='ViT-B/32'):
    model, preprocess = clip.load(model, device='cpu')
    return model.visual, preprocess

class ViT(torch.nn.Module):
    def __init__(self, clipViT, num_class):
        super(ViT, self).__init__()
        self.feature_extractor = clipViT
        self.fc = torch.nn.Linear(512, num_class)
    
    def forward(self, input):
        features = self.feature_extractor(input)
        output = self.fc(features)
        return output
