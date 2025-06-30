import os
import torch
import torch.nn as nn
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.utils.utils import download

def build_earthformer_model(config, checkpoint_url, save_dir):
    checkpoint_path = os.path.join(os.getcwd(), save_dir, "earthformer_earthnet2021.pt")
    #download(url=checkpoint_url, path=checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model = CuboidTransformerModel(input_shape=[1, 128, 128, 2], target_shape=[1, 128, 128, 64], **config)
    model_state = model.state_dict()

    compatible = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)

    #print(f"Compatible Keys: {compatible.keys()}")
    print(f"Missing Keys: {missing_keys}")
    print(f"Unexpected Keys: {unexpected_keys}")
    print("Keys in compatible AND missing keys dicts:")
    for k in missing_keys:
        if k in compatible.items():
            print(k)
    compatible = {k: v for k, v in compatible.items() if k not in missing_keys}
    
    return model, compatible

def build_classifier(num_classes, config, checkpoint_url, save_dir):
    base_model, compatible = build_earthformer_model(config, checkpoint_url, save_dir)
    model = EarthformerClassifier(base_model, num_classes)
    return model, compatible

class EarthformerClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = base_model
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Pool over T, H, W
        self.classifier = nn.Linear(self.model.target_shape[-1], num_classes) # (nr. channels) -> (nr. classes)

    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool(x).squeeze()
        logits = self.classifier(x)        
        return logits
