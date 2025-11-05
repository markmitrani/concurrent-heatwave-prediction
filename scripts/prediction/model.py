import os
import torch
import torch.nn as nn
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.utils.utils import download

def build_earthformer_model(config, input_len, checkpoint_url, save_dir = None):
    state_dict = {}
    if save_dir is not None:
        checkpoint_path = os.path.join(os.getcwd(), save_dir, "earthformer_earthnet2021.pt")
        #download(url=checkpoint_url, path=checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model = CuboidTransformerModel(input_shape=[input_len, 128, 128, 2], target_shape=[1, 128, 128, 64], **config)
    model_state = model.state_dict()

    compatible = None

    if save_dir is not None:
        compatible = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)

        # print(f"Missing Keys: {missing_keys}")
        # print(f"Unexpected Keys: {unexpected_keys}")
        
        compatible = {k: v for k, v in compatible.items() if k not in missing_keys}
            
        # Reset positional embeddings
        for name, param in model.named_parameters():
            if 'pos_embed' in name:
                nn.init.zeros_(param)
    
    return model, compatible

def build_full_model(num_classes, input_len, config, checkpoint_url, save_dir = None):
    base_model, compatible = build_earthformer_model(config, input_len, checkpoint_url, save_dir)
    # model = EarthformerClassifier(base_model, num_classes)
    model = EarthformerPredictor(base_model)
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

class EarthformerPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        print(f"target_shape[-1]: {self.model.target_shape[-1]}")
        self.pool = nn.AdaptiveMaxPool3d((1, 8, 8)) # Pool over T, H, W
        self.fc = nn.Sequential(
            nn.Linear(self.model.target_shape[-1] * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512,1),
            nn.Sigmoid()
        ) # (nr. channels) -> (nr. classes)
        
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.uniform_(m.bias) # mean is: 0.14672432839870453
                nn.init.zeros_(m.bias)

    # x shape before model pass: torch.Size([10, 5, 128, 128, 2])
    # x shape after model pass: torch.Size([10, 1, 128, 128, 64])
    # x shape after permutation: torch.Size([10, 64, 1, 128, 128])
    # x shape after pool: torch.Size([10, 64, 1, 8, 8])
    # x shape after squeeze: torch.Size([10, 64, 8, 8])

    # S_PCHA shapes:
    # S_PCHA shapes:
    # (8, 9200)
    # w:170, h:46
    # torch.Size([9200, 35, 128, 2])
    # Out shape: torch.Size([9200, 128, 128, 2])
    # x final shape: torch.Size([8700, 5, 128, 128, 2])
    # y final shape: torch.Size([8700])
    # target_shape[-1]: 64
    # x shape before model pass: torch.Size([10, 5, 128, 128, 2])
    # x shape after model pass: torch.Size([10, 1, 128, 128, 64])
    # x shape after permutation: torch.Size([10, 64, 1, 128, 128])
    # x shape after pool: torch.Size([10, 64, 1, 8, 8])
    # x shape after squeeze+flatten: torch.Size([10, 4096])
    # x shape after fc: torch.Size([10, 1])
    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool(x)
        x = x.squeeze(2).flatten(start_dim=1)
        x = self.fc(x)
        return x
