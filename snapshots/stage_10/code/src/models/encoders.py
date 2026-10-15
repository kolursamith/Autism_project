"""Spatial encoders: timm ImageNet backbones and the custom from-scratch CNN.

Every encoder owns its preprocessing: uint8 (N, 3, H, W) -> float in [0, 1] -> resize to its input size (if needed)
-> normalise with the mean/std its weights were trained with (ImageNet mean/std; 0.5/0.5 for the ViT augreg weights;
ImageNet mean/std for the custom CNN). Two outputs:
  pooled(x)  -> (N, C)        the backbone's own global pooling (CNN: avg pool; ViT: CLS token after the final norm)
  tokens(x)  -> (N, h*w, C)   final feature map flattened row-major (ViT: patch tokens without CLS)
Fine-tuning policy: "frozen" | "last_stage" | "full". Frozen parameters get requires_grad=False and the modules that
hold them stay in eval mode (BatchNorm running statistics are never updated)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# module-name prefixes that form the "last stage" of each backbone family
LAST_STAGE = {
    "efficientnet": ["blocks.6", "conv_head", "bn2"],
    "resnet": ["layer4"],
    "convnext": ["stages.3"],
    "densenet": ["features.denseblock4", "features.norm5"],
    "vit": ["blocks.11", "norm", "fc_norm"],
}
# extra modules used only by pooled() (sequence mode): ConvNeXt applies head.norm after global pooling
LAST_STAGE_POOLED_EXTRA = {"convnext": ["head.norm"]}


class SmallCNN(nn.Module):
    """Custom CNN trained from scratch: 4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2], channels 32-64-128-256.
    112 x 112 input -> 7 x 7 x 256 map; 224 x 224 input -> 14 x 14 x 256 map."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.num_features = c_in
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        return self.features(x)


class SpatialEncoder(nn.Module):
    def __init__(self, enc_cfg, input_size, mode="sequence"):
        super().__init__()
        self.kind, self.mode = enc_cfg["type"], mode
        self.input_size = input_size
        if self.kind == "timm":
            self.backbone = timm.create_model(enc_cfg["timm_name"], pretrained=True, num_classes=0)
            pc = self.backbone.pretrained_cfg
            mean, std = pc["mean"], pc["std"]
            self.family = enc_cfg["family"]
            self.is_vit = self.family == "vit"
            self.timm_name = enc_cfg["timm_name"]
        elif self.kind == "custom_cnn":
            self.backbone = SmallCNN(tuple(enc_cfg["channels"]))
            mean, std = IMAGENET_MEAN, IMAGENET_STD
            self.family, self.is_vit, self.timm_name = "custom_cnn", False, None
        else:
            raise ValueError(self.kind)
        self.out_dim = self.backbone.num_features
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.policy = "full"
        self._trainable_prefixes = []

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, x):
        x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False,
                              antialias=True)
        return (x - self.mean) / self.std

    # ---------------------------------------------------------------- outputs
    def _map(self, x):
        return self.backbone.forward_features(x)

    def pooled(self, x):
        f = self._map(x)
        if self.kind == "custom_cnn":
            return f.mean(dim=(2, 3))
        return self.backbone.forward_head(f, pre_logits=True)

    def tokens(self, x):
        f = self._map(x)
        if self.is_vit:
            return f[:, self.backbone.num_prefix_tokens:, :]
        return f.flatten(2).transpose(1, 2)

    def n_tokens(self):
        if self.is_vit:
            g = self.backbone.patch_embed.grid_size
            return g[0] * g[1]
        with torch.no_grad():
            dev = self.mean.device
            return self.tokens(torch.zeros(1, 3, self.input_size, self.input_size, device=dev)).shape[1]

    # ---------------------------------------------------------------- fine-tuning policy
    def set_policy(self, policy):
        if self.kind == "custom_cnn":
            policy = "full"  # custom CNNs are always fully trainable
        if policy not in {"frozen", "last_stage", "full"}:
            raise ValueError(policy)
        self.policy = policy
        last = LAST_STAGE.get(self.family, []) + (LAST_STAGE_POOLED_EXTRA.get(self.family, []) if self.mode == "sequence"
                                                   else [])
        self._trainable_prefixes = {"frozen": [], "full": [""], "last_stage": last}[policy]
        for name, p in self.backbone.named_parameters():
            p.requires_grad = self._is_trainable(name)
        self.train(self.training)
        return self

    def _is_trainable(self, name):
        return any(pref == "" or name == pref or name.startswith(pref + ".") for pref in self._trainable_prefixes)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.policy != "full":
            for name, m in self.backbone.named_modules():
                if not self._is_trainable(name):
                    m.training = False  # non-recursive: trainable sub-modules keep training mode
        return self

    def trainable_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]
