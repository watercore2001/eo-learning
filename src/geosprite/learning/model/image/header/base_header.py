from torch import nn


class BaseHeader(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
