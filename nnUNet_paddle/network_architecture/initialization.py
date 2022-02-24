
from paddle import nn

class InitWeights_He:
    def __init__(self, negative_slope=1e-2):
        self.negative_slop = negative_slope
    
    def __call__(self, module):
        if isinstance(module, nn.Conv3D) or isinstance(module, nn.Conv2D) or isinstance(module, nn.Conv2DTranspose) or isinstance(module, nn.Conv3DTranspose):
            module.weight = nn.initializer.KaimingNormal(self.negative_slop)
            if module.bias is not None:
                module.bias = nn.initializer.Constant(0)(module.bias)













