import torch.nn as nn
import torch

def get_activation(activation_type):
    """Get activation function from string name, defaults to ReLU if not found."""
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    """Create a sequence of n convolutional layers with batch norm and activation."""
    layers = []
    # First convolution layer changes channels from in_channels to out_channels
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    # Add remaining convolution layers (keeping channels constant)
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """Basic building block: Conv2d -> BatchNorm2d -> Activation"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        # 3x3 convolution with padding to maintain spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        # Batch normalization for stable training
        self.norm = nn.BatchNorm2d(out_channels)
        # Get activation function based on input string
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> Multiple Conv layers"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        # Max pooling to reduce spatial dimensions by half
        self.maxpool = nn.MaxPool2d(2)
        # Convolution layers for feature extraction
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)  # Reduce spatial size
        return self.nConvs(out)  # Apply convolutions

class UpBlock(nn.Module):
    """Upsampling block: Transposed Conv -> Concatenate skip connection -> Conv layers"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # Transposed convolution to upsample by factor of 2
        # in_channels//2 because we'll concatenate with skip connection
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        # Convolution layers after concatenation
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)  # Upsample the input
        x = torch.cat([out, skip_x], dim=1)  # Concatenate with skip connection
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        
        # Encoder path (downsampling)
        self.inc = ConvBatchNorm(n_channels, in_channels)  # Initial conv
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)  # 64 -> 128
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)  # 128 -> 256
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)  # 256 -> 512
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)  # 512 -> 512
        
        # Decoder path (upsampling)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)  # 1024 -> 256
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)    # 512 -> 128
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)      # 256 -> 64
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)      # 128 -> 64
        
        # Output layer to get desired number of classes
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        
        # Sigmoid activation for binary segmentation, None for multi-class
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        x = x.float()
        # Encoder path - store skip connections
        x1 = self.inc(x)        # Store for skip connection
        x2 = self.down1(x1)     # Store for skip connection
        x3 = self.down2(x2)     # Store for skip connection
        x4 = self.down3(x3)     # Store for skip connection
        x5 = self.down4(x4)     # Bottleneck
        
        # Decoder path - use skip connections
        x = self.up4(x5, x4)    # Upsample and concatenate with x4
        x = self.up3(x, x3)     # Upsample and concatenate with x3
        x = self.up2(x, x2)     # Upsample and concatenate with x2
        x = self.up1(x, x1)     # Upsample and concatenate with x1
        
        # Output layer with optional activation
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        return logits