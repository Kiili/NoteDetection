import torch
from torch import nn, Tensor
from pytorchsummary import summary


class PianoModelBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pad=True, ksize=(3, 3), stride=(1, 1), drop=0.0):
        super().__init__()
        padding = (ksize[0] // 2, ksize[1] // 2) if pad else (0, 0)

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_dim, out_dim, kernel_size=ksize, stride=stride, bias=False, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=drop),

            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, bias=False, padding=padding),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: Tensor):
        x = self.relu(self.main(x) + self.downsample(x))
        return x


class PianoModelSmallSelf(nn.Module):
    def __init__(self, input_size=(480, 640), k=5) -> None:
        super().__init__()
        self.k = k
        self.dim_after_preprocessing = 16
        downscale_dim_sizes = [32, 32, 64, 128, 128, 256]

        self.agg_input_frames = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(16, downscale_dim_sizes[0], kernel_size=(self.k - 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(downscale_dim_sizes[0]),
            nn.LeakyReLU(inplace=True),
        )

        self.pred_frame_conv = PianoModelBlock(in_dim=1, out_dim=downscale_dim_sizes[0], ksize=(3, 3), stride=1)

        self.blocks = nn.ModuleList([
            PianoModelBlock(in_dim=downscale_dim_sizes[0], out_dim=downscale_dim_sizes[1], ksize=(3, 3), stride=2,
                            drop=0.2),
            PianoModelBlock(in_dim=downscale_dim_sizes[1], out_dim=downscale_dim_sizes[2], ksize=(3, 3), stride=2,
                            drop=0.2),
            PianoModelBlock(in_dim=downscale_dim_sizes[2], out_dim=downscale_dim_sizes[3], ksize=(3, 3), stride=(2, 1),
                            drop=0.2),
            PianoModelBlock(in_dim=downscale_dim_sizes[3], out_dim=downscale_dim_sizes[4], ksize=(3, 3), stride=(2, 1),
                            drop=0.2),
            PianoModelBlock(in_dim=downscale_dim_sizes[4], out_dim=downscale_dim_sizes[5], ksize=(3, 3), stride=(2, 1),
                            drop=0.0),
        ])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 88))
        self.fc1 = nn.Linear(input_size[0] // 32, 1)
        final_conv_dim = 256
        self.final_conv = nn.Conv1d(downscale_dim_sizes[-1], final_conv_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(input_size[1] // 4 * final_conv_dim, 88)

    def forward(self, x: Tensor):
        pred_frame = self.pred_frame_conv(x[:, :, -1])
        x = self.agg_input_frames(x)

        x = x[:, :, 0] + pred_frame

        for block in self.blocks:
            x = block(x)

        x = torch.moveaxis(x, 2, 3)
        x = self.fc1(x)
        x = x[:, :, :, 0]
        # x = self.avgpool(x)
        x = self.final_conv(x)
        x = self.fc(x.flatten(1))

        return x


if __name__ == '__main__':
    k = 5
    inpt = torch.rand(size=[1, 1, k, 480, 640])
    model = PianoModelSmallSelf(k=k)
    print(summary(input_size=inpt.squeeze(0).shape, model=model))
    print("in shape:", inpt.shape)
    print("out shape:", model(inpt).shape)
