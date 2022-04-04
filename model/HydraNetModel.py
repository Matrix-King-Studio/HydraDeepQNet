from torch import nn


class Bottleneck(nn.Module):
	expansion = 1
	
	def __init__(self, in_channels, out_channels, stride=1, downsample=None, group_width=1, dilation=1, norm_layer=None):
		super().__init__()
		if norm_layer is None:  # 如果未指定标准化层，则初始化为 BatchNorm2d
			norm_layer = nn.BatchNorm2d
		width = out_channels * self.expansion
		self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
		self.bn1 = norm_layer(width)
		self.conv2 = nn.Conv2d(width, )


class RegNet(nn.Module):
	def __init__(self):
		super().__init__()


class HydraNet(nn.Module):
	def __init__(self):
		super().__init__()
