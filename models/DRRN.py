import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

class DRRN(nn.Module):
	def __init__(self, num_chanels, scale_factor):
		super(DRRN, self).__init__()

		self.scale_factor = scale_factor

		self.input = nn.Conv2d(in_channels=num_chanels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=128, out_channels=num_chanels, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(25):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out

if __name__ == '__main__':
	x = torch.rand((1, 1, 32, 32))

	drrn = DRRN(num_chanels=1, scale_factor=4)

	out = drrn(x)
	print(out.shape)
