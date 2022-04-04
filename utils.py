from subprocess import Popen, PIPE, STDOUT
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
	def __init__(self, log_dir):
		self.log_dir = log_dir
		self.writer = SummaryWriter(self.log_dir)


def exe_command(command):
	"""
	执行 shell 命令并实时打印输出
	:param command: shell 命令
	:return: process, exitcode
	"""
	print(command)
	process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
	with process.stdout:
		for line in iter(process.stdout.readline, b''):
			print(line.decode().strip())
	exitcode = process.wait()
	return process, exitcode
