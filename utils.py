from subprocess import Popen, PIPE, STDOUT
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
	def __init__(self, log_dir):
		self.log_dir = log_dir
		self.writer = SummaryWriter(self.log_dir)
	
	def process_info(self, episode, info_dict):
		"""
		处理训练过程信息，记录日志
		:param episode:
		:param info_dict:
		:return:
		"""
		steps = len(info_dict.get("reward"))
		for step in range(steps):
			scalar_dict = {"reward": info_dict.get("reward")[step], "distance": info_dict.get("distance")[step]}
			self.writer.add_scalars("Train Process", scalar_dict, steps)


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
