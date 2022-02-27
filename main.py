import os.path

from datetime import datetime

from Model import DQN
from Environment import CarlaEnv
from utils import TensorBoard

if __name__ == '__main__':
	runs_name = ""
	log_dir = os.path.join("./logs/train", runs_name if runs_name else datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	tensorboard = TensorBoard(log_dir)
	
	dqn_model = DQN(tensorboard=tensorboard)  # DQN 模型
	carla_env = CarlaEnv(tensorboard=tensorboard)  # 初始化 Carla 模拟环境
	
	episodes = 1000  # 模拟训练轮数
	for episode in range(episodes):
		old_observation = carla_env.reset()
		
		info_dict = {"reward": [], "distance": []}  # 用于记录当前轮训练过程中的一些信息
		while True:
			action = dqn_model.choose_action(old_observation)
			observation, reward, done, info = carla_env.step(action)
			
			dqn_model.push_memory(old_observation, action, reward, observation)
			old_observation = observation
			
			info_dict["reward"].append(reward)
			info_dict["distance"].append(info.get("target_distance"))
			
			if done:  # 本集结束，重置环境
				observation = carla_env.reset()
				break
		
		tensorboard.process_info(episode, info_dict)  # 处理当前轮训练信息，记录日志
		if (dqn_model.position % 10) == 0:
			dqn_model.learn(epoch=episode)
	carla_env.close()
