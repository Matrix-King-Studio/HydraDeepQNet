import os.path

from datetime import datetime

from Model import DQN
from utils import TensorBoard
from Environment import CarlaEnv

if __name__ == '__main__':
	runs_name = ""
	log_dir = os.path.join("./logs/train", runs_name if runs_name else datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	tensorboard = TensorBoard(log_dir)
	
	dqn_model = DQN(tensorboard=tensorboard)  # DQN 模型
	carla_env = CarlaEnv(tensorboard=tensorboard)  # 初始化 Carla 模拟环境
	
	episodes = 1000  # 模拟训练轮数
	for episode in range(episodes):
		old_observation = carla_env.reset()
		
		step = 0
		while True:
			step += 1
			action = dqn_model.choose_action(old_observation)
			observation, reward, done, info = carla_env.step(action)
			
			dqn_model.push_memory(old_observation, action, reward, observation)
			old_observation = observation

			if episode and episode % 10 == 0:   # 每 10 集记录一次训练过程中的各种参数
				tensorboard.writer.add_scalar(f"Train Process {episode} Reward", reward, step)
				tensorboard.writer.add_scalar(f"Train Process {episode} Speed", info.get("speed"), step)
				tensorboard.writer.add_scalar(f"Train Process {episode} Distance", info.get("target_distance"), step)
			
			if done:  # 本集结束，重置环境
				observation = carla_env.reset()
				break
		
		if (dqn_model.position % 5) == 0:
			dqn_model.learn(epoch=episode)
	carla_env.close()
