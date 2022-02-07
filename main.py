from Model import DQN
from Environment import CarlaEnv

if __name__ == '__main__':
    dqn = DQN()  # DQN 模型
    carla_env = CarlaEnv()  # 初始化 Carla 模拟环境

    num_episodes = 100
    for i_episode in range(num_episodes):
        old_observation = carla_env.reset()

        while True:
            action = dqn.choose_action(old_observation)
            observation, reward, done, info = carla_env.step(action)

            dqn.push_memory(old_observation, action, reward, observation)
            old_observation = observation

            if done:    # 本集结束，重置环境
                observation = carla_env.reset()
                break

        if (dqn.position % 10) == 0:
            dqn.learn()
    carla_env.close()
