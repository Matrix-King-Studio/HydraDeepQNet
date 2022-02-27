import math
import random
import time

import cv2
import gym
import carla
import numpy as np

from spawn_npc import spawn_npc

# ---------- 定义全局常量 ----------
from util.process_image import concat_4_image

SAVE_CAMERA_IMAGE = False
IMG_WIDTH = 160
IMG_HEIGHT = 120


class CarlaEnv(gym.Env):
	"""
	汽车交互环境
	"""
	STEER_AMT = 1.0
	seconds_per_episode = 100
	save_camera_image = SAVE_CAMERA_IMAGE
	
	def __init__(self, tensorboard):
		self.image_width = IMG_WIDTH
		self.image_height = IMG_HEIGHT
		
		self.tensorboard = tensorboard
		
		self.actor_list = list()
		self.collision_hist = list()
		self.collision_list = list()  # 碰撞传感器历史
		
		self.full_camera_image = None
		# 用于存储前后左右四个摄像头捕获的图像
		self.four_camera_image = {"front": None, "back": None, "left": None, "right": None}
		
		self.client = carla.Client("localhost", 7000)
		self.client.set_timeout(5)
		self.world = self.client.get_world()
		
		self.target_distance = None
		
		self.blueprint_library = self.world.get_blueprint_library()
		self.ego_vehicle = self.blueprint_library.find("vehicle.tesla.model3")  # 随机召唤一辆汽车
		self.collision_sensor = None  # 碰撞检测传感器
		
		spawn_npc(self.client, self.world, self.blueprint_library)  # 生成随机车辆和行人 NPC
	
	def step(self, action):
		"""
		:param action: 驾驶操作：0 - 执行加速，1 - 左转，2 - 右转，3 - 直行减速，4 - 直行倒车
		:return:
		"""
		# 绑定观察者视角与 ego vehicle
		spectator = self.world.get_spectator()
		ego_vehicle_location = self.vehicle.get_transform().location
		spectator_location = ego_vehicle_location + carla.Location(x=-5, z=3)
		spectator_transform = carla.Transform(spectator_location, carla.Rotation(pitch=-20))
		spectator.set_transform(spectator_transform)
		
		if action == 0:
			control = carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False)
		elif action == 1:
			control = carla.VehicleControl(throttle=0.5, steer=-1, brake=0.0, hand_brake=False, reverse=False)
		elif action == 2:
			control = carla.VehicleControl(throttle=0.5, steer=1, brake=0.0, hand_brake=False, reverse=False)
		elif action == 4:
			control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, hand_brake=False, reverse=False)
		else:
			control = carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True)
		self.vehicle.apply_control(control)
		
		reward = 0  # 此次 step 的奖励
		
		if action == 4:  # 尽量避免执行倒车操作
			reward -= 0.5
		
		last_distance = self.target_distance
		self.target_distance = self.target_transform.location.distance(self.vehicle.get_location())
		if last_distance < self.target_distance:  # 如果 car 距离目的地越来越远
			reward -= 0.5
		if abs(self.target_distance - last_distance) < 10:
			reward -= 0.5
		
		velocity = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
		
		if len(self.collision_hist) != 0:  # 如果发生碰撞，本次训练终止，执行惩罚
			print(f"检测到碰撞发生！")
			done = True
			reward -= 1
		elif self.target_distance < 3:  # 到达终点
			print(f"到达终点！")
			done = True
			reward += 10
		else:
			done = False
			reward += (kmh - 50) / 100
		
		if self.episode_start + self.seconds_per_episode < time.time():
			done = True
		
		if not done:
			reward *= (time.time() - self.episode_start) / self.seconds_per_episode
		
		if self.episode_start + self.seconds_per_episode < time.time():
			print(f"本轮训练时间到！")
			done = True
		
		# 拼接四个摄像头捕获的图像
		self.full_camera_image = concat_4_image(list(self.four_camera_image.values()))
		if self.save_camera_image:  # 保存摄像头拍摄到的图片
			cv2.imwrite("./logs/2022-02-07/images/full_camera_image.png", self.full_camera_image)
		
		info = {"target_distance": self.target_distance}
		return self.full_camera_image, reward, done, info
	
	def render(self, mode="human"):
		pass
	
	def reset(self):
		self.destroy_agents()
		
		# 生成 ego vehicle
		spawn_start = time.time()
		while True:
			try:  # 随机选择一个诞生点并保证不会发生冲突
				ego_vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
				self.vehicle = self.world.spawn_actor(self.ego_vehicle, ego_vehicle_transform)
				break
			except Exception as e:
				print(f"Reset Exception: {e}")
				time.sleep(0.1)
			
			# 如果超过 3 秒都无法 spawn，抛出异常
			if time.time() > spawn_start + 3:
				raise Exception("Can't spawn a car!")
		
		# 将 vehicle 添加到 actor_list，环境重置时删除
		self.actor_list.append(self.vehicle)
		# ------------------------------
		
		# ---------- 添加 4 个 RGB 摄像头传感器 ----------
		rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
		rgb_cam.set_attribute("image_size_x", f"{self.image_width}")
		rgb_cam.set_attribute("image_size_y", f"{self.image_height}")
		rgb_cam.set_attribute("fov", "110")
		
		front_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		back_transform = carla.Transform(carla.Location(x=-2.5, z=0.7), carla.Rotation(yaw=180))
		left_transform = carla.Transform(carla.Location(y=-2.5, z=0.7), carla.Rotation(yaw=-90))
		right_transform = carla.Transform(carla.Location(y=2.5, z=0.7), carla.Rotation(yaw=90))
		
		front_camera_sensor = self.world.spawn_actor(rgb_cam, front_transform, attach_to=self.vehicle)
		back_camera_sensor = self.world.spawn_actor(rgb_cam, back_transform, attach_to=self.vehicle)
		left_camera_sensor = self.world.spawn_actor(rgb_cam, left_transform, attach_to=self.vehicle)
		right_camera_sensor = self.world.spawn_actor(rgb_cam, right_transform, attach_to=self.vehicle)
		
		front_camera_sensor.listen(lambda image: self._process_img(image, "front"))
		back_camera_sensor.listen(lambda image: self._process_img(image, "back"))
		left_camera_sensor.listen(lambda image: self._process_img(image, "left"))
		right_camera_sensor.listen(lambda image: self._process_img(image, "right"))
		
		self.actor_list.extend([front_camera_sensor, back_camera_sensor, left_camera_sensor, right_camera_sensor])
		# -----------------------------------
		
		self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
		
		time.sleep(3)
		
		# ---------- 碰撞检测传感器 ----------
		self.collision_hist = []
		collision_sensor = self.blueprint_library.find("sensor.other.collision")
		self.collision_sensor = self.world.spawn_actor(collision_sensor, carla.Transform(), attach_to=self.vehicle)
		self.collision_sensor.listen(self._collision_data)
		self.actor_list.append(self.collision_sensor)
		# ---------------------------------
		
		# target_transform 定义驾驶目的地坐标
		self.target_transform = random.choice(self.world.get_map().get_spawn_points())
		self.target_distance = self.target_transform.location.distance(self.vehicle.get_location())
		
		# 保证前后左右四个摄像头都已正常工作
		while True:
			for k, v in self.four_camera_image.items():
				if v is None:
					time.sleep(0.01)
					break
			else:
				break
		self.full_camera_image = concat_4_image(list(self.four_camera_image.values()))
		
		self.episode_start = time.time()
		self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
		
		return self.full_camera_image
	
	def _process_img(self, image, direction):
		"""
		摄像头传感器回调处理
		:param image:
		:param direction: 前后左右四个方向
		:return:
		"""
		# 获取图片并去掉最后一个 alpha 通道
		image = np.array(image.raw_data)
		image = image.reshape((self.image_height, self.image_width, 4))
		image = image[:, :, :3]
		
		if self.save_camera_image:  # 保存摄像头拍摄到的图片
			cv2.imwrite(f"./logs/2022-02-07/images/{direction}.png", image)
		
		self.four_camera_image[direction] = image
	
	def _collision_data(self, event):
		"""
		碰撞传感器回调处理
		:param event:
		:return:
		"""
		COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 500]]
		
		# What we collided with and what was the impulse
		collision_actor_id = event.other_actor.type_id
		collision_impulse = math.sqrt(
			event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)
		
		# Filter collisions
		for actor_id, impulse in COLLISION_FILTER:
			if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
				return
		# 增加碰撞事件数据
		self.collision_hist.append(event)
	
	def destroy_agents(self):
		"""
		Destroys all agents created from last .reset() call
		:return:
		"""
		if self.actor_list:
			print(f"正在销毁所有 actor")
		for actor in self.actor_list:
			# If it has a callback attached, remove it first
			if hasattr(actor, 'is_listening') and actor.is_listening:
				actor.stop()
		
		self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
		self.actor_list = []
