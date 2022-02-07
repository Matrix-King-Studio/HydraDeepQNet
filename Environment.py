import math
import random
import time

import carla
import cv2
import gym
import numpy as np

from spawn_npc import spawn_npc

# ---------- 定义全局常量 ----------
SHOW_PREVIEW = False  # 是否显示预览
IMG_WIDTH = 80
IMG_HEIGHT = 60


class CarlaEnv(gym.Env):
    """
    汽车交互环境
    """
    SHOW_CAM = SHOW_PREVIEW  # 是否现实摄像头画面
    STEER_AMT = 1.0
    seconds_per_episode = 100

    def __init__(self):
        self.image_width = IMG_WIDTH
        self.image_height = IMG_HEIGHT

        self.actor_list = list()
        self.collision_hist = list()
        self.front_camera = None
        self.collision_list = list()  # 碰撞传感器历史

        self.client = carla.Client("localhost", 7000)
        self.client.set_timeout(5)
        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = self.blueprint_library.find("vehicle.tesla.model3")  # 随机召唤一辆汽车
        self.camera_sensor = None  # 前置摄像头传感器
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
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False))
        elif action == 1:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.5, steer=-1, brake=0.0, hand_brake=False, reverse=False))
        elif action == 2:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.5, steer=1, brake=0.0, hand_brake=False, reverse=False))
        elif action == 4:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, hand_brake=False, reverse=False))
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True))

        reward = 0  # 此次 step 的奖励

        if action == 4:  # 尽量避免执行倒车操作
            reward -= 0.1

        last_dis = self.target_dis
        self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())
        if last_dis < self.target_dis:  # 如果 car 距离目的地越来越远
            reward -= 0.5
        if abs(self.target_dis - last_dis) < 10:
            reward -= 0.5

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

        # 如果发生碰撞，本次训练终止，执行惩罚
        if len(self.collision_hist) != 0:
            print(f"检测到碰撞发生！")
            done = True
            reward -= 1
        elif self.target_dis < 3:  # 到达终点
            print(f"到达终点！")
            done = True
            reward += 3
        else:
            done = False
            reward += (kmh - 50) / 100

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True

        # Weights rewards (not for terminal state)
        if not done:
            reward *= (time.time() - self.episode_start) / self.seconds_per_episode

        if self.episode_start + self.seconds_per_episode < time.time():
            print(f"本轮训练时间到！")
            done = True

        return self.front_camera, reward, done, None

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

        # ---------- RGB 摄像头传感器 ----------
        rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam.set_attribute("image_size_x", f"{self.image_width}")
        rgb_cam.set_attribute("image_size_y", f"{self.image_height}")
        rgb_cam.set_attribute("fov", "110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(rgb_cam, transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self._process_img)
        self.actor_list.append(self.camera_sensor)
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
        self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera

    def _process_img(self, image):
        """
        摄像头传感器回调处理
        :param image:
        :return:
        """
        # 获取图片并去掉最后一个 alpha 通道
        image = np.array(image.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        image = image[:, :, :3]

        if self.SHOW_CAM:  # 展示预览图片
            cv2.imshow("image", image)
            cv2.waitKey(1)

        self.front_camera = image
        return image / 255.0

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
