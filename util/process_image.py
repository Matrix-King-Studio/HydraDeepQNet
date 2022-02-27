import cv2
import numpy as np


def concat_4_image(image_list):
    """
    将前后左右四个摄像头拍摄的图片拼接起来
    :param image_list: [前、后、左、右]
    :return:
    """
    front_left_concat = np.concatenate((image_list[0], image_list[2]), axis=0)  # 前、左摄像头图像拼接
    right_back_concat = np.concatenate((image_list[3], image_list[1]), axis=0)  # 右、后摄像头图像拼接
    full_concat = np.concatenate((front_left_concat, right_back_concat), axis=1)    # 前后左右拼接在一起
    return full_concat / 255.0


if __name__ == '__main__':
    front_image = cv2.imread("../logs/2022-02-07/images/front.png")
    back_image = cv2.imread("../logs/2022-02-07/images/back.png")
    left_image = cv2.imread("../logs/2022-02-07/images/left.png")
    right_image = cv2.imread("../logs/2022-02-07/images/right.png")
    cv2.imshow("image", concat_4_image([front_image, back_image, left_image, right_image]))
    cv2.waitKey(0)
