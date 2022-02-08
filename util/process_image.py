import cv2
import numpy as np


def concat_4_image(image_list):
    """
    将前后左右四个摄像头拍摄的图片拼接起来
    :param image_list: [前、后、左、右]
    :return:
    """
    front_back_concat = np.concatenate((image_list[0], image_list[1]), axis=0)  # 前后摄像头图像拼接
    left_right_concat = np.concatenate((image_list[2], image_list[3]), axis=0)  # 左右摄像头图像拼接
    full_concat = np.concatenate((front_back_concat, left_right_concat), axis=1)    # 前后左右拼接在一起
    return full_concat / 255.0


if __name__ == '__main__':
    front_image = cv2.imread("../logs/2022-02-07/images/front.png")
    back_image = cv2.imread("../logs/2022-02-07/images/back.png")
    left_image = cv2.imread("../logs/2022-02-07/images/left.png")
    right_image = cv2.imread("../logs/2022-02-07/images/right.png")
    cv2.imshow("image", concat_4_image([front_image, back_image, left_image, right_image]))
    cv2.waitKey(0)
