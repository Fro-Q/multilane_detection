"""

"""

import time
import os
import json
import tempfile
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import urllib
import PIL
from PIL import Image
from io import BytesIO
import base64

from moviepy.editor import VideoFileClip


app = Flask(__name__)
CORS(app)


@app.route("/api/upload/", methods=["POST"])
def upload():
    data = request.get_json()["fileURL"]
    data_type = request.get_json()["fileType"]
    if data_type == "image":
        url = urllib.request.urlopen(data)
        img = Image.open(url)
        input_img = np.array(img)
        result_img = process_image(input_img)
        result_img = Image.fromarray(result_img)
        img_byte_array = BytesIO()
        result_img.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()
        result = "data:image/png;base64," + str(
            base64.b64encode(img_byte_array), "utf-8"
        )
    else:
        result_video = process_video(data)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            result_video.write_videofile(temp_video.name, codec="libx264")
            temp_video.seek(0)
            video_byte_array = temp_video.read()

        result = "data:video/mp4;base64," + str(
            base64.b64encode(video_byte_array), "utf-8"
        )
        # video_byte_array = BytesIO()
        # result_video.write_videofile(video_byte_array, codec="mpeg4")
        # video_byte_array = video_byte_array.getvalue()
        # result = "data:video/mp4;base64," + str(
        #     base64.b64encode(video_byte_array), "utf-8"
        # )

    return {
        "status": "success",
        "data": {"result": result},
    }


def process_image(input_img):
    gray_img = grayscale(input_img)
    blur_img = gaussian_blur(gray_img, 5)
    height, width = blur_img.shape

    edges = canny(blur_img, 50, 150)

    # 显示图片

    # 以屏幕半高取掩膜
    all_bound_lt = (0, height)
    all_bound_lb = (0, height * 0.6)
    all_bound_rb = (width, height * 0.6)
    all_bound_rt = (width, height)
    all_lines_mask = np.array(
        [[all_bound_lt, all_bound_lb, all_bound_rb, all_bound_rt]], dtype=np.int32
    )
    all_lines_img = canny(region_of_interest(edges, all_lines_mask), 50, 150)

    # 显示图片

    # 当前车道线的梯形掩膜
    cur_bound_lt = (width * 0.1, height)
    cur_bound_lb = (width * 0.4, height * 0.65)
    cur_bound_rb = (width * 0.6, height * 0.65)
    cur_bound_rt = (width * 0.9, height)
    cur_lines_mask = np.array(
        [[cur_bound_lt, cur_bound_lb, cur_bound_rb, cur_bound_rt]], dtype=np.int32
    )
    cur_lines_img = canny(region_of_interest(edges, cur_lines_mask), 50, 150)

    # 获取当前车道线

    cur_lines = get_cur_lines(
        hough_lines(cur_lines_img, 1, np.pi / 180, 15, 50, 20), width, height
    )

    if cur_lines is None:
        return input_img
    for line in cur_lines:
        if line is None:
            return input_img

    # all_lines = get_all_lines(
    #     hough_lines(all_lines_img, 1, np.pi / 180, 15, 100, 20),
    #     width,
    #     height,
    #     cur_lines[0],
    #     cur_lines[1],
    # )

    # 透视变换
    lb = [cur_lines[0][0], cur_lines[0][1]]
    lt = [cur_lines[0][2], cur_lines[0][3]]
    rt = [cur_lines[1][2], cur_lines[1][3]]
    rb = [cur_lines[1][0], cur_lines[1][1]]
    src = np.float32([lb, lt, rt, rb])
    dst = np.float32(
        [
            [cur_lines[0][2], height],
            [cur_lines[0][2], 0],
            [cur_lines[1][2], 0],
            [cur_lines[1][2], height],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(input_img, M, (width, height), flags=cv2.INTER_LINEAR)

    per_grayscale = grayscale(warped)
    per_edges = canny(per_grayscale, 50, 150)
    per_all_lines = hough_lines(per_edges, 1, np.pi / 180, 15, 50, 20)
    all_left_lines, all_right_lines, left_line, right_line = get_all_lines_from_per(
        per_all_lines,
        width,
        height,
        cur_lines[0][2],
        cur_lines[1][2],
    )

    per_all_lines_img = np.zeros((height, width, 3), dtype=np.uint8)
    # for line in all_left_lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(per_all_lines_img, (x1, y1), (x2, y2), [255, 0, 0], 2)

    # for line in all_right_lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(per_all_lines_img, (x1, y1), (x2, y2), [0, 0, 255], 2)

    # 绘制左右车道线
    if not np.isnan(left_line[0]) and not np.isnan(left_line[2]):
        cv2.line(
            per_all_lines_img,
            (int(left_line[0]), int(left_line[1])),
            (int(left_line[2]), int(left_line[3])),
            [0, 0, 255],
            10,
        )
    if not np.isnan(right_line[0]) and not np.isnan(right_line[2]):
        cv2.line(
            per_all_lines_img,
            (int(right_line[0]), int(right_line[1])),
            (int(right_line[2]), int(right_line[3])),
            [255, 0, 0],
            10,
        )

    # 反透视变换

    reM = cv2.getPerspectiveTransform(dst, src)
    per_all_lines_img = cv2.warpPerspective(
        per_all_lines_img, reM, (width, height), flags=cv2.INTER_LINEAR
    )

    # 绘制当前车道线

    cur_lines_img = np.zeros((height, width, 3), dtype=np.uint8)
    for line in cur_lines:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(
                cur_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 10
            )

    # 绘制所有车道线

    # all_lines_img = np.zeros((height, width, 3), dtype=np.uint8)

    # for line in all_lines[0]:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(all_lines_img, (x1, y1), (x2, y2), [255, 0, 0], 2)

    # for line in all_lines[1]:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(all_lines_img, (x1, y1), (x2, y2), [0, 0, 255], 2)

    # 绘制当前车道线和所有车道线

    lines_img = cv2.addWeighted(cur_lines_img, 0.8, per_all_lines_img, 1, 0)
    result_img = cv2.addWeighted(input_img, 0.8, lines_img, 1, 0)

    # result_img = cv2.addWeighted(input_image, 0.8, cur_lines_img, 1, 0)
    # result_img = cv2.addWeighted(result_img, 0.8, all_lines_img, 1, 0)

    # 输出结果

    return result_img


def process_video(data):
    url = urllib.request.urlopen(data)
    with open("video.mp4", "wb") as f:
        f.write(url.read())
    clip = VideoFileClip("video.mp4")
    result_video = clip.fl_image(process_image)
    return result_video


def grayscale(img):
    """
    灰度处理
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """
    高斯滤波
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """
    Canny 边缘检测
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    创建一个掩膜
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Hough 变换
    """
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    return lines


def get_cur_lines(lines, width, height):
    """
    获取当前车道线
    """
    cur_left_lines = []
    cur_right_lines = []
    left_k = []
    left_b = []
    right_k = []
    right_b = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            if abs(k) > 2 or abs(k) < 0.3:
                continue
            if k < 0:
                cur_left_lines.append(line)
                left_k.append(k)
                left_b.append(b)
            else:
                cur_right_lines.append(line)
                right_k.append(k)
                right_b.append(b)

    # 去除异常值

    # left_k = np.array(left_k)
    # left_b = np.array(left_b)
    # right_k = np.array(right_k)
    # right_b = np.array(right_b)
    # left_k = left_k[np.abs(left_k - np.mean(left_k)) < 2 * np.std(left_k)]
    # left_b = left_b[np.abs(left_b - np.mean(left_b)) < 2 * np.std(left_b)]
    # right_k = right_k[np.abs(right_k - np.mean(right_k)) < 2 * np.std(right_k)]
    # right_b = right_b[np.abs(right_b - np.mean(right_b)) < 2 * np.std(right_b)]
    # 去除斜率异常值

    # 计算左右车道线的斜率的均值
    left_k_mean = np.mean(left_k)
    right_k_mean = np.mean(right_k)
    # 计算左右车道线的斜率的标准差
    left_k_std = np.std(left_k)
    right_k_std = np.std(right_k)
    # 如果左右车道线的斜率的均值和标准差都在一定范围内，则认为提取到了左右车道线
    if left_k_std < 0.5 and right_k_std < 0.5:
        cur_left_line_k = np.mean(left_k)
        cur_right_line_k = np.mean(right_k)
        cur_left_line_b = np.mean(left_b)
        cur_right_line_b = np.mean(right_b)

        cur_left_line = (
            (height - cur_left_line_b) / cur_left_line_k,
            height,
            (height * 0.7 - cur_left_line_b) / cur_left_line_k,
            height * 0.7,
        )
        cur_right_line = (
            (height - cur_right_line_b) / cur_right_line_k,
            height,
            (height * 0.7 - cur_right_line_b) / cur_right_line_k,
            height * 0.7,
        )
        return cur_left_line, cur_right_line
    else:
        return None, None


def get_all_lines(lines, width, height, cur_left_line, cur_right_line):
    """
    获取所有车道线
    """
    # 计算当前左右两条车道线的交点
    if cur_left_line is not None and cur_right_line is not None:
        cur_left_x1, cur_left_y1, cur_left_x2, cur_left_y2 = cur_left_line
        cur_right_x1, cur_right_y1, cur_right_x2, cur_right_y2 = cur_right_line
        cur_left_k = (cur_left_y2 - cur_left_y1) / (cur_left_x2 - cur_left_x1)
        cur_left_b = cur_left_y1 - cur_left_k * cur_left_x1
        cur_right_k = (cur_right_y2 - cur_right_y1) / (cur_right_x2 - cur_right_x1)
        cur_right_b = cur_right_y1 - cur_right_k * cur_right_x1
        cross_x = (cur_right_b - cur_left_b) / (cur_left_k - cur_right_k)
        cross_y = cur_left_k * cross_x + cur_left_b
    else:
        cross_x = width / 2
        cross_y = height / 2

    # 以当前车道线为中心，获取所有车道线
    # 左侧的所有车道线的斜率应小于当前车道线的斜率，截距应大于当前车道线的截距
    # 右侧的所有车道线的斜率应大于当前车道线的斜率，截距应小于当前车道线的截距
    all_left_lines = []
    all_right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1

            if abs(k) > 2 or abs(k) < 0.2:
                continue

            # 左侧的车道线的斜率应小于当前车道线的斜率，截距应大于当前车道线的截距，且在当前车道线的左侧（上方）
            # 右侧的车道线的斜率应大于当前车道线的斜率，截距应小于当前车道线的截距，且在当前车道线的右侧（上方）
            # 车道线的长度应大于一定值
            if k < 0:
                if cur_left_line is not None:
                    cur_left_x1, cur_left_y1, cur_left_x2, cur_left_y2 = cur_left_line
                    cur_k = (cur_left_y2 - cur_left_y1) / (cur_left_x2 - cur_left_x1)
                    cur_b = cur_left_y1 - cur_k * cur_left_x1
                    if (
                        k > cur_k
                        # and b < cur_b
                        and abs((cross_x * k + b) - cross_y) < 50
                    ):
                        all_left_lines.append(line)
                else:
                    all_left_lines.append(line)

            else:
                if cur_right_line is not None:
                    cur_right_x1, cur_right_y1, cur_right_x2, cur_right_y2 = (
                        cur_right_line
                    )
                    cur_k = (cur_right_y2 - cur_right_y1) / (
                        cur_right_x2 - cur_right_x1
                    )
                    cur_b = cur_right_y1 - cur_k * cur_right_x1
                    if (
                        k < cur_k
                        # and b > cur_b
                        and abs((cross_x * k + b) - cross_y) < 50
                    ):
                        all_right_lines.append(line)
                else:
                    all_right_lines.append(line)

    return all_left_lines, all_right_lines


def get_all_lines_from_per(lines, width, height, cur_left_x, cur_right_x):
    """
    从透视变换后的图像获取所有车道线
    """
    # 以当前车道线为中心，获取所有车道线
    # 左侧车道线应在当前车道线的左侧，右侧则在右侧
    # 车道线斜率应接近正无穷
    all_left_lines = []
    all_right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 斜率是否接近正无穷
            if x1 == x2:
                if x1 < width / 2:
                    if abs(x1 - cur_left_x) < 50:
                        continue
                    all_left_lines.append(line)
                else:
                    if abs(x1 - cur_right_x) < 50:
                        continue
                    all_right_lines.append(line)
            else:
                k = (y2 - y1) / (x2 - x1)
                if abs(k) <= 5:
                    continue
                if x1 < width / 2:
                    if abs(x1 - cur_left_x) < 50:
                        continue
                    all_left_lines.append(line)
                else:
                    if abs(x1 - cur_right_x) < 50:
                        continue
                    all_right_lines.append(line)

    # 取均值，但由于截距可能为无穷大，所以只取斜率的均值
    left_x1_mean = np.mean([line[0][0] for line in all_left_lines])
    left_x2_mean = np.mean([line[0][2] for line in all_left_lines])
    left_x_mean = (left_x1_mean + left_x2_mean) / 2
    right_x1_mean = np.mean([line[0][0] for line in all_right_lines])
    right_x2_mean = np.mean([line[0][2] for line in all_right_lines])
    right_x_mean = (right_x1_mean + right_x2_mean) / 2
    left_line = (
        left_x_mean,
        height,
        left_x_mean,
        0,
    )
    right_line = (
        right_x_mean,
        height,
        right_x_mean,
        0,
    )

    return all_left_lines, all_right_lines, left_line, right_line


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
