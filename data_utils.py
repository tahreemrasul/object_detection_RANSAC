# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

def load_cloud_image(example):
    path_prefix = os.path.join(os.path.dirname(__file__), "data1")
    img_name = "example" + str(example) + "kinect.mat"
    img_path = os.path.join(path_prefix, img_name)

    img = sio.loadmat(img_path)
    return img["cloud" + str(example)]

def show_image(img, name=""):
    fig = plt.figure()

    if len(img.shape) == 3:
        # subsampling for visualization
        subsampled_image = img[::10, ::10, :]
        reshaped_image = subsampled_image.reshape((-1, 3))

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reshaped_image[:, 0], reshaped_image[:, 1], zs=reshaped_image[:, 2], marker='o')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.title.set_text(name)
    else:
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.title.set_text(name)

    plt.show()

def show_images(img1, img2, img3, name=""):
    if len(img1.shape) == 2 and len(img2.shape) == 2 and len(img3.shape) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(img1)
        ax = fig.add_subplot(132)
        ax.imshow(img2)
        ax.title.set_text(name)
        ax = fig.add_subplot(133)
        ax.imshow(img3)
        plt.show()

