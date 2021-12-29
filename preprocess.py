import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset

class IrisPreprocessor():
    def __init__(self, labels_dir, image_dir):
        self.labels_dir = labels_dir
        self.labels = self.labels_preprocessing()
        self.image_dir = image_dir

    def labels_preprocessing(self):
        labels_list = os.listdir(self.labels_dir)
        labels_df = pd.DataFrame(
            columns=[
                "image",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
                "x4",
                "y4",
                "x5",
                "y5",
                "x6",
                "y6",
            ],
        )

        for label in labels_list:
            data = pd.read_csv(
                self.labels_dir + label,
                sep="\t",
                header=None,
                names=[
                    "image",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "x3",
                    "y3",
                    "x4",
                    "y4",
                    "x5",
                    "y5",
                    "x6",
                    "y6",
                ],
            )
            labels_df = labels_df.append(data, ignore_index=True)

        return labels_df

    def preprocess_eye(self, image, left_corner, iris, right_corner):
        width = right_corner[0] - left_corner[0]
        eye = image[
            int((left_corner[1] + right_corner[1]) // 2 - width // 2) : int(
                (left_corner[1] + right_corner[1]) // 2 + width // 2
            ),
            int(left_corner[0]) : int(right_corner[0]),
        ]

        iris = (
            iris[0] - left_corner[0],
            iris[1] - (left_corner[1] + right_corner[1]) // 2 - width // 2,
        )
        iris = (int(iris[0] * 48 / width), int(iris[1] * 48 / width))
        iris_label = np.zeros((48, 48))
        iris_label[iris[0], iris[1]] = 1
        return eye, iris_label

    def crop_eyes(self, image, x_coords, y_coords):
        left_eye_left_corner = (x_coords[2], y_coords[2])
        left_eye_iris = (x_coords[1], y_coords[1])
        left_eye_right_corner = (x_coords[0], y_coords[0])
        left_eye, left_iris = self.preprocess_eye(
            image, left_eye_left_corner, left_eye_iris, left_eye_right_corner
        )

        right_eye_left_corner = (x_coords[5], y_coords[5])
        right_eye_iris = (x_coords[4], y_coords[4])
        right_eye_right_corner = (x_coords[3], y_coords[3])
        right_eye, right_iris = self.preprocess_eye(
            image, right_eye_left_corner, right_eye_iris, right_eye_right_corner
        )

        left_eye = cv2.resize(left_eye, (48, 48))
        right_eye = cv2.resize(right_eye, (48, 48))
        return left_eye, right_eye, left_iris, right_iris

    def preprocess(self, visualize_flag=True):
        image_list = os.listdir(self.image_dir)
        X = []
        Y = []

        for name in image_list:
            if name.endswith(".png"):

                image = cv2.imread(self.image_dir + name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                coords = self.labels[self.labels["image"] == name]
                x_coords = coords[["x1", "x2", "x3", "x4", "x5", "x6"]]
                y_coords = coords[["y1", "y2", "y3", "y4", "y5", "y6"]]
                if visualize_flag:
                    visualize_flag = False
                    fig, axs = plt.subplots(1, figsize=(15, 15))
                    axs.imshow(image, cmap="gray")
                    axs.axis("off")
                    axs.scatter(
                        x_coords[["x1", "x3", "x4", "x6"]],
                        y_coords[["y1", "y3", "y4", "y6"]],
                        c="blue",
                        label="corners",
                    )
                    axs.scatter(
                        x_coords[["x2", "x5"]],
                        y_coords[["y2", "y5"]],
                        c="red",
                        label="irises",
                    )

                image = image / 255

                left_eye, right_eye, left_iris, right_iris = self.crop_eyes(
                    image, x_coords.values.tolist()[0], y_coords.values.tolist()[0]
                )
                X.append(np.expand_dims(left_eye, axis=0))
                X.append(np.expand_dims(right_eye, axis=0))

                Y.append(np.expand_dims(left_iris, axis=0))
                Y.append(np.expand_dims(right_iris, axis=0))

        X = np.array(X)
        Y = np.array(Y)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)

        eye_dataset = TensorDataset(X, Y)
        train_size = len(eye_dataset) * 4 // 5
        test_size = len(eye_dataset) - train_size
        train, test = torch.utils.data.random_split(
        eye_dataset, [train_size, test_size], torch.Generator().manual_seed(42))
        return train, test
