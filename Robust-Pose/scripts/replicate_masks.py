import os

import cv2


def gen_masks(dir_path):
    # this function should fill missing masks with the prior one
    folder = sorted(os.listdir(os.path.join(dir_path, "masks")))
    for i in folder:
        mask = cv2.imread(os.path.join(os.path.join(dir_path, "masks"), i))
        num = i[:-5]
        new_num = int(num)
        diff = len(num) - len(str(new_num))
        prefix = diff * "0"
        cv2.imwrite(
            os.path.join(
                os.path.join(dir_path, "masks"), (prefix + str(new_num + 1) + "l.png")
            ),
            mask,
        )


if __name__ == "__main__":
    dir_path = "data/P3_2_4k"

    gen_masks(dir_path)
