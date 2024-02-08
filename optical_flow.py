import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

csv_file_path = 'training data phase 1.csv'
dir_path = 'train_images/'

df = pd.read_csv(csv_file_path)
image_name_df = df['image_names']
frame_time_df = df['frame_time']

for index, file_name in enumerate(image_name_df):
    frame_a  = cv2.imread(dir_path + file_name, cv2.IMREAD_GRAYSCALE)
    frame_b  = cv2.imread(dir_path + image_name_df[index+1], cv2.IMREAD_GRAYSCALE)
    if frame_time_df[index+1] < frame_time_df[index]:
        continue

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(frame_a, frame_b, None, 0.5, 5, 15, 3, 5, 1.1, 0)

    # Create empty image to draw vectors on
    h, w = frame_a.shape[:2]
    vis = cv2.cvtColor(frame_a, cv2.COLOR_GRAY2BGR)  # Create a color version of grayscale for visualization

    # Calculate the step size dynamically
    max_dim = max(h, w)
    step = max_dim // 50  # Adjust the divisor to control the density of vectors
    dfx, dfy, dffx, dffy = [], [], [], []
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            if fx >=0.1 or fy >=0.1:
                cv2.arrowedLine(vis, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.5)
            dfx.append(x)
            dfy.append(y)
            dffx.append(fx)
            dffy.append(fy)

    data = {
        'x': dfx,
        'y': dfy,
        'fx': dffx,
        'fy': dffy
    }

    # Saving the image using OpenCV
    filename = f'PIV_results/{file_name[:-4]}.png'
    cv2.imwrite(filename, vis)

    df = pd.DataFrame(data)
    csv_file_path = f'PIV_vectors/' + file_name[:-4] + '.csv'
    df.to_csv(csv_file_path, index=False)


