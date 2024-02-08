import cv2
import os

import numpy as np
import pandas as pd

# Open the video file
folder_path = 'PIV_video'
# Open the csv file contains the first frame number that the motor moves
csv_file_path = 'CSCI699 video starting frame.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Get a list of file names in the folder
file_names = os.listdir(folder_path)

frame_time = []  # frame number that indicate relative time from motor starting
exca_seq_num = []  # sequence number of excavation actions
image_name = []  # extracted image names

for file_name in file_names:
    matching_row = df[df['video_names'] == file_name]
    start_frame = int(matching_row['starting frame'].iloc[0])

    video_path = folder_path + '/' + file_name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error opening video file: {file_name}")
    else:
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in the video: {total_frames}")

    # Optionally, if the video format supports it, jump directly to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[0:1000, 450:1450]  # Crop ROI

        # Process the frame as needed
        frame = cv2.flip(frame, 0)  # flip the image vertically

        output_frames.append(frame)

        frame_count += 1

    cap.release()

    # Save the processed frames
    for i, frame in enumerate(output_frames[:-2]):
        cv2.imwrite(f"train_images/{file_name[:-4]}_{i}.png", frame)
        image_name.append(f"{file_name[:-4]}_{i}.png")
        frame_time.append(i)
        exca_seq_num.append(np.floor(i/22.5))


# create csv file that contains extraced images information(image_name, frame_num, frame_time, exca_seq_num)
data = {
    'image_names': image_name,
    'frame_time': frame_time,
    'sequence_number_excavation_action': exca_seq_num
}
train_df = pd.DataFrame(data)
csv_file_path = 'training data phase 1.csv'
train_df.to_csv(csv_file_path, index=False)

# create the csv that contains all video names
# data = {
#     'video_names': file_names,
# }
# df = pd.DataFrame(data)
# csv_file_path = 'Video_names.csv'
# df.to_csv(csv_file_path, index=False)