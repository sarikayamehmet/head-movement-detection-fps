import cv2
import numpy as np
import time

# Initialize video file
video_file = '2022-03-14 11-08-40-aca.mp4'  # Replace with the path to your MP4 video file

# Open the video file
cap = cv2.VideoCapture(video_file)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
old_frame = None
old_points = None

start_time = time.time()  # Start time of the video

while True:
    # Read the current frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop when the video ends

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If it's the first frame, initialize the previous frame and points
    if old_frame is None:
        old_frame = gray_frame
        p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        old_points = p0

    # Calculate optical flow using Lucas-Kanade algorithm
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, gray_frame, old_points, None, **lk_params)
    
    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        old_points = p0
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, gray_frame, old_points, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = old_points[st == 1]

    # Calculate the mean movement of points
    if len(good_new) > 0 and len(good_old) > 0:
        mean_movement = np.mean(good_new - good_old, axis=0)
        dx, dy = mean_movement.ravel()
        dz = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate elapsed time based on frame index and FPS
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        elapsed_time = current_frame / fps
        print(f"Time: {elapsed_time:.2f} s - Movement - X: {dx}, Y: {dy}, Z: {dz}")

    # Update previous frame and points
    old_frame = gray_frame.copy()
    old_points = good_new.reshape(-1, 1, 2)

    # Display the frame with motion vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a.astype(int), b.astype(int)), (c.astype(int), d.astype(int)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a.astype(int), b.astype(int)), 3, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
