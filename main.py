# import the necessary packages
from moviepy.editor import VideoFileClip
from collections import deque
import numpy as np
import cv2
import imutils
import time


# Function that accepts video path and start time
# and shows slow motion video to user and also
# stores the file clipped for slow motion video
# as well as slow motion video
def video_processor(video_path, start_time):
    cap = VideoFileClip(video_path)
    starting_point = start_time  # start at given time
    end_point = starting_point + 0.1
    subclip = cap.subclip(starting_point, end_point)
    subclip.write_videofile("output/check1.mp4")
    cap = cv2.VideoCapture("output/check1.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('output/output1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret is True:
            # Write the frame into the file 'output.avi'
            out.write(frame)
            # Display the resulting frame
            cv2.imshow('Slow Motion Video Analysis', frame)
            cv2.waitKey(1000)
        # Break the loop
        else:
            break
        # When everything done, release the video capture and video write objects
    cap.release()
    out.release()


# define the lower and upper boundaries of the "white"
# ball in the HSV color space, then initialize the
# list of tracked points
whiteLower = (0, 0, 220)
whiteUpper = (255, 35, 255)
pts = deque(maxlen=64)
counter = 0
(dX, dY) = (0, 0)
direction = ""
loopEnd = False
path_of_video = "video/2.avi"

# Video File Path
vs = cv2.VideoCapture(path_of_video)
# allow the camera or video file to warm up
time.sleep(1.0)
# Calculating fps to know time of collision of ball and club
fps = vs.get(cv2.CAP_PROP_FPS)
# keep looping until ball gets hit by clubs
while loopEnd is False:
    # grab the current frame
    frame = vs.read()
    # handle the frame from VideoStream
    frame = frame[1]

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
            # check to see if enough points have been accumulated in
            # the buffer
        if counter >= 10 and i == 1 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ("", "")
            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 6:
                dirX = "East" if np.sign(dX) == 1 else "West"
                print("Stopped here - East/West")
                loopEnd = True
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 3:
                dirY = "North" if np.sign(dY) == 1 else "South"
                print("Stopped here - North/South")
                loopEnd = True

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the movement deltas and the direction of movement on
    # the frame
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    # show the frame to our screen and increment the frame counter
    cv2.imshow("Golf Analysis", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
# close all windows
startTime = counter / fps
print("Time of collision: ", startTime)
video_processor(path_of_video, startTime)
# Closes all the frames
cv2.destroyAllWindows()
