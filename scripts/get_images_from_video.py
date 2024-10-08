# Importing all necessary libraries
import cv2
import os

data_dir = './data/sponge_shortside'
video_path = data_dir + '/video.mp4'
images_dir = data_dir + '/images'
# Read the video from specified path
cam = cv2.VideoCapture(video_path)

try:
    # creating a folder named data
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

# frame
currentframe = 0

while(True):
    # reading from frame
    ret,frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = images_dir + '/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
