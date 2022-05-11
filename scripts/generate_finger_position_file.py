"""
Tracks the finger and produces a text file with coordinates of its center
at each frame.
"""


import cv2


class ColourTracker:
    def init(self, roi, circle_box):
        cx, cy, r = circle_box
        self.bbox = (cx - r, cy - r, 2 * r, 2 * r)

        return True

    def update(self, roi):
        return True, self.bbox


def ColourTracker_create():
    return ColourTracker()


class FingerTracker:
    tracker_color = (255, 255, 255)  # Color used to draw the tracker's position
    circle_tracker_color = (
        255,
        255,
        0,
    )  # Color used to draw the finger's tracked circle

    def __init__(self, first_img, finger_data, tracker_creator_name=None):
        """

        :param first_img:
        :param finger_data:
        :param tracker_creator_name: Name of OpenCV tracker create function, if one of them works.
        """
        roi_ratio = self.roi_ratio = 2  # based on the finger radio
        # Use ROI around finger
        print(finger_data)
        cx = finger_data["x"]
        cy = finger_data["y"]
        r = finger_data["r"]
        self.circle_data = (cx, cy, r)

        x1 = cx - roi_ratio * r
        x2 = cx + roi_ratio * r
        y1 = cy - roi_ratio * r
        y2 = cy + roi_ratio * r
        roi = first_img[y1:y2, x1:x2]  # numpy shape: (120, 120, 3) white box

        # bbox = cv2.selectROI(first_img)  # Allows the user to select the region using the mouse
        bbox = (
            (roi_ratio - 1) * r,
            (roi_ratio - 1) * r,
            2 * r,
            2 * r,
        )  # in roi local coordinates

        if tracker_creator_name:  # if specific cv tracker is selected
            tracker_creator = getattr(cv2, tracker_creator_name)
            tracker = tracker_creator()
            ok = tracker.init(roi, bbox)
        else:
            tracker = ColourTracker_create()
            ok = tracker.init(roi, (roi_ratio * r, roi_ratio * r, r))

        self.tracker = tracker
        self.draw_inidicators(first_img, (x1, y1, x2, y2), bbox)

    #  img -> roi_coords -> bbox : contention hierarchy. bbox is inside roi and so on
    def draw_inidicators(self, img, roi_coords, bbox):
        """
        Draws indicators on img
        :param roi_coords: (x1, y1, x2, y2)
        :return:
        """
        x1, y1, x2, y2 = roi_coords
        roi = img[y1:y2, x1:x2]

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(roi, p1, p2, (255, 0, 0), 2, 1)  # bbox

        cx, cy, r = self.circle_data
        cv2.rectangle(img, (x1, y1), (x2, y2), FingerTracker.tracker_color, 2, 1)  # roi
        cv2.circle(img, (cx, cy), r, FingerTracker.circle_tracker_color)  # finger

    def track(self, next_frame):
        cx, cy, r = self.circle_data
        roi_ratio = self.roi_ratio

        x1 = cx - roi_ratio * r
        x2 = cx + roi_ratio * r
        y1 = cy - roi_ratio * r
        y2 = cy + roi_ratio * r
        roi = next_frame[y1:y2, x1:x2]

        successful_update, bbox = self.tracker.update(roi)

        if successful_update:
            new_cx = int(x1 + (bbox[2] + 2 * bbox[0]) / 2)
            new_cy = int(y1 + (bbox[3] + 2 * bbox[1]) / 2)
            self.circle_data = (new_cx, new_cy, r)

            # visual bug: delayed roi, the old coordinates are being passed
            self.draw_inidicators(next_frame, (x1, y1, x2, y2), bbox)
        else:
            print("Tracking Failure")
            cv2.putText(
                frame,
                "Tracking failure detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )


intial_finger_positions = {
    "sponge_centre": {"x": 617, "y": 311, "r": 30},
    "sponge_longside": {"x": 617, "y": 280, "r": 30},
    "sponge_shortside": {"x": 619, "y": 201, "r": 30},
    "plasticine_centre": {"x": 723, "y": 146, "r": 30},
    "plasticine_longside": {"x": 721, "y": 167, "r": 30},
    "plasticine_shortside": {"x": 724, "y": 31, "r": 26},
}
# press ESC to exit the visualization
if __name__ == "__main__":
    CVWAIT = 20  # reproduction speed of video
    SAVE_DATA = False  # if true, the read info will be saved into file
    dir_name = "sponge_shortside"
    input_file = f"data/{dir_name}/video.mp4"
    output_file = f"data/{dir_name}/finger_position.txt"
    cv2_tracker_name = "TrackerKCF_create"
    cv2_tracker_name = "TrackerMIL_create"
    print("Analizing file: " + input_file)

    if SAVE_DATA:
        finger_file = open(output_file, "w")
        print("Saving to ", output_file)

    cap = cv2.VideoCapture(input_file)
    any_problem, frame = cap.read()
    if not any_problem:
        raise Exception("There was a problem while reading the video")
    finger_tracker = FingerTracker(
        frame, intial_finger_positions[dir_name], cv2_tracker_name
    )
    if SAVE_DATA:
        finger_file.write(
            "{x} {y}\n".format(
                x=finger_tracker.circle_data[0], y=finger_tracker.circle_data[1]
            )
        )
    window_name = "Finger position " + dir_name
    cv2.imshow(
        window_name, frame
    )  # show first frame with the generated roi's from FingerTracker

    cv2.waitKey(CVWAIT) & 0xFF

    pause = True
    step = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        finger_tracker.track(frame)

        record = "{x} {y}\n".format(
            x=finger_tracker.circle_data[0], y=finger_tracker.circle_data[1]
        )
        print(str(step) + " - " + record, end="")
        if SAVE_DATA:
            finger_file.write(record)
        cv2.imshow(window_name, frame)

        # Exit if ESC pressed
        k = (
            cv2.waitKey(CVWAIT) & 0xFF
        )  # will get stucked here in the end if ESC is not pressed
        if k == 27:
            pause = False
            break
        step = step + 1

    if SAVE_DATA:
        finger_file.close()

    # When everything gets done, release the capture
    if pause:
        cv2.waitKey(-1)
    cap.release()
    cv2.destroyAllWindows()
