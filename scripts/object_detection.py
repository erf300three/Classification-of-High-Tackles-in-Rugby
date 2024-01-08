from ultralytics import YOLO
from ultralytics.solutions import heatmap
import os
import torch 
import cv2

from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

print("Hello, World!")

def object_tracking(input_path, output_path, yolo):
    """
    Tracks objects in the video
    :param video_path: path to the video
    :param output_path: path to the output video
    :param yolo: yolo object
    :return: whether the operation was successful
    """

    results = yolo.track(input_path, show=False, save=True, project=output_path, name='tracked', tracker="bytetrack.yaml", persist=True, classes=[0,32])


    # results = model.track(video_path, show=True, save=True, save_dir=output_path, tracker="bytetrack.yaml")
    
# torch.cuda.set_device(0)

model = YOLO('yolov8x.pt')
# model.to("cuda")


def heatmap_tracking(input_path):
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"

    # Video writer
    video_writer = cv2.VideoWriter("heatmap_line_counting_output.avi",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   int(cap.get(5)), # This is the number of frames per second
                                   (int(cap.get(3)), int(cap.get(4)))) # This is the height and width of the video

    line_points = [(100, 300), (600, 400)]  # line for object counting
    # Init heatmap
    heatmap_obj = heatmap.Heatmap()
    heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA ,
                         imw=cap.get(4),  # should same as cap height
                         imh=cap.get(3),  # should same as cap width
                         view_img=True,
                         shape="circle", 
                         count_reg_pts=line_points)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False)

        im0 = heatmap_obj.generate_heatmap(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def instance_segmentation_tracking(input_path):
    track_history = defaultdict(lambda: [])
    model = YOLO("yolov8n-seg.pt")
    cap = cv2.VideoCapture(input_path)

    out = cv2.VideoWriter('instance-segmentation-object-tracking.avi',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          int(cap.get(5)),
                         (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.track(im0, persist=True)
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotator = Annotator(im0, line_width=2)

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=str(track_id))

        out.write(im0)
        # This is what displays the video to the screen so don't uncomment it on ssh DCS
        # cv2.imshow("instance-segmentation-object-tracking", im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows() 

# object_tracking("clips", "output" , model)
# heatmap_tracking("clips/Barnes_v_Wombledon_Yellow_Card_High_Tackle1_trimmed.mp4")
instance_segmentation_tracking("data/output_set/trimmed/Barnes_v_Wombledon_Yellow_Card_High_Tackle1_trimmed.mp4")
