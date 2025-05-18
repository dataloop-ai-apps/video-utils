import numpy as np
from trackings.utils import plot_one_box, load_opt
from numpy import random
import cv2
import sys
import dtlpy as dl
from dotenv import load_dotenv
import os

sys.path.append('./BoT_SORT')
from BoT_SORT.tracker.mc_bot_sort import BoTSORT

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('DATALOOP_API_KEY')
    dl.login_api_key(api_key=api_key)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    item = dl.items.get(item_id='682626b53bf48f4cb49072f3')
    item_file = item.download()
    annotations = item.annotations.list()

    tracker = BoTSORT(load_opt(), frame_rate=20.0)
    tracker.track_high_thresh = 0.11
    tracker.args.track_high_thresh = 0.11
    tracker.new_track_thresh = 0.2
    tracker.args.new_track_thresh = 0.2

    cap = cv2.VideoCapture(item_file)
    fn = -1
    t_id_set = set()
    while True:
        ret, orig_img = cap.read()
        if not ret:
            break
        fn += 1
        frame_annotation = annotations.get_frame(frame_num=fn).annotations
        print(f"-HHH- len(frame_annotation) {len(frame_annotation)}")
        detections = np.zeros((len(frame_annotation), 6))
        for i, ann in enumerate(frame_annotation):
            detections[i, :4] = [ann.top, ann.left, ann.bottom, ann.right]
            detections[i, 4] = 1
            detections[i, 5] = 0
        im0 = orig_img.copy()
        online_targets = tracker.update(detections, im0)
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls

            label = "person"
            print("-HHH- 42")
            if tlwh[2] * tlwh[3] > 5:
                print("-HHH- 45")
                # save results

                if fn == 0:
                    fixed = True
                else:
                    fixed = False

                label = f'{tid}, {label}'
                print(f"-HHH- tlbr {tlbr}")
                plot_one_box(
                    [tlbr[1], tlbr[0], tlbr[3], tlbr[2]],
                    im0,
                    label=label,
                    color=colors[int(tid) % len(colors)],
                    line_thickness=2,
                )

        cv2.imshow("im0", im0)
        cv2.waitKey(50)  # 1000ms/20fps = 50ms per frame
