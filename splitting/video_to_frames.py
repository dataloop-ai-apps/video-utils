import logging
import os
import tempfile

import cv2
import dtlpy as dl
from dotenv import load_dotenv
import os
from skimage.metrics import structural_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
import numpy as np


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.embedder = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def get_embedding(self, frame):
        # Resize and preprocess the frame
        img = cv2.resize(frame, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        # Get the embedding
        return self.embedder.predict(img).flatten()

    def get_frames_list(self, cap):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if self.split_type == 'frames_interval':
            print(f"frames_interval: {self.splitter_arg}")
            return list(range(0, total_frames, self.splitter_arg))
        if self.split_type == 'time_interval':
            print(f"time_interval: {self.splitter_arg}")
            return list(range(0, total_frames, int(fps * self.splitter_arg)))
        if self.split_type == 'num_splits':
            print(f"num_splits: {self.splitter_arg}")
            divisor = (self.splitter_arg - 1) - (0 if total_frames % (self.splitter_arg - 1) else 1)
            frames_interval = total_frames // divisor
            print(f"frames_interval: {frames_interval}")
            return list(range(0, total_frames, frames_interval))
        if self.split_type == 'embedding_similarity_sampling':
            print(f"embedding_similarity_sampling: {self.threshold}, {self.min_interval}")
            frames_list = []
            frame_count = 0
            prev_embedding = None
            min_interval = int(fps * self.min_interval)  # convert seconds to frames
            i_saved = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                if (frame_count - i_saved) < min_interval:
                    frame_count += 1
                    continue

                if prev_embedding is None:
                    prev_embedding = self.get_embedding(frame)
                    frames_list.append(frame_count)
                    i_saved = frame_count
                    frame_count += 1
                    continue

                embedding = self.get_embedding(frame)
                distance = cosine(prev_embedding, embedding)

                if distance > self.threshold:
                    frames_list.append(frame_count)
                    i_saved = frame_count
                    prev_embedding = embedding

                frame_count += 1

            return frames_list

        # structural similarity sampling
        print(f"structural_similarity_sampling: {self.window_size}, {self.threshold}, {self.min_interval}")

        frames_list = []
        frame_count = 0
        reference = None
        min_interval = int(fps * self.min_interval)  # convert seconds to frames
        i_saved = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if (frame_count - i_saved) < min_interval:
                frame_count += 1
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if reference is None:
                reference = frame_gray
                frames_list.append(frame_count)
                i_saved = frame_count
                frame_count += 1
                continue
            ssim = structural_similarity(reference, frame_gray, win_size=self.window_size)
            # Convert SSIM to distance measure (1 - SSIM) so higher values mean more different
            ssim_distance = 1 - ssim
            if ssim_distance > self.threshold:
                frames_list.append(frame_count)
                reference = frame_gray
                i_saved = frame_count
            frame_count += 1
        return frames_list

    def upload_frames(self, item, frames_list, cap, temp_dir):
        if len(frames_list) == 0:
            return
        num_digits = len(str(max(frames_list)))
        item_dataset = item.dataset
        annotations = item.annotations.list()
        # TODO : check if using batch upload will reduce upload time.
        for frame_idx in frames_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break
            frame_path = os.path.join(
                temp_dir,
                f"{os.path.splitext(os.path.basename(item.filename))[0]}_{str(frame_idx).zfill(num_digits)}.jpg",
            )
            cv2.imwrite(frame_path, frame)
            frame_item = item_dataset.items.upload(local_path=frame_path, remote_path=self.dl_output_folder)
            if annotations:
                frame_annotation = annotations.get_frame(frame_num=frame_idx)
                builder = frame_item.annotations.builder()
                for ann in frame_annotation.annotations:
                    if ann.object_visible:
                        builder.add(
                            annotation_definition=dl.Box(
                                top=ann.top, left=ann.left, bottom=ann.bottom, right=ann.right, label=ann.label
                            )
                        )
                    frame_item.annotations.upload(builder)

    def video_to_frames(self, item: dl.Item, context: dl.Context):
        logger = logging.getLogger('video-utils.video_to_frames')
        logger.info('Running service Video To Frames')

        node = context.node
        self.split_type = node.metadata['customNodeConfig'].get('split_type', 'structural_similarity_sampling')
        self.dl_output_folder = node.metadata['customNodeConfig']['output_dir']

        if self.split_type != 'structural_similarity_sampling' and self.split_type != 'embedding_similarity_sampling':
            self.splitter_arg = node.metadata['customNodeConfig']['splitter_arg']
        else:
            self.threshold = node.metadata['customNodeConfig']['threshold']
            self.min_interval = node.metadata['customNodeConfig']['min_interval']
            if self.split_type == 'structural_similarity_sampling':
                self.window_size = node.metadata['customNodeConfig']['window_size']

        temp_dir = tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)
        input_video = item.download(local_path=temp_dir)
        cap = cv2.VideoCapture(input_video)
        frames_list = self.get_frames_list(cap)
        self.upload_frames(item, frames_list, cap, temp_dir)
        cap.release()


if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv('DATALOOP_API_KEY')
    dl.login_api_key(api_key=api_key)
    if dl.token_expired():
        dl.login()

    runner = ServiceRunner()
    context = dl.Context()
    context.pipeline_id = "682069122afb795bc3c41d59"
    context.node_id = "bd1dc151-6067-4197-85aa-1b65394e2077"
    # context.node.metadata["customNodeConfig"] = {
    #     "split_type": "frames_interval",
    #     "splitter_arg": 1,
    #     "output_dir": "/white_dancers_frames",
    # }

    context.node.metadata["customNodeConfig"] = {
        "split_type": "structural_similarity_sampling",
        "window_size": 9,
        "threshold": 0.8,
        "min_interval": 0.2,
        "output_dir": "/check_structural_similarity_high_threshold_1905",
    }
    runner.video_to_frames(item=dl.items.get(item_id="6821ec8fb188d7f242334661"), context=context)
