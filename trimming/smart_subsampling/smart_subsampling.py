import os
import cv2
import datetime
import dtlpy as dl
from skimage.metrics import structural_similarity


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        ...

    @staticmethod
    def upload_frame(item, frame, fc, input_base_name, output_folder, dataset):
        """
        uploads frame to Dataloop
        :param item: the video item, where the frames are taken from
        :param frame: the frame to upload
        :param fc: the number of the frame
        :param input_base_name: the base name of the frame
        :param output_folder: the remote folder path
        :param dataset: the dataset to upload the item to
        """
        tmp_file = f"{input_base_name}_{fc}_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg"
        cv2.imwrite(tmp_file, frame)  # save frame as JPEG file
        frame_item = dataset.items.upload(local_path=tmp_file,
                                          remote_path=output_folder,
                                          item_metadata={
                                              "user": {"parentItemId": item.id}
                                          })
        os.remove(tmp_file)
        return frame_item

    @staticmethod
    def video_to_frames_smart_subsampling(item, output_folder, threshold, window_size):
        """
        splits video to sub videos by similarity of frames
        :param item: the video item to split
        :param output_folder: the remote output folder
        :param threshold: the threshold of the similarity
        :param window_size: the side-length of the sliding window used in comparison
        """
        frame_count = 0
        frame_items = list()
        dataset = item.dataset
        input_base_name = os.path.splitext(os.path.basename(item.filename))[0]

        video = item.download()
        vidcap = cv2.VideoCapture(video)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_fc_len = len(str(total_frames))

        success, frame_rgb = vidcap.read()
        reference = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        frame_item = ServiceRunner.upload_frame(item, frame_rgb, str(frame_count).zfill(max_fc_len), input_base_name,
                                                output_folder, dataset)
        frame_items.append(frame_item)
        success, frame_rgb = vidcap.read()

        while success:
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            ssim = structural_similarity(reference, frame_gray, win_size=window_size)
            if ssim <= threshold:
                frame_count += 1
                frame_item = ServiceRunner.upload_frame(item, frame_rgb, str(frame_count).zfill(max_fc_len),
                                                        input_base_name, output_folder, dataset)
                frame_items.append(frame_item)
                reference = frame_gray
            success, frame_rgb = vidcap.read()

        print(f'Uploaded {frame_count + 1}/{int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))} frames successfully.')
        os.remove(video)
        return item, frame_items
