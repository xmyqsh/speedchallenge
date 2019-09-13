import numpy as np
import cv2
from extract import create_records, write_records
import glob
from multiprocessing import Pool

def extract_segment(segment_info):
    num, path = segment_info
    frame_times = np.load(path + '/global_pose/frame_times')
    frame_velocities =  np.linalg.norm(np.load(path + '/global_pose/frame_velocities'),axis=1)
    frame_velocities = list(map(lambda x: x.item(), frame_velocities))

    for time, velocity in zip(frame_times[:20], frame_velocities[:20]):
        print(time, type(velocity))

    cam = cv2.VideoCapture(path + "/video.hevc")

    print(hash(path))

    examples = create_records(cam, frame_velocities)

    print(len(examples))

    write_records(examples, "/mnt/d/commaai/segments/segment_{}.tfrecords".format(num))



segment_dirs = glob.glob("/mnt/d/commaai/comma2k19/*/*/*")

print(len(segment_dirs))

enumerated_dirs = [x for x in enumerate(segment_dirs)]

print(enumerated_dirs[:10])

with Pool(32) as p:
    p.map(extract_segment, enumerated_dirs)

# extract_segment("/mnt/d/commaai/comma2k19/Chunk_1/b0c9d2329ad1606b_2018-07-27--06-03-57/3", 1)