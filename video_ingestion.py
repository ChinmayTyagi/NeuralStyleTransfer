import cv2 
import os 
import shutil
import numpy as np

class VideoInfo:
    def __init__(self, name, fps):
        self.name = name
        self.fps = fps

def get_path():
    while True:
        rel_path = input('Enter relative path of video to process: ')
        if os.path.exists(rel_path):
            return rel_path
        else:
            print("File does not exist, try again")
    
def video_to_frames(file_path):
    vid = cv2.VideoCapture(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    try: 
        if not os.path.exists(f'output_frames'): 
            os.mkdir(f'output_frames') 
    except OSError: 
        print ('Error: Failed to create output_frames directory') 

    currentframe = 0

    ret, frame = vid.read() 
    while(ret): 
        name = f'./output_frames/frame{str(currentframe)}.jpg'
        print(f'Creating..{name}') 

        # ANY VIDEO PROCESSING WOULD GO HERE

        cv2.imwrite(name, frame) 
        currentframe += 1
        ret, frame = vid.read()

    fps = 0
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {fps}")
    else :
        fps = vid.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

    vid.release() 
    cv2.destroyAllWindows() 
    return VideoInfo(file_name, fps)


def frames_to_video(video_info):
    dir_path = 'output_frames/'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] 
    files.sort(key = lambda x: int(x[5:-4]))

    frames = []
    size = ()

    for i in range(len(files)):
        filename = dir_path + files[i]
        frame = cv2.imread(filename)
        h, w, layers = frame.shape
        size = (w, h)
        frames.append(frame)
    
    out = cv2.VideoWriter(f'{video_info.name}_stylized.mp4',cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, size)

    print(f"Writing final video to '{video_info.name}_stylized.mp4'...")
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def clear_working_dirs():
    print("Clearing directory 'output_frames'... ")
    if os.path.exists(f'output_frames'): 
        shutil.rmtree('output_frames')


if __name__ == "__main__":
    clear_working_dirs()
    video_info = video_to_frames(get_path())
    frames_to_video(video_info)

    
    

