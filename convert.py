import sys
import os

if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    start_frame_id = sys.argv[3]
    end_frame_id = sys.argv[4]
    
    print('src:', src_dir)
    print('dst:', dst_dir)
    print('start:', start_frame_id)
    print('end:', end_frame_id)

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))] 
    files.sort(key = lambda x: int(x[5:-4]))

    for i in range(start_frame_id, end_frame_id):
        print(i)
