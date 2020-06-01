import sys

if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    start_frame_id = sys.argv[3]
    end_frame_id = sys.argv[4]
    
    print('src:', src_dir)
    print('dst:', dst_dir)
    print('start:', start_frame_id)
    print('end:', end_frame_id)