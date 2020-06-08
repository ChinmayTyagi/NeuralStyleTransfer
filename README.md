# NeuralStyleTransfer

How do humans perceive artistic imagery? This project, created by Chinmay Tyagi, Michael Auld, and Tycho Bellers, aims to create artificial artwork by manipulating an image to adopt the visual style of another image. Eg, "Picasso-fy" a selfie of myself to adopt the style of one of his paintings. 

### Demo
Run the included `demo.py`. Note that it can take a few minues to run on CPU. Increasing the number of epochs will make the output look better but currently the demo only does 1 epoch so it doesn't take too long.

### Video Processing
To run the style transfer on a video, follow these steps:
1) Run `video_ingestion.py`. This will take a single video and convert each frame to a jpeg.
2) Open the java project `java/Nerual` in intelliJ and edit the parameters of the convert function to match your liking. (This step was required due to the memory issue with Tensorflow and GPU memory allocation. If you are using CPU only, you can probably just run `convert2.py` directly with the appropriate arguments and skip step 3. `convert2.py` takes the following arguments: `<src_dir> <dst_dir> <start_frame_id> <end_frame_id>`)
3) Run the java program. This will style each frame and output it to a new directory.
4) Edit `video_ingestion.py` to call the `frames_to_video` function on the appropriate directory to convert the images back into a video.
