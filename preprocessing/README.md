
We explain preprocessing steps more details here. Feel free to contact if you have problems running them.


### 2D object detection
We use [Yolo](https://pjreddie.com/darknet/yolo/) to detect 2D object bounding box, which can be replaced by other algorithms. We add functions to batch predict all images under a folder and save results to images and txts. See the modification code under ```2D_object_detect```. Might need to change rgb file name in line 495 of detector.c


Create two folders to save results of images and txts:  ```yolov2_obj```  ```yolov2_obj_txts```. After compiling Yolo, run command:

```bash
./darknet detect_folder cfg/yolo.cfg trained_model/yolo.weights  path/to/rgb path/to/yolov2_obj -thresh 0.25
```

Pay attention to the cfg and weights for different yolo version.
