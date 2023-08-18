# yolov8-ncnn

## how to run
run `build/bin/yolov8` for any img/video(must be .mp4) input
```
cd build/bin
./yolov8 img|video filepath [modelpathWithoutExt]
# for video
./yolov8 video ../../video/test.mp4 ../../model/yolov8s
# for img
./yolov8 img ../../img/busstop.jpg ../../model/yolov8s
```
## about validation/test
run `build/test/coco/test_coco` for coco validation
```
cd build/bin/test/coco
./test_coco
```