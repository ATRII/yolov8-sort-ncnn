# YoloV8-ncnn

## How to run
run `build/bin/yolov8` for any img/video(must be .mp4) input
```
cd build/bin
./yolov8 img|video filepath [modelpathWithoutExt]
# for video
./yolov8 video ../../video/test.mp4 ../../model/yolov8s
# for img
./yolov8 img ../../img/busstop.jpg ../../model/yolov8s
```
## About validation/test
run `build/test/coco/vali_coco` for coco validation
```
cd build/bin/test/coco
./vali_coco
```
## SORT
run `build/bin/sort` for mutil-target tracking
```
cd build/bin
./sort test.mp4
```
## Modificatoin
### Display
#### yolo
![display_1](markdownlink/busstop.jpg)
- only draw when detect man/car/bus/...
- use different colors for each class of bouding box
#### sort
![display_2](markdownlink/test.gif)
### Output files' saving
- imgs under ./build/output/img
- videos under ./build/output/video
### Padding
- only use bottom/right padding(like yolox)
### Params
#### yolo
argv for `./yolov8 img|video filepath [modelpathWithoutExt]`
#### sort
argv for `./sort filenameWithExt [modelpathWithoutExt]`
