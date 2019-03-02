# OpenVINO-EmotionRecognition
OpenVINO+NCS2/NCS+MutiModel(FaceDetection, EmotionRecognition)+MultiStick+MultiProcess+MultiThread+USB Camera/PiCamera. RaspberryPi 3 compatible.

```bash
$ sudo apt-get install -y python3-picamera
$ sudo -H pip3 install imutils --upgrade
$ git clone https://github.com/PINTO0309/OpenVINO-EmotionRecognition.git
$ cd OpenVINO-EmotionRecognition
$ python3 main.py
```

```bash
usage: main.py [-h] [-cm MODE_OF_CAMERA] [-cn NUMBER_OF_CAMERA]
               [-wd CAMERA_WIDTH] [-ht CAMERA_HEIGHT] [-numncs NUMBER_OF_NCS]
               [-vidfps FPS_OF_VIDEO] [-fdmp FD_MODEL_PATH]
               [-emmp EM_MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -cm MODE_OF_CAMERA, --modeofcamera MODE_OF_CAMERA
                        Camera Mode. 0:=USB Camera, 1:=PiCamera (Default=0)
  -cn NUMBER_OF_CAMERA, --numberofcamera NUMBER_OF_CAMERA
                        USB camera number. (Default=0)
  -wd CAMERA_WIDTH, --width CAMERA_WIDTH
                        Width of the frames in the video stream. (Default=640)
  -ht CAMERA_HEIGHT, --height CAMERA_HEIGHT
                        Height of the frames in the video stream.
                        (Default=480)
  -numncs NUMBER_OF_NCS, --numberofncs NUMBER_OF_NCS
                        Number of NCS. (Default=1)
  -vidfps FPS_OF_VIDEO, --fpsofvideo FPS_OF_VIDEO
                        FPS of Video. (Default=30)
  -fdmp FD_MODEL_PATH, --facedetectionmodelpath FD_MODEL_PATH
                        Face Detection model path. (xml and bin. Except
                        extension.)
  -emmp EM_MODEL_PATH, --emotionrecognitionmodelpath EM_MODEL_PATH
                        Emotion Recognition model path. (xml and bin. Except
                        extension.)

```
