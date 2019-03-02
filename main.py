##############################################
# sudo apt-get install -y python3-picamera
# sudo -H pip3 install imutils --upgrade
##############################################

import sys
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading
try:
    from imutils.video.pivideostream import PiVideoStream
    from imutils.video.filevideostream import FileVideoStream
    import imutils
except:
    pass

lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
vs = None
window_name = ""
elapsedtime = 0.0

g_plugin = None
g_inferred_request = None
g_heap_request = None
g_inferred_cnt = 0
g_number_of_allocated_ncs = False

LABELS = ["neutral", "happy", "sad", "surprise", "anger"]
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

def camThread(LABELS, resultsEm, frameBuffer, camera_width, camera_height, vidfps, number_of_camera, mode_of_camera):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global vs
    global window_name


    if mode_of_camera == 0:
        cam = cv2.VideoCapture(number_of_camera)
        if cam.isOpened() != True:
            print("USB Camera Open Error!!!")
            sys.exit(0)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        window_name = "USB Camera"
    else:
        vs = PiVideoStream((camera_width, camera_height), vidfps).start()
        window_name = "PiCamera"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        # USB Camera Stream or PiCamera Stream Read
        color_image = None
        if mode_of_camera == 0:
            s, color_image = cam.read()
            if not s:
                continue
        else:
            color_image = vs.read()

        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        res = None

        if not resultsEm.empty():
            res = resultsEm.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults)

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        if cv2.waitKey(1)&0xFF == ord('q'):
            sys.exit(0)

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworkerFd, ncsworkerEm):

    while True:
        ncsworkerFd.predict_async()
        ncsworkerEm.predict_async()


class BaseNcsWorker():

    def __init__(self, devid, model_path, number_of_ncs):
        global g_plugin
        global g_inferred_request
        global g_heap_request
        global g_inferred_cnt
        global g_number_of_allocated_ncs

        self.devid = devid
        self.num_requests = 4

        if g_number_of_allocated_ncs < number_of_ncs:
            self.plugin = IEPlugin(device="MYRIAD")
            self.inferred_request = [0] * self.num_requests
            self.heap_request = []
            self.inferred_cnt = 0
            g_plugin = self.plugin
            g_inferred_request = self.inferred_request
            g_heap_request = self.heap_request
            g_inferred_cnt = self.inferred_cnt
            g_number_of_allocated_ncs += 1
        else:
            self.plugin = g_plugin
            self.inferred_request = g_inferred_request
            self.heap_request = g_heap_request
            self.inferred_cnt = g_inferred_cnt

        self.model_xml = model_path + ".xml"
        self.model_bin = model_path + ".bin"
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)


class NcsWorkerFd(BaseNcsWorker):

    def __init__(self, devid, frameBuffer, resultsFd, model_path, number_of_ncs):

        super().__init__(devid, model_path, number_of_ncs)
        self.frameBuffer = frameBuffer
        self.resultsFd   = resultsFd


    def image_preprocessing(self, color_image):

        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            color_image = self.frameBuffer.get()
            prepimg = self.image_preprocessing(color_image)
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1

                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0

                self.exec_net.requests[reqnum].wait(-1)
                out = self.exec_net.requests[reqnum].outputs["detection_out"].flatten()

                detection_list = []
                face_image_list = []

                for detection in out.reshape(-1, 7):

                    confidence = float(detection[2])

                    if confidence > 0.3:
                        detection[3] = int(detection[3] * color_image.shape[1])
                        detection[4] = int(detection[4] * color_image.shape[0])
                        detection[5] = int(detection[5] * color_image.shape[1])
                        detection[6] = int(detection[6] * color_image.shape[0])
                        if (detection[6] - detection[4]) > 0 and (detection[5] - detection[3]) > 0:
                            detection_list.extend(detection)
                            face_image_list.extend([color_image[int(detection[4]):int(detection[6]), int(detection[3]):int(detection[5]), :]])

                if len(detection_list) > 0:
                    self.resultsFd.put([detection_list, face_image_list])

                self.inferred_request[reqnum] = 0


        except:
            import traceback
            traceback.print_exc()


class NcsWorkerEm(BaseNcsWorker):

    def __init__(self, devid, resultsFd, resultsEm, model_path, number_of_ncs):

        super().__init__(devid, model_path, number_of_ncs)
        self.resultsFd = resultsFd
        self.resultsEm = resultsEm


    def image_preprocessing(self, color_image):

        try:
            prepimg = cv2.resize(color_image, (64, 64))
        except:
            prepimg = np.full((64, 64, 3), 128)
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.resultsFd.empty():
                return

            resultFd = self.resultsFd.get()
            detection_list  = resultFd[0]
            face_image_list = resultFd[1]
            emotion_list    = []
            max_face_image_list_cnt = len(face_image_list)
            image_idx = 0
            end_cnt_processing = 0
            heapflg = False
            cnt = 0
            dev = 0

            if max_face_image_list_cnt <= 0:
                detection_list.extend([""])
                self.resultsEm.put([detection_list])
                return

            while True:
                reqnum = searchlist(self.inferred_request, 0)

                if reqnum > -1 and image_idx <= (max_face_image_list_cnt - 1) and len(face_image_list[image_idx]) > 0:

                    if len(face_image_list[image_idx]) == []:
                        image_idx += 1
                        continue
                    else:
                        prepimg = self.image_preprocessing(face_image_list[image_idx])
                        image_idx += 1

                    self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                    self.inferred_request[reqnum] = 1
                    self.inferred_cnt += 1
                    if self.inferred_cnt == sys.maxsize:
                        self.inferred_request = [0] * self.num_requests
                        self.heap_request = []
                        self.inferred_cnt = 0
                    heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))
                    heapflg = True

                if heapflg:
                    cnt, dev = heapq.heappop(self.heap_request)
                    heapflg = False

                if self.exec_net.requests[dev].wait(0) == 0:
                    self.exec_net.requests[dev].wait(-1)
                    out = self.exec_net.requests[dev].outputs["prob_emotion"].flatten()
                    emotion = LABELS[int(np.argmax(out))]
                    detection_list.extend([emotion])
                    self.resultsEm.put([detection_list])
                    self.inferred_request[dev] = 0
                    end_cnt_processing += 1
                    if end_cnt_processing >= max_face_image_list_cnt:
                        break
                else:
                    heapq.heappush(self.heap_request, (cnt, dev))
                    heapflg = True

        except:
            import traceback
            traceback.print_exc()


def inferencer(resultsFd, resultsEm, frameBuffer, number_of_ncs, fd_model_path, em_model_path):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        # Face Detection, Emotion Recognition start
        thworker = threading.Thread(target=async_infer, args=(NcsWorkerFd(devid, frameBuffer, resultsFd, fd_model_path, number_of_ncs),
                                                              NcsWorkerEm(devid, resultsFd, resultsEm, em_model_path, 0),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


def overlay_on_image(frames, object_infos):

    try:

        color_image = frames

        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()

        for object_info in object_infos:

            if object_info[2] == 0.0:
                break

            if (not np.isfinite(object_info[0]) or
                not np.isfinite(object_info[1]) or
                not np.isfinite(object_info[2]) or
                not np.isfinite(object_info[3]) or
                not np.isfinite(object_info[4]) or
                not np.isfinite(object_info[5]) or
                not np.isfinite(object_info[6])):
                continue

            min_score_percent = 60
            source_image_width = width
            source_image_height = height
            percentage = int(object_info[2] * 100)

            if (percentage <= min_score_percent):
                continue

            box_left   = int(object_info[3])
            box_top    = int(object_info[4])
            box_right  = int(object_info[5])
            box_bottom = int(object_info[6])
            emotion    = str(object_info[7])

            label_text = emotion + " (" + str(percentage) + "%)"

            box_color =  COLORS[searchlist(LABELS, emotion, 0)]
            box_thickness = 2
            cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
            label_background_color = (125, 175, 75)
            label_text_color = (255, 255, 255)
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = box_left
            label_top = box_top - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
            cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


        cv2.putText(img_cp, fps,       (width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        return img_cp

    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cm','--modeofcamera',dest='mode_of_camera',type=int,default=0,help='Camera Mode. 0:=USB Camera, 1:=PiCamera (Default=0)')
    parser.add_argument('-cn','--numberofcamera',dest='number_of_camera',type=int,default=0,help='USB camera number. (Default=0)')
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=640,help='Width of the frames in the video stream. (Default=640)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=480,help='Height of the frames in the video stream. (Default=480)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (Default=30)')
    parser.add_argument('-fdmp','--facedetectionmodelpath',dest='fd_model_path',default='./FP16/face-detection-retail-0004',help='Face Detection model path. (xml and bin. Except extension.)')
    parser.add_argument('-emmp','--emotionrecognitionmodelpath',dest='em_model_path',default='./FP16/emotions-recognition-retail-0003',help='Emotion Recognition model path. (xml and bin. Except extension.)')

    args = parser.parse_args()
    mode_of_camera = args.mode_of_camera
    number_of_camera = args.number_of_camera
    camera_width  = args.camera_width
    camera_height = args.camera_height
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video
    fd_model_path = args.fd_model_path
    em_model_path = args.em_model_path

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        resultsFd = mp.Queue() # Face Detection Queue
        resultsEm = mp.Queue() # Emotion Recognition Queue

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(LABELS, resultsEm, frameBuffer, camera_width, camera_height, vidfps, number_of_camera, mode_of_camera),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(resultsFd, resultsEm, frameBuffer, number_of_ncs, fd_model_path, em_model_path),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")
