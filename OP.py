"""
    My own layer above the official Openpose python wrapper : https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
    Tested only on ubuntu.
    Modify MODEL_FOLDER to point to the directory where the models are installed
"""

import cv2
import os, sys
from sys import platform
import argparse
from collections import namedtuple
import numpy as np
from FPS import FPS
from matplotlib import pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":   #如果是32位就更改路径到64位
        # 更改这些变量以指向正确的文件夹（Release/x64等） 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # 更改这些变量以指向正确的文件夹（Release/x64等） 
        sys.path.append('/usr/local/python');
        # 如果您运行“make-install”（Ubuntu的默认路径为“/usr/local/python”），您也可以从那里访问OpenPose/python模块。这将在您想要的安装路径上安装OpenPose和python库。确保它在您的python路径中以便使用它.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

MODEL_FOLDER = "/home/gx/openpose/models/"

body_kp_id_to_name = {
    0: "Nose", #鼻子
    1: "Neck", #脖子
    2: "RShoulder", #右肩膀
    3: "RElbow", #右肘
    4: "RWrist", #右腕
    5: "LShoulder", #左肩膀
    6: "LElbow", #左肘
    7: "LWrist", #左腕
    8: "MidHip", #髋部
    9: "RHip", #右臀
    10: "RKnee", #右膝
    11: "RAnkle", #右脚踝
    12: "LHip", #左臀
    13: "LKnee", #左膝
    14: "LAnkle", #左脚踝
    15: "REye", #右眼
    16: "LEye", #左眼
    17: "REar", #右耳
    18: "LEar", #左耳
    19: "LBigToe", #左大拇指
    20: "LSmallToe", #左小拇指
    21: "LHeel", #左脚后跟
    22: "RBigToe", #右大拇指
    23: "RSmallToe", #右小拇指
    24: "RHeel", #右脚后跟
    25: "Background"}


body_kp_name_to_id = {v: k for k, v in body_kp_id_to_name.items()} # body_kp_name_to_id的值成为新字典的键，body_kp_name_to_id的键成为新字典的值

Pair = namedtuple('Pair', ['p1', 'p2', 'color'])
color_right_side = (0,255,0) #绿色
color_left_side = (0,0,255) #蓝色
color_middle = (0,255,255) #青色
color_face = (255,255,255) #白色

#左边置蓝色，右边置绿色  
pairs_head = [
    Pair("Nose", "REye", color_right_side),
    Pair("Nose", "LEye", color_left_side),
    Pair("REye", "REar", color_right_side),
    Pair("LEye", "LEar", color_left_side)
]

pairs_upper_limbs = [
    Pair("Neck", "RShoulder", color_right_side),
    Pair("RShoulder", "RElbow", color_right_side),
    Pair("RElbow", "RWrist", color_right_side),
    Pair("Neck", "LShoulder", color_left_side),
    Pair("LShoulder", "LElbow", color_left_side),
    Pair("LElbow", "LWrist", color_left_side)
]

pairs_lower_limbs = [
    Pair("MidHip", "RHip", color_right_side),
    Pair("RHip", "RKnee", color_right_side),
    Pair("RKnee", "RAnkle", color_right_side),
    Pair("RAnkle", "RHeel", color_right_side),
    Pair("MidHip", "LHip", color_left_side),
    Pair("LHip", "LKnee", color_left_side),
    Pair("LKnee", "LAnkle", color_left_side),
    Pair("LAnkle", "LHeel", color_left_side)
]

pairs_spine = [
    Pair("Nose", "Neck", color_middle),
    Pair("Neck", "MidHip", color_middle)
]

pairs_feet = [
    Pair("RAnkle", "RBigToe", color_right_side),
    Pair("RAnkle", "RHeel", color_right_side),
    Pair("LAnkle", "LBigToe", color_left_side),
    Pair("LAnkle", "LHeel", color_left_side)
]


pairs_body = pairs_head + pairs_upper_limbs + pairs_lower_limbs + pairs_spine + pairs_feet

face_kp_id_to_name = {}
for i in range(17):
    face_kp_id_to_name[i] = f"Jaw{i+1}" #为下颌的17个关键点命名为Jaw1到Jaw17
for i in range(5):
    face_kp_id_to_name[i+17 ] = f"REyebrow{5-i}"
    face_kp_id_to_name[i+22] = f"LEyebrow{i+1}"  #为左右眉毛的5个关键点命名
for i in range(6):
    face_kp_id_to_name[(39-i) if i<4 else (45-i)] = f"REye{i+1}"
    face_kp_id_to_name[i+42] = f"LEye{i+1}"   
face_kp_id_to_name[68] = "REyeCenter"
face_kp_id_to_name[69] = "LEyeCenter"  #为右眼睛的6个关键点赋予名称，命名为"REye1"到"REye6"，同时为左眼睛的6个关键点赋予名称，命名为"LEye1"到"LEye6，同时68和69号关键点被命名为"REyeCenter"和"LEyeCenter"
for i in range(9):
    face_kp_id_to_name[27+i] = f"Nose{i+1}" #为鼻子的9个关键点赋予名称
for i in range(12):
    face_kp_id_to_name[i+48] = f"OuterLips{i+1}" #为嘴唇外侧的12个关键点赋予名称
for i in range(8):
    face_kp_id_to_name[i+60] = f"InnerLips{i+1}" #为嘴唇内侧的8个关键点赋予名称

face_kp_name_to_id = {v: k for k, v in face_kp_id_to_name.items()} #face_kp_name_to_id的值成为新字典的键，facee_kp_name_to_id的键成为新字典的值

pairs_jaw = [ Pair(f"Jaw{i+1}", f"Jaw{i+2}", color_face) for i in range(16)] #pairs_jaw 列表中包含了所有相邻下颌关键点之间的配对，用于描述下颌的结构
pairs_nose = [ Pair(f"Nose{i+1}", f"Nose{i+2}", color_face) for i in range(3)] + [ Pair(f"Nose{i+1}", f"Nose{i+2}", color_face) for i in range(4,8)] #包含描述鼻子关键点之间配对关系的对象列表。该列表中的对象表示鼻子关键点之间的连续配对，由多个列表合并而成

pairs_left_eye = [ Pair(f"LEye{i+1}", f"LEye{i+2}", color_face) for i in range(5)] + [Pair("LEye6","LEye1",color_face)]
pairs_right_eye = [ Pair(f"REye{i+1}", f"REye{i+2}", color_face) for i in range(5)] + [Pair("REye6","REye1",color_face)]
pairs_eyes = pairs_left_eye + pairs_right_eye #分别表示左眼和右眼关键点之间的配对关系的对象列表。这些列表中的对象由多个列表合并而成。

pairs_left_eyebrow = [ Pair(f"LEyebrow{i+1}", f"LEyebrow{i+2}", color_face) for i in range(4)]
pairs_right_eyebrow = [ Pair(f"REyebrow{i+1}", f"REyebrow{i+2}", color_face) for i in range(4)]
pairs_eyesbrow = pairs_left_eyebrow + pairs_right_eyebrow #分别表示左眉和右眉关键点之间的配对关系的对象列表。这些列表中的对象由多个列表合并而成

pairs_outer_lips = [ Pair(f"OuterLips{i+1}", f"OuterLips{i+2}", color_face) for i in range(11)] + [Pair("OuterLips12","OuterLips1",color_face)]
pairs_inner_lips = [ Pair(f"InnerLips{i+1}", f"InnerLips{i+2}", color_face) for i in range(7)] + [Pair("InnerLips8","InnerLips1",color_face)]
pairs_mouth = pairs_outer_lips + pairs_inner_lips #分别表示嘴唇外侧和内侧关键点之间的配对关系的对象列表。这些列表中的对象由多个列表合并而成

pairs_face = pairs_jaw + pairs_nose + pairs_eyes + pairs_eyesbrow + pairs_mouth

class OP:
    @staticmethod
    # def distance_kps(kp1,kp2):
    #     x1,y1,c1 = kp1
    #     x2,y2,c2 = kp2
    #     if c1 > 0 and c2 > 0:
    #         return abs(x2-x1)+abs(y2-y1)
    #     else: 
    #         return 0
    def distance_kps(kp1,kp2):  #用于计算两个关键点的距离
        # kp1 and kp2: numpy array of shape (3,): [x,y,conf]
        x1,y1,c1 = kp1
        x2,y2,c2 = kp2
        if kp1[2] > 0 and kp2[2] > 0:
            return np.linalg.norm(kp1[:2]-kp2[:2])
        else: 
            return 0
    @staticmethod
    def distance (p1, p2):
            """
                Distance between p1(x1,y1) and p2(x2,y2)
            """
            return np.linalg.norm(np.array(p1)-np.array(p2))

    def __init__(self, number_people_max=-1, min_size=-1, openpose_rendering=False, face_detection=False, frt=0.4, hand_detection=False, debug=False):
        #设置最大检测人数默认为-1，最小检测尺寸默认为-1，如果openpose_rendering为True则使用Openpose渲染，默认为False,如果face_detection为True则进行人脸检测，frt设置人脸渲染阈值，默认为0.4,hand_detection用于手部检测,debug为True则使用调试模式
        """
        openpose_rendering : if True, rendering is made by original Openpose library. Otherwise rendering is to the
        responsability of the user (~0.2 fps faster)
        """
        self.openpose_rendering = openpose_rendering
        self.min_size = min_size
        self.debug = debug
        self.face_detection = face_detection
        self.hand_detection = hand_detection
        self.frt = frt
        
        params = dict()
        params["model_folder"] = MODEL_FOLDER  #模型文件夹路径
        params["model_pose"] = "BODY_25"  #姿势模型类型
        params["number_people_max"] = number_people_max #最大检测人数
        if not self.openpose_rendering: #是否使用Openpose渲染
            params["render_pose"] = 0
        if self.face_detection:
            params["face"] = True #是否使用人脸检测
            params["face_render_threshold"] = self.frt #是否启用人脸渲染
        # if self.hand_detection:
            params["hand"] = True
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()  #创建了一个 OpenPose 的 Python 包装器对象
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.datum = op.Datum()

        
    def eval(self, frame):
        self.frame = frame
        self.datum.cvInputData = frame.copy()
        self.opWrapper.emplaceAndPop([self.datum])
        if self.openpose_rendering:
            cv2.imshow("Openpose", self.datum.cvOutputData)
        if self.datum.poseKeypoints.shape: # 当未检测到人时，shape=（），else（nb_persons，25，3）
            self.body_kps = self.datum.poseKeypoints
            self.face_kps = self.datum.faceKeypoints


            # 我们根据人们对体型的“估计”对他们进行分类
            # 大小与一个人的实际大小无关，但它是一个任意值，此处计算为距离（鼻子、脖子）+0.33*距离（脖子、臀部）
            sizes = np.array([self.length(pairs_spine, person_idx=i, coefs=[1, 0.33]) for i in range(self.body_kps.shape[0])])

            # 从大到小分类
            order = np.argsort(-sizes)
            sizes = sizes[order]
            #if self-debug：打印（“OpenPose size=”，size）
            self.body_kps = self.body_kps[order]

            # 只留住足够“高大”的人
            big_enough = sizes > self.min_size
            self.body_kps = self.body_kps[big_enough]

            self.nb_persons = len(self.body_kps)

            if self.face_detection: 
                self.face_kps = self.face_kps[order]
                self.face_kps = self.face_kps[big_enough]
                filter=self.face_kps[:,:,2]<self.frt
                self.face_kps[filter] = 0

        else:
            self.nb_persons = 0
            self.body_kps = []
            self.face_kps = [] 
        
        return self.nb_persons,self.body_kps, self.face_kps
 #这段代码的作用是根据 face_detection 属性的值来决定是否处理人脸关键点，并根据设定的阈值 self.frt 过滤出置信度较高的关键点
    
    def draw_pairs_person(self, frame, kps, kp_name_to_id, pairs, person_idx=0, thickness=3, color=None): # 用于在给定的图像帧上绘制人体关键点对之间的连线
        """
            Draw on 'frame' pairs of keypoints 
        """
        person = kps[person_idx]
        for pair in pairs:
            p1_x,p1_y,p1_conf = person[kp_name_to_id[pair.p1]]
            p2_x,p2_y,p2_conf = person[kp_name_to_id[pair.p2]]
            if p1_conf != 0 and p2_conf != 0:
                col = color if color else pair.color #如果未指定颜色，则使用关键点对对象中定义的颜色 pair.color
                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), col, thickness)

    def draw_pairs(self, frame, kps, kp_name_to_id, pairs, thickness=3, color=None): #函数首先通过循环遍历每个人体，然后调用 draw_pairs_person 方法来为每个人体绘制关键点对的连线。在绘制时，使用指定的颜色 color，如果未指定颜色，则使用默认的颜色。
        """
            Draw on 'frame' pairs of keypoints 
        """
        for person_idx in range(self.nb_persons):
            self.draw_pairs_person(frame, kps, kp_name_to_id, pairs, person_idx, thickness=thickness, color=color)


    def draw_body(self, frame, pairs=pairs_body, thickness=3, color=None):
        """
            Draw on 'frame' pairs of keypoints 
        """
        self.draw_pairs(frame, self.body_kps, body_kp_name_to_id, pairs, thickness, color)

    def draw_face(self, frame, pairs=pairs_face, thickness=2, color=None):
        """
            Draw on 'frame' pairs of keypoints 
        """
        self.draw_pairs(frame, self.face_kps, face_kp_name_to_id, pairs, thickness, color)

    def draw_eyes_person (self, frame, person_idx=0):
        eyes_status = self.check_eyes(person_idx=person_idx)
        if eyes_status in [1,3]:
            color = (0,200,230)
        else:
            color = (230,230,0)
        self.draw_pairs_person(frame,self.face_kps,face_kp_name_to_id,pairs_right_eye,person_idx,2,color)
        if eyes_status in [2,3]:
            color = (0,200,230)
        else:
            color = (230,230,0)
        self.draw_pairs_person(frame,self.face_kps,face_kp_name_to_id,pairs_left_eye,person_idx,2,color)


    def draw_eyes (self, frame):
        for person_idx in range(self.nb_persons):
            self.draw_eyes_person(frame, person_idx)

    def get_body_kp(self, kp_name="Neck", person_idx=0):  #是从人体关键点信息中获取指定关键点的坐标，用于后续的处理或可视化
        """
            Return the coordinates of a keypoint named 'kp_name' of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            kps = self.datum.poseKeypoints[person_idx]  #函数首先尝试从 datum.poseKeypoints 中获取指定索引 person_idx 的人体关键点信息 kps，如果指定的索引无效，则打印错误信息并返回 None
        except:
            print(f"get_body_kp: invalid person_idx '{person_idx}'")
            return None
        try:  
            x,y,conf = kps[body_kp_name_to_id[kp_name]]  #然后，尝试从 kps 中获取指定关键点名称 kp_name 对应的关键点坐标和置信度信息 (x, y, conf)，如果指定的关键点名称无效，则打印错误信息并返回 None。
        except:
            print(f"get_body_kp: invalid kp_name '{kp_name}'")
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None   #最后，如果关键点的坐标 x 或 y 不为 0，则将坐标取整并返回 (int(x), int(y))；否则返回 None，表示未检测到指定的关键点

    def get_face_kp(self, kp_name="Nose_Tip", person_idx=0): #用于获取指定人脸的特定面部关键点的坐标
        """
            Return the coordinates of a keypoint named 'kp_name' of the face of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            kps = self.datum.faceKeypoints[person_idx]
        except:
            print(f"get_face_kp: invalid person_idx '{person_idx}'")
            return None
        try:
            x,y,conf = kps[face_kp_name_to_id[kp_name]]
        except:
            print(f"get_face_kp: invalid kp_name '{kp_name}'")
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None
    
    def length(self, pairs, person_idx=0, coefs = None): #这个方法的作用是计算给定人体的关键点对列表中关键点对的平均长度，可用于评估人体姿势的伸展或姿态的变化等情况
        """
            Calculate the mean of the length of the pairs in the list 'pairs' for the person of index 'person_idx' (from 0)
            If one (or both) of the 2 points of a pair is missing, the number of pairs used to calculate the average is decremented of 1
        """
        if coefs is None: #conefs用于加权计算长度的系数列表，如果为None，则将其设为与关键点对列表pairs长度为1的系数列表
            coefs = [1] * len(pairs)

        person = self.body_kps[person_idx]

        l_cum = 0
        n = 0
        for i,pair in enumerate(pairs):
            l = self.distance_kps(person[body_kp_name_to_id[pair.p1]], person[body_kp_name_to_id[pair.p2]])
            if l != 0:
                l_cum += l * coefs[i]
                n += 1
        if n>0:  #如果计数器 n 大于 0，则返回平均长度 l_cum/n；否则返回 0，表示未检测到有效的关键点对
            return l_cum/n
        else:
            return 0
    
    def check_eyes(self, person_idx=0): #用于检查指定人体的眼睛是否闭合
        """
            Check if the person whose index is 'person_idx' has his eyes closed
            Return :
            0 if both eyes are open,
            1 if only right eye is closed
            2 if only left eye is closed
            3 if both eyes are closed            
        """
        

        eye_aspect_ratio_threshold = 0.2 # 定义了一个阈值 eye_aspect_ratio_threshold，用于判断眼睛是否闭合的阈值。当眼睛的纵横比（eye aspect ratio）小于阈值时，视为眼睛闭合

        reye_closed = False
        reye1 = self.get_face_kp("REye1", person_idx=person_idx)
        reye2 = self.get_face_kp("REye2", person_idx=person_idx)
        reye3 = self.get_face_kp("REye3", person_idx=person_idx)
        reye4 = self.get_face_kp("REye4", person_idx=person_idx)
        reye5 = self.get_face_kp("REye5", person_idx=person_idx)
        reye6 = self.get_face_kp("REye6", person_idx=person_idx)
        if reye1 and reye2 and reye3 and reye4 and reye5 and reye6:
            right_eye_aspect_ratio = (self.distance(reye2, reye6)+self.distance(reye3, reye5))/(2*self.distance(reye1, reye4))
            if right_eye_aspect_ratio < eye_aspect_ratio_threshold:
                reye_closed = True
                print("RIGHT EYE CLOSED")
        
        leye_closed = False
        leye1 = self.get_face_kp("LEye1", person_idx=person_idx)
        leye2 = self.get_face_kp("LEye2", person_idx=person_idx)
        leye3 = self.get_face_kp("LEye3", person_idx=person_idx)
        leye4 = self.get_face_kp("LEye4", person_idx=person_idx)
        leye5 = self.get_face_kp("LEye5", person_idx=person_idx)
        leye6 = self.get_face_kp("LEye6", person_idx=person_idx)
        if leye1 and leye2 and leye3 and leye4 and leye5 and leye6:
            left_eye_aspect_ratio = (self.distance(leye2, leye6)+self.distance(leye3, leye5))/(2*self.distance(leye1, leye4))
            if left_eye_aspect_ratio < eye_aspect_ratio_threshold:
                leye_closed = True
                print("LEFT EYE CLOSED")
        if reye_closed:
            if leye_closed:
                return 3   #如果两只眼睛都闭合，返回 3
            else:
                return 1 #如果只有右眼闭合，返回 1
        elif leye_closed:
            return 2 #如果只有左眼闭合，返回 2
        else:
            return 0 #如果两只眼睛都没有闭合，返回 0
        
    


if __name__ == '__main__' :
   

    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--input",default="0",help="input video file (0, filename, rtsp://admin:admin@192.168.1.71/1, ...")
    ap.add_argument("-n","--number_people_max",default=-1,help="limit the number of people detected")
    ap.add_argument("-f","--face",action="store_true", help="enable face keypoint detection")
    ap.add_argument("--frt",type=float,default=0.4,help="face rendering threshold")
    ap.add_argument("-o","--output",help="path to output video file")
    ap.add_argument("-r", "--rendering",action="store_true",help="display in a separate window the original rendering made by Openpose lib")

    args=ap.parse_args()
#解析命令行参数，其中包括输入视频文件、最大检测人数、是否启用人脸关键点检测等参数

    if args.input.isdigit():
        args.input=int(args.input)
        w_h_list = [(960,720), (640, 480), (320, 240)]
        w_h_idx = 0
    
    # Read video
    video=cv2.VideoCapture(args.input)
    if isinstance(args.input, int):
        w,h = w_h_list[w_h_idx]
        video.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    ok, frame = video.read()
    h,w,_=frame.shape
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out=cv2.VideoWriter(args.output,fourcc,30,(w,h))

    my_op = OP(openpose_rendering=args.rendering, number_people_max=args.number_people_max, min_size=60, face_detection=args.face, frt=args.frt) #创建一个 OP 类的实例 my_op，用于进行姿势检测，根据命令行参数设置姿势检测的相关参数

    fps = FPS()
    while True: #进入循环，逐帧读取视频，进行姿势检测和可视化，包括绘制人体关键点和人脸关键点等信息
        # 如果按下空格键，会暂停视频播放，再次按下空格键会恢复播放
        ok, frame = video.read()
        if not ok:
            break
        fps.update()
        frame = frame.copy()
        nb_persons,body_kps,face_kps = my_op.eval(frame)
        #如果按下 "s" 键，会循环切换视频帧的宽高（在使用摄像头输入时）
        my_op.draw_body(frame)
        if args.face: 
            my_op.draw_face(frame)
            my_op.draw_eyes(frame)
            
        fps.display(frame)
        cv2.imshow("Rendering", frame)
        if args.output:
            out.write(frame)
        # 如果按下 ESC 键，或者视频读取结束，退出循环，释放资源，关闭所有窗口
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        elif k== 32: # space
            cv2.waitKey(0)     
        elif k == ord("s") and isinstance(args.input, int):
            w_h_idx = (w_h_idx+1)%len(w_h_list)
            w,h = w_h_list[w_h_idx]
            video.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


    video.release()
    cv2.destroyAllWindows()