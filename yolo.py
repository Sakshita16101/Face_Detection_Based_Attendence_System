import cv2 as cv
import numpy as np

conf_threshold = 0.25 #0.50
nms_threshold = 0.40
inpWidth = 416
inpHeight = 416


COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)



def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



def post_process(frame, outs):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes




def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    p = 10
    cv.rectangle(frame, (left-p, top-p), (right+p, bottom+p), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv.putText(frame, text, (left, top - 4), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom



classesFile= "/home/sakshita/Downloads/yolo/face.names"
classes = None



with open(classesFile, 'rt') as f:
    classes=f.read().rstrip('\n').split('\n')

modelConf ='/home/sakshita/Downloads/yolo/yolov3-face.cfg'
modelWeights='/home/sakshita/Downloads/yolo/yolov3-wider_16000.weights'

net =cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
val=input("enter the type of input")
# print(val)
# print("hello")
list=[]

if(val == '0' ):

    #print("hello")
    cap = cv.VideoCapture("/home/sakshita/Downloads/test.jpg")

    # cap = cv.VideoCapture('/home/sakshita/Downloads/test.mp4')
    count = 0

    while True:
        hasFrame, frame = cap.read()

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # net.setInput(blob)

        net.setInput(blob)

        outs = net.forward(get_outputs_names(net))

        post_process(frame, outs)
        faces = post_process(frame, outs)
        x = len(faces)
        list.append(x)
        t=4
        count = count + 1
        #print("faces present", x)
        cv.waitKey(4000)

        # print("hello")
        if (count == 1):
            break

        # id=0
        # for (x, y, w, h) in faces:
        #     cropped = frame[y: y + h, x:x + h]
        #     cv.imwrite("/home/sakshita/Desktop/images/cropped_face" + str(id) + ".jpg", cropped)
        #     id = id + 1

        # cv.imshow(winName, frame)

        # cv.destroyAllWindows()
elif(val == '1'):
    cap = cv.VideoCapture(0)

    # cap = cv.VideoCapture('/home/sakshita/Downloads/test.mp4')
    count = 0
    while True:
        hasFrame, frame = cap.read()

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # net.setInput(blob)

        net.setInput(blob)

        outs = net.forward(get_outputs_names(net))

        post_process(frame, outs)
        faces = post_process(frame, outs)
        x = len(faces)
        list.append(x)
        count = count + 1
        # print("faces present", x)
        cv.waitKey(4000)
        t=40

        # print("hello")
        if (count == 10):
            break

        # id=0
        # for (x, y, w, h) in faces:
        #     cropped = frame[y: y + h, x:x + h]
        #     cv.imwrite("/home/sakshita/Desktop/images/cropped_face" + str(id) + ".jpg", cropped)
        #     id = id + 1

        # cv.imshow(winName, frame)

        # cv.destroyAllWindows()
for i in list:
    print(i)
print(t)
print(count)
# winName='face detetcion with yolo'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 1000, 1000)
#/home/sakshita/Downloads/1771926_2019-06-17 13_49_37.jpg
