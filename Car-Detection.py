from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']

# # For Webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# For Video
cap = cv2.VideoCapture("cars3.mp4")

model = YOLO("yolov8n.pt")

mask = cv2.imread("mask-cars3.png")

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

limits = [100, 1250, 900, 1250]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)

            # x1, y1, w, h = box.xywh[0]
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            # cvzone.cornerRect(img, bbox, l=8, rt=5)
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)
            # cvzone.putTextRect(img, f'{conf}',(max(0,x1),max(35,y1-20)))
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1-20)),
                                   scale=1, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]),
             (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2-x1, y2-y1
        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1-20)),
                           scale=2, thickness=3, offset=10)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        print(cx, cy)
        cv2.circle(img, (cx, cy), 5, (255, 1, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]),
                         (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100),
                cv2.FONT_HERSHEY_PLAIN, 5, (50, 255, 255), 8)

    # cv2.namedWindow("Image",0)
    # cv2.resizeWindow("Image", 1920,1080)
    scale = 0.7
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Image", resized)

    # cv2.namedWindow("ImageRegion",0)
    # cv2.resizeWindow("ImageRegion", 720,1280)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
