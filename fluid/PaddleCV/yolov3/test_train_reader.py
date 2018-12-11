import numpy as np
import box_utils
import reader
import cv2


category_names = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
train_reader = reader.train(416, 1, shuffle=True)
for data in train_reader():
    img, gtboxes, gtlabels = data[0]
    c, h, w = img.shape
    real_img = (img.transpose((1, 2, 0)) * stds + means) * 255.0
    cv2.imwrite("output/test.jpg", real_img.astype("uint8"))
    gtboxes = box_utils.box_xywh_to_xyxy(gtboxes)
    gtboxes = box_utils.rescale_box_in_input_image(gtboxes, (h, w), 1.0)
    scores = np.ones_like(gtlabels)
    box_utils.draw_boxes_on_image("output/test.jpg", gtboxes, scores, gtlabels, category_names)
    break

