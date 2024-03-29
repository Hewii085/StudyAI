import os
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.nn import YOLO as ConvNet
from learning.evaluators import RecallEvaluator as Evaluator
from learning.utils import draw_pred_boxes, predict_nms_boxes, convert_boxes
import cv2
import glob

""" 1. Load dataset """
root_dir = os.path.join('.\\data\\face')
test_dir = os.path.join(root_dir, 'test')
IM_SIZE = (416, 416)
NUM_CLASSES = 1

# Load test set
X_test, y_test = dataset.read_data(test_dir, IM_SIZE)
test_set = dataset.DataSet(X_test, y_test)

""" 2. Set test hyperparameters """
anchors = dataset.load_json(os.path.join(test_dir, 'anchors.json'))
class_map = dataset.load_json(os.path.join(test_dir, 'classes.json'))
nms_flag = True
hp_d = dict()
hp_d['batch_size'] = 16
hp_d['nms_flag'] = nms_flag

""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, './ckpt/model.ckpt')
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test performance: {}'.format(test_score))

""" 4. Draw boxes on image """
draw_dir = os.path.join('.\\draws') # FIXME
im_dir = os.path.join(test_dir, 'images') # FIXME
im_paths = []
im_paths.extend(glob.glob(os.path.join(im_dir, '*.jpg')))
for idx, (img, y_pred, im_path) in enumerate(zip(test_set.images, test_y_pred, im_paths)):
    name = im_path.split('\\')[-1]
    draw_path =os.path.join(draw_dir, name)
    if nms_flag:
        bboxes = predict_nms_boxes(y_pred, conf_thres=0.5, iou_thres=0.5)
    else:
        bboxes = convert_boxes(y_pred)
    bboxes = bboxes[np.nonzero(np.any(bboxes > 0, axis=1))]
    boxed_img = draw_pred_boxes(img, bboxes, class_map)
    cv2.imwrite(draw_path, boxed_img)
    
