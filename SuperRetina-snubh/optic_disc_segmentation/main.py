import os
import glob
from datetime import datetime
#import warnings
#warnings.simplefilter('ignore')
# import scipy as sp
# import scipy.ndimage
import numpy as np
# import pandas as pd
import tensorflow as tf
import skimage
import skimage.exposure
# import imutils
# from imutils import contours
from skimage import measure
# import mahotas as mh
# from sklearn.model_selection import KFold
from PIL import Image
import cv2
import sys
import h5py
sys.path.append("./code/SuperRetina-snubh/optic_disc_segmentation/")
from dual_IDG import DualImageDataGenerator
from model import get_unet_light

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Conv2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, \
    Lambda, UpSampling2D, Cropping2D, Concatenate
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

def tf_to_th_encoding(X):
    return np.rollaxis(X, 3, 1)

def get_contour(image) :
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(image,9,75,75)
    median=cv2.medianBlur(blur,5)
    thresh = cv2.threshold(median, 155, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue


        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels >300:
            mask = cv2.add(mask, labelMask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    print(cnts)
    for (i, c) in enumerate(cnts):
        ellipse = cv2.fitEllipse(c)
        (x, y, w, h) = cv2.boundingRect(c)
        return w, h
    

def find_long_short_axis(mask, center, image):
    x_line = np.arange(0, mask.shape[1])
    y_line = np.arange(0, mask.shape[0])
    x, y = np.meshgrid(x_line, y_line)
    x = (x - center[1])**2
    y = (y - center[0])**2
    x = np.bitwise_and(x,mask) * x
    y = np.bitwise_and(y,mask) * y
    cnts = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    print(cnts)
    for (i, c) in enumerate(cnts):
        #(x,y), (major_axis, minor_axis), angle
        ellipse = cv2.fitEllipse(c)
        rect_image = cv2.ellipse(image, ellipse, (0,255,0), 2)
        # cv2.imwrite("tmp.png", rect_image)
        (x, y, w, h) = cv2.boundingRect(c)
        return w, h, rect_image
    # return x, y, rect_image


def persepctive_matrix_rect(wide_info, fp_info, wide_center, fp_center, wide_image):
    wide_w,wide_h = wide_info
    fp_w, fp_h = fp_info
    wide_cen_y, wide_cen_x = wide_center
    fp_cen_y, fp_cen_x = fp_center
    
    dst = np.array([[fp_cen_x - int(fp_w/2.), fp_cen_y - int(fp_h/2.)], [fp_cen_x - int(fp_w/2.), fp_cen_y + int(fp_h/2.)],
                    [fp_cen_x + int(fp_w/2.), fp_cen_y - int(fp_h/2.)], [fp_cen_x + int(fp_w/2.), fp_cen_y + int(fp_h/2.)]], np.float32)
    src = np.array([[wide_cen_x - int(wide_w/2.), wide_cen_y - int(wide_h/2.)], [wide_cen_x - int(wide_w/2.), wide_cen_y + int(wide_h/2.)],
                    [wide_cen_x + int(wide_w/2.), wide_cen_y - int(wide_h/2.)], [wide_cen_x + int(wide_w/2.), wide_cen_y + int(wide_h/2.)]], np.float32)
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    trans_wide_image = cv2.warpPerspective(wide_image, matrix, (0,0))
    return trans_wide_image



clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))

def find_disc_segmentation(model, image):
    image = np.array(image.resize((256,256)))
    if image.shape.__len__() < 3:
        image = image[:,:,np.newaxis]
    
    #make input for disc segmentation
    clahe_image = image.copy()
    clahe_image[:,:,0] = clahe.apply(image[:,:,0])
    clahe_image[:,:,1] = clahe.apply(image[:,:,1])
    clahe_image[:,:,2] = clahe.apply(image[:,:,2])
    clahe_image_input = clahe_image.copy().transpose((2,0,1))[np.newaxis,:,:,:]/255.
    image_disc = (model.predict(clahe_image_input)[0][0]*255).astype(np.uint8) * 255
    return image_disc

def find_disc_center(image):
    _, fp_label, fp_stats, _ = cv2.connectedComponentsWithStats(image)
    fp_max_arg = np.argmax(fp_stats[1:,-1]) + 1
    fp_disc_mask = (fp_label == fp_max_arg)
    x,y,w,h,_ = fp_stats[fp_max_arg].copy()*3
    fp_disc_center = (y+int(h/2.), x+int(w/2.))
    return fp_disc_center


def disc_matching(fp_path, widefp_path, wide_vessel_path, model):
    fp = Image.open(fp_path)
    fp_disc_seg = find_disc_segmentation(model, fp)
    fp_center = find_disc_center(fp_disc_seg)

    wide_fp = Image.open(widefp_path)
    wide_vessel = Image.open(wide_vessel_path)
    widefp_disc_seg = find_disc_segmentation(model, wide_fp)
    widefp_center = find_disc_center(widefp_disc_seg)
    wide_fp_moving = persepctive_matrix_rect((80, 80), (105, 127), widefp_center, fp_center, np.array(wide_fp))
    wide_vessel_moving = persepctive_matrix_rect((80, 80), (105, 127), widefp_center, fp_center, np.array(wide_vessel))
    return wide_fp_moving, wide_vessel_moving

def wide_fp_load():
    image_path = "./data/Lab/ImageData/"
    vessel_path = "./data/Lab_Ves/ImageData/"
    image_dict = {}
    vessel_dict = {}
    image_list = sorted(glob.glob(image_path + "*.png"))
    for i in image_list:
        key = i.split("/")[-1].split("_")[-1]
        if key in image_dict:
            image_dict[key].append(i)
            vessel_dict[key].append(vessel_path + i.split("/")[-1])
        else:
            image_dict[key] = [i]
            vessel_dict[key] = [vessel_path + i.split("/")[-1]]
    return image_path, image_dict, vessel_dict

def main():
    print('Keras version:', keras.__version__)
    print('TensorFlow version:', tf.__version__)
    K.set_image_data_format('channels_first')
    model = get_unet_light(img_rows=256, img_cols=256)
    # h5f = h5py.File(os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets', 'DRIONS_DB.hdf5'), 'r')
    model.load_weights("./code/SuperRetina-snubh/optic_disc_segmentation/last_checkpoint.hdf5")
    image_path, image_dict, vessel_dict = wide_fp_load()
    matched_image_path = "./data/Lab/wide_disc_matching/"
    os.makedirs(matched_image_path, exist_ok = True)
    matched_gray_path = "./data/Lab_Ves/wide_disc_matching/"
    os.makedirs(matched_gray_path, exist_ok = True)
    for i in image_dict.keys():
        if image_dict.__len__() < 2:
            continue
        #f2 fundus, f1 wide fundus
        f2 = image_dict[i][0]
        f1 = image_dict[i][-1]

        vessel_f2 = vessel_dict[i][0]
        vessel_f1 = vessel_dict[i][-1]
        mapped_widefp, mapped_wide_gray = disc_matching(f2, f1, vessel_f1, model)
        mapped_widefp = cv2.cvtColor(mapped_widefp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(matched_image_path + image_dict[i][-1].split("/")[-1], mapped_widefp)
        cv2.imwrite(matched_gray_path + image_dict[i][-1].split("/")[-1], mapped_wide_gray)

def tmp_code():
    image_path = './code/SuperRetina-snubh/optic_disc_segmentation/vistel_f_000_color.png'
    wide_image_path = './code/SuperRetina-snubh/optic_disc_segmentation/vistel_w_000_color.png'
    image_gray_path = './code/SuperRetina-snubh/optic_disc_segmentation/vistel_f_000_gray.png'
    wide_image_gray_path = './code/SuperRetina-snubh/optic_disc_segmentation/vistel_w_000_gray.png'
    # image_gray = np.array(Image.open(image_gray_path).convert('L'))
    wide_image_gray = np.array(Image.open(wide_image_gray_path).convert('L'))
    wide_image_gray_contour = get_contour(wide_image_gray)
    image = np.array(Image.open(image_path).resize((256,256)))
    image_input = image.copy().transpose((2,0,1))[np.newaxis,:,:,:]/255.
    wide_image_org = np.array(Image.open(wide_image_path))
    wide_image = np.array(Image.open(wide_image_path).resize((256,256)))
    wide_image_input = wide_image.copy().transpose((2,0,1))[np.newaxis,:,:,:]/255.
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    clahe_image = image.copy()
    clahe_image[:,:,0] = clahe.apply(image[:,:,0])
    clahe_image[:,:,1] = clahe.apply(image[:,:,1])
    clahe_image[:,:,2] = clahe.apply(image[:,:,2])
    clahe_image_input = clahe_image.copy().transpose((2,0,1))[np.newaxis,:,:,:]/255.
    clahe_wide_image = wide_image.copy()
    clahe_wide_image[:,:,0] = clahe.apply(wide_image[:,:,0])
    clahe_wide_image[:,:,1] = clahe.apply(wide_image[:,:,1])
    clahe_wide_image[:,:,2] = clahe.apply(wide_image[:,:,2])
    clahe_wide_image_input = clahe_wide_image.copy().transpose((2,0,1))[np.newaxis,:,:,:]/255.
    image_seg = (model.predict(image_input) > 0.01).astype(np.uint8)*255
    wide_image_seg = (model.predict(wide_image_input) > 0.1).astype(np.uint8)*255
    # clahe_image_seg = (model.predict(clahe_image_input)[0][0]*255).astype(np.uint8)
    # clahe_wide_seg = (model.predict(clahe_wide_image_input)[0][0]*255).astype(np.uint8)
    clahe_image_seg = (model.predict(clahe_image_input) > 0.49).astype(np.uint8)*255
    clahe_wide_seg = (model.predict(clahe_wide_image_input) > 0.49).astype(np.uint8)*255
    # cv2.imwrite("image_seg.jpg", image_seg[0][0])
    # cv2.imwrite("wide_image_seg.jpg", wide_image_seg[0][0])
    # cv2.imwrite("clahe_image_seg.jpg", clahe_image_seg[0][0])
    # cv2.imwrite("clahe_wide_seg.jpg", clahe_wide_seg[0][0])

    _, fp_label, fp_stats, _ = cv2.connectedComponentsWithStats(clahe_image_seg[0][0])
    _, wide_label, wide_stats, _ = cv2.connectedComponentsWithStats(clahe_wide_seg[0][0])

    fp_max_arg = np.argmax(fp_stats[1:,-1]) + 1
    wide_max_arg = np.argmax(wide_stats[1:,-1]) + 1

    fp_disc_mask = (fp_label == fp_max_arg)
    x,y,w,h,_ = fp_stats[fp_max_arg].copy()*3
    fp_disc_center = (y+int(h/2.), x+int(w/2.))

    wide_disc_mask = (wide_label == wide_max_arg)
    x,y,w,h,_ = wide_stats[wide_max_arg].copy()*3
    wide_disc_center = (y+int(h/2.), x+int(w/2.))

    # fp_disc_w, fp_disc_h, vis_image = find_long_short_axis(fp_disc_mask, fp_disc_center, image.copy())
    # cv2.imwrite("fp_ellipse.jpg", vis_image)
    # wide_disc_w, wide_disc_h, vis_wide_image = find_long_short_axis(wide_disc_mask, wide_disc_center, wide_image.copy())
    # cv2.imwrite("wide_ellipse.jpg", vis_wide_image)

    # result = persepctive_matrix_rect((wide_disc_w, wide_disc_h), (fp_disc_w, fp_disc_h), wide_disc_center, fp_disc_center, wide_image_gray)
    result = persepctive_matrix_rect((80, 80), (105, 127), wide_disc_center, fp_disc_center, wide_image_org)
    Image.fromarray(result).save("perspective_result.jpg")

    Image.fromarray(wide_image_seg[0][0]).save("vistel_w_000_seg.png")
    Image.fromarray(image_seg[0][0]).save("vistel_f_000_seg.png")

    output = model.predict(image)


main()
