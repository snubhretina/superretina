import configparser
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.registration_BSpline import regist_BSpline
from torchvision import transforms
sys.path.append("./code/SuperRetina-snubh")

from common.common_util import pre_processing, simple_nms, remove_borders, \
    sample_keypoint_desc
from model.super_retina import SuperRetina

from PIL import Image
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "6"
class Predictor:
    def __init__(self, config):

        predict_config = config['PREDICT']

        device = predict_config['device']
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        model_save_path = predict_config['model_save_path']
        self.nms_size = predict_config['nms_size']
        self.nms_thresh = predict_config['nms_thresh']
        self.scale = 8
        self.knn_thresh = predict_config['knn_thresh']

        self.image_width = 768
        self.image_height = 768

        self.matched_keypoint = 0
        self.query_keypoint = 0

        self.keypoint_option = 'superretina'

        self.model_image_width = predict_config['model_image_width']
        self.model_image_height = predict_config['model_image_height']

        checkpoint = torch.load(model_save_path, map_location=device)
        model = SuperRetina()
        model.load_state_dict(checkpoint['net'])
        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.knn_matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.trasformer = transforms.Compose([
            transforms.Resize((self.model_image_height, self.model_image_width)),
            transforms.ToTensor(),

        ])

    def image_read(self, query_path, refer_path, query_is_image=False):
        if query_is_image:
            query_image = query_path
        else:
            query_image = cv2.imread(query_path, cv2.IMREAD_COLOR)
            # green channel
            query_image = query_image[:, :, 1]
            query_image = cv2.resize(query_image, [self.image_width, self.image_height])
            query_image = pre_processing(query_image)
            
        refer_image = cv2.imread(refer_path, cv2.IMREAD_COLOR)
        refer_image = cv2.resize(refer_image, [self.image_width, self.image_height])

        assert query_image.shape[:2] == refer_image.shape[:2]
        self.image_height, self.image_width = query_image.shape[:2]

        refer_image = refer_image[:, :, 1]
        refer_image = pre_processing(refer_image)

        query_image = (query_image * 255).astype(np.uint8)
        refer_image = (refer_image * 255).astype(np.uint8)

        return query_image, refer_image

    def draw_result(self, query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, status):
        def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
            # initialize the output visualization image
            (hA, wA) = imageA.shape[:2]
            (hB, wB) = imageB.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            if len(imageA.shape) == 2:
                imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2RGB)
                imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2RGB)

            vis[0:hA, 0:wA] = imageA
            vis[0:hB, wA:] = imageB

            # loop over the matches
            for (match, _), s in zip(matches, status):
                trainIdx, queryIdx = match.trainIdx, match.queryIdx
                # only process the match if the keypoint was successfully
                # matched
                if s == 1:
                    # draw the match
                    ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                    ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 2)

                # return the visualization
            return vis

        query_np = np.array([kp.pt for kp in cv_kpts_query])
        refer_np = np.array([kp.pt for kp in cv_kpts_refer])
        refer_np[:, 0] += query_image.shape[1]
        matched_image = drawMatches(query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, status)
        # cv2.imwrite("matched_image_siftfeat_{}.png".format(threshold), matched_image)
        cv2.imwrite("matched_image.png", matched_image)
        # plt.figure(dpi=300)
        # plt.scatter(query_np[:, 0], query_np[:, 1], s=1, c='r')
        # plt.scatter(refer_np[:, 0], refer_np[:, 1], s=1, c='r')
        # plt.axis('off')
        # plt.title('Match Result, #goodMatch: {}'.format(status.sum()))
        # plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        # plt.show()
        # plt.close()

    def original_run_pair(self, query_image, refer_image,method='SIFT'):
        if method == 'SIFT':
            sift = cv2.SIFT_create()
            query_keypt, query_des = sift.detectAndCompute(query_image, None)
            refer_keypt, refer_des = sift.detectAndCompute(refer_image, None)
        return [query_keypt, refer_keypt], [query_des, refer_des]
    
    def keypoint_visualizing(self, image, keypoints, descriptors):
        image
        return 0
        
        

    def model_run_pair(self, query_tensor, refer_tensor):
        inputs = torch.cat((query_tensor.unsqueeze(0), refer_tensor.unsqueeze(0)))
        inputs = inputs.to(self.device)

        with torch.no_grad():
            detector_pred, descriptor_pred = self.model(inputs)

        scores = simple_nms(detector_pred, self.nms_size)

        b, _, h, w = detector_pred.shape
        scores = scores.reshape(-1, h, w)

        keypoints = [
            torch.nonzero(s > self.nms_thresh)
            for s in scores]

        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, 4, h, w)
            for k, s in zip(keypoints, scores)]))

        keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

        descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu()
                       for k, d in zip(keypoints, descriptor_pred)]
        keypoints = [k.cpu() for k in keypoints]
        return keypoints, descriptors

    def match(self, query_path, refer_path, show=False, query_is_image=False):
        query_image, refer_image = self.image_read(query_path, refer_path, query_is_image)
        query_tensor = self.trasformer(Image.fromarray(query_image))
        refer_tensor = self.trasformer(Image.fromarray(refer_image))
        if self.keypoint_option == 'superretina':
            keypoints, descriptors = self.model_run_pair(query_tensor, refer_tensor)
            query_keypoints, refer_keypoints = keypoints[0], keypoints[1]
            query_desc, refer_desc = descriptors[0].permute(1, 0).numpy(), descriptors[1].permute(1, 0).numpy()
            # mapping keypoints to scaled keypoints
            cv_kpts_query = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                        int(i[1] / self.model_image_height * self.image_height), 30)
                            for i in query_keypoints]
            cv_kpts_refer = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                        int(i[1] / self.model_image_height * self.image_height), 30)
                            for i in refer_keypoints]
        elif self.keypoint_option == 'sift':
            keypoints, descriptors = self.original_run_pair(query_image, refer_image)
            cv_kpts_query, cv_kpts_refer = keypoints
            query_desc, refer_desc = descriptors
        # query_image_key = np.zeros(query_image.shape)
        # refer_image_key = np.zeros(refer_image.shape)
        # query_image_key = cv2.drawKeypoints(query_image, cv_kpts_query, query_image_key)
        # refer_image_key = cv2.drawKeypoints(refer_image, cv_kpts_refer, refer_image_key)
        # cv2.drawKeypoints(query_image, cv_kpts_query, refer_image_key)
        # debug
        # cv2.imwrite("0_query_image_key.jpg", query_image_key)
        # cv2.imwrite("0_refer_image_key.jpg", refer_image_key)
        
        # keypoints, descriptors = self.original_run_pair(query_image, refer_image)
        # cv_kpts_query, cv_kpts_refer = keypoints
        # query_desc, refer_desc = descriptors

        
        # for i in range(1,10):
        goodMatch = []
        status = []
        matches = []
        
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < (self.knn_thresh) * n.distance:
                    goodMatch.append(m)
                    status.append(True)
                else:
                    status.append(False)
        except Exception:
            pass

        if show:
            print(self.knn_thresh, np.array(status).sum(), status.__len__())
            self.draw_result(query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, np.array(status))
        return goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image

    def compute_homography(self, query_path, refer_path, query_is_image=False):
        goodMatch, cv_kpts_query, cv_kpts_refer, raw_query_image, raw_refer_image = \
            self.match(query_path, refer_path, query_is_image=query_is_image)
        H_m = None
        inliers_num_rate = 0

        if len(goodMatch) >= 4:
            src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

            H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

            # src_pts = src_pts[mask.ravel() == 1]
            # dst_pts = dst_pts[mask.ravel() == 1]

            goodMatch = np.array(goodMatch)[mask.ravel() == 1]
            inliers_num_rate = mask.sum() / len(mask.ravel())

        return H_m, inliers_num_rate, raw_query_image, raw_refer_image

    def align_image_pair(self, query_path, refer_path, show=False):
        H_m, inliers_num_rate, raw_query_image, raw_refer_image = self.compute_homography(query_path, refer_path)

        if H_m is not None:
            h, w = self.image_height, self.image_width
            query_align = cv2.warpPerspective(raw_query_image, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0))

            merged = np.zeros((h, w, 3), dtype=np.uint8)

            if len(query_align.shape) == 3:
                query_align = cv2.cvtColor(query_align, cv2.COLOR_BGR2GRAY)
            if len(raw_refer_image.shape) == 3:
                refer_gray = cv2.cvtColor(raw_refer_image, cv2.COLOR_BGR2GRAY)
            else:
                refer_gray = raw_refer_image
            # debug
            # cv2.imwrite("query_align.jpg", query_align)
            # cv2.imwrite("refer_align.jpg", refer_gray)
            merged[:, :, 0] = query_align
            merged[:, :, 1] = refer_gray

            if show:
                plt.figure(dpi=200)
                plt.imshow(merged)
                plt.axis('off')
                plt.title('Registration Result')
                plt.show()
                plt.close()
            return merged

        print("Matched Failed!")
        return None, None
    
    def align_image_pair_vessel(self, query_path, refer_path, query_vessel_path, refer_vessel_path, show=False):
        query_vessel = cv2.imread(query_vessel_path)
        refer_vessel = cv2.imread(refer_vessel_path)
        query_vessel = cv2.resize(query_vessel, [self.image_width, self.image_height])
        refer_vessel = cv2.resize(refer_vessel, [self.image_width, self.image_height])
        query_align_fundus = cv2.imread(query_path)
        # query_align_fundus = cv2.resize(query_align_fundus, [self.image_width, self.image_height])
        H_m, inliers_num_rate, raw_query_image, raw_refer_image = self.compute_homography(query_path, refer_path)

        if H_m is not None:
            h, w = self.image_height, self.image_width
            query_align_vessel = cv2.warpPerspective(query_vessel, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0))
            query_align_fundus = cv2.warpPerspective(raw_query_image, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0))

            merged = np.zeros((h, w, 3), dtype=np.uint8)
            merged_bspline = np.zeros((h, w, 3), dtype=np.uint8)

            if len(query_align_vessel.shape) == 3:
                query_align_vessel = cv2.cvtColor(query_align_vessel, cv2.COLOR_BGR2GRAY)
            if len(refer_vessel.shape) == 3:
                refer_vessel = cv2.cvtColor(refer_vessel, cv2.COLOR_BGR2GRAY)
            else:
                refer_vessel = refer_vessel
            cv2.imwrite("query_align_vessel.jpg", query_align_vessel)
            cv2.imwrite("refer_align.jpg", refer_vessel)
            merged[:, :, 0] = query_align_vessel
            merged[:, :, 1] = refer_vessel
            merged_bspline[:, :, 1] = refer_vessel

            if show:
                plt.figure(dpi=200)
                plt.imshow(merged)
                plt.axis('off')
                plt.title('Registration Result')
                plt.show()
                plt.close()
            bsp = regist_BSpline(refer_vessel, query_align_vessel)
            query_align_vessel = bsp.do_registration()
            merged_bspline[:, :, 0] = query_align_vessel.astype(np.uint8)
            query_align_fundus = bsp.registrationFromMatrix(query_align_fundus).astype(np.uint8)
            return merged, merged_bspline, query_align_fundus

        print("Matched Failed!")
        return None, None, None

    def model_run_one_image(self, image_path, save_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image[:, :, 1]
        self.image_height, self.image_width = image.shape[:2]

        image = pre_processing(image)
        image_tensor = self.trasformer(Image.fromarray(image))
        inputs = image_tensor.unsqueeze(0)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            detector_pred, descriptor_pred = self.model(inputs)

        scores = simple_nms(detector_pred, self.nms_size)

        b, _, h, w = detector_pred.shape
        scores = scores.reshape(-1, h, w)

        keypoints = [
            torch.nonzero(s > self.nms_thresh)
            for s in scores]

        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, 4, h, w)
            for k, s in zip(keypoints, scores)]))

        keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

        descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu()
                       for k, d in zip(keypoints, descriptor_pred)]
        keypoints = [k.cpu() for k in keypoints]

        if save_path is not None:
            save_info = {'kp': keypoints[0].cpu(), 'desc': descriptors[0].cpu()}
            torch.save(save_info, save_path)

        return keypoints[0], descriptors[0]

    def homography_from_tensor(self, query_info, refer_info):
        query_keypoints, query_desc = query_info['kp'], query_info['desc']
        refer_keypoints, refer_desc = refer_info['kp'], refer_info['desc']

        query_desc = query_desc.permute(1, 0).numpy()
        refer_desc = refer_desc.permute(1, 0).numpy()
        cv_kpts_query = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in query_keypoints]
        cv_kpts_refer = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in refer_keypoints]

        goodMatch = []
        status = []
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    goodMatch.append(m)
                    status.append(True)
                else:
                    status.append(False)
        except Exception:
            pass

        H_m = None
        inliers_num = 0

        if len(goodMatch) >= 4:
            src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

            H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

            # src_pts = src_pts[mask.ravel() == 1]
            # dst_pts = dst_pts[mask.ravel() == 1]

            goodMatch = np.array(goodMatch)[mask.ravel() == 1]
            inliers_num = mask.sum()

        return H_m, inliers_num

import glob
def fire_load():
    fire_path = "./data/FIRE/Images/"
    fire_dict = {}
    fire_list = glob.glob(fire_path + "*.jpg")
    for i in fire_list:
        name_identity = i.split("/")[-1].split("_")[0]
        if name_identity in fire_dict:
            fire_dict[name_identity].append(i)
        else:
            fire_dict[name_identity] = [i]
    return fire_path, fire_dict

def jeju_load():
    jeju_path = "./data/jeju_vessel/"
    jeju_dict = {}
    vessel_dict = {}
    jeju_list = sorted(glob.glob(jeju_path + "*.png"))
    for i in jeju_list:
        if i.find("pred") > -1:
            continue
        pid, date, laterality = i.split("/")[-1].split("_")[:3]
        key = "{}_{}_{}".format(pid, date, laterality[:2])
        if key in jeju_dict:
            jeju_dict[key].append(i)
            vessel_dict[key].append(i[:-11] + "_pred.png")
        else:
            jeju_dict[key] = [i]
            vessel_dict[key] = [i[:-11] + "_pred.png"]
    return jeju_path, jeju_dict, vessel_dict

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


if __name__ == '__main__':
    import yaml

    config_path = './code/SuperRetina-snubh/config/test_SNUBH.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config File doesn't Exist")

    P = Predictor(config)
    # vessel_f2 = './code/SuperRetina-snubh/vistel_f_000_vessel.png'
    # vessel_f1 = './code/SuperRetina-snubh/vistel_w_000_vessel.png'
    # f2 = './code/SuperRetina-snubh_vessel/disc_center.jpg'
    # f1 = './code/SuperRetina-snubh_vessel/fovea_center.jpg'

    # P.match(f1, f2, show=True)
    # merged = P.align_image_pair(f1, f2)
    # # merged = P.align_image_pair_vessel(f1, f2, vessel_f1, vessel_f2)
    # plt.imsave('./code/SuperRetina-snubh/fundus-wide-preproc.png', merged)

    dataset = 'widefp'
    os.makedirs('./experiment/processing/{}/'.format(dataset), exist_ok = True)
    image_path, image_dict, vessel_dict = wide_fp_load()
    for i in image_dict.keys():
        if image_dict.__len__() < 2:
            continue

        f2 = image_dict[i][0]
        f1 = image_dict[i][-1]

        vessel_f2 = vessel_dict[i][0]
        vessel_f1 = vessel_dict[i][-1]

        # P.match(f1, f2, show=True, keypoint_option = 'superretina')
        P.keypoint_option = 'superretina'
        superretina_merged = P.align_image_pair(f1, f2)
        superretina_vessel_merged, supreretina_merged_bspline, supreretina_align_fundus = P.align_image_pair_vessel(f1, f2, vessel_f1, vessel_f2)
        # P.match(f1, f2, show=True, keypoint_option = 'sift')
        P.keypoint_option = 'sift'
        sift_merged = P.align_image_pair(f1, f2)
        sift_vessel_merged, sift_merged_bspline, sift_align_fundus = P.align_image_pair_vessel(f1, f2, vessel_f1, vessel_f2)
        # if merged == None:
        #     continue
        # merged = P.align_image_pair_vessel(f1, f2, vessel_f1, vessel_f2)
        # plt.imsave('./experiment/processing/jeju/{}_superretina.jpg'.format(i), merged)
        try:
            plt.imsave('./experiment/processing/{}/{}_superretina.jpg'.format(dataset,i), superretina_merged)
            plt.imsave('./experiment/processing/{}/{}_sift.jpg'.format(dataset,i), sift_merged)
            plt.imsave('./experiment/processing/{}/{}_superretina_vessel.jpg'.format(dataset,i), superretina_vessel_merged)
            plt.imsave('./experiment/processing/{}/{}_sift_vessel.jpg'.format(dataset,i), sift_vessel_merged)
            plt.imsave('./experiment/processing/{}/{}_superretina_bsplined_vessel.jpg'.format(dataset,i), supreretina_merged_bspline)
            plt.imsave('./experiment/processing/{}/{}_sift_bsplined_vessel.jpg'.format(dataset,i), sift_merged_bspline)
            plt.imsave('./experiment/processing/{}/{}_superretina_bsplined_fundus.jpg'.format(dataset,i), supreretina_align_fundus)
            plt.imsave('./experiment/processing/{}/{}_sift_bsplined_fundus.jpg'.format(dataset,i), sift_align_fundus)
        except:
            continue
    # plt.imshow(merged)
    plt.show()
