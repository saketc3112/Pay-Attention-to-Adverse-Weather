import tensorflow as tf
import numpy as np
import os
import PIL.Image
from PIL import Image
import json
from pyquaternion import Quaternion
import cv2

all_classes = []

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def read_label(file, label_dir):
    """Read label file and return object list"""
    file_name = file.split('.png')[0]
    print(file_name)
    object_list = get_kitti_object_list(os.path.join(label_dir, file_name + '.txt'), camera_to_velodyne=camera_to_velodyne)
    return object_list


def load_calib_data(path_total_dataset, name_camera_calib, tf_tree):
    """
    :param path_total_dataset: Path to dataset root dir
    :param name_camera_calib: Camera calib file containing image intrinsic
    :param tf_tree: TF (transformation) tree containing translations from velodyne to cameras
    :return:
    """

    with open(os.path.join(path_total_dataset, name_camera_calib), 'r') as f:
        data_camera = json.load(f)

    with open(os.path.join(path_total_dataset, tf_tree), 'r') as f:
        data_extrinsics = json.load(f)

    # Scan data extrinsics for transformation from lidar to camera
    important_translations = ['lidar_hdl64_s3_roof', 'radar_ars300', 'cam_stereo_left_optical']
    translations = []

    for item in data_extrinsics:
        if item['child_frame_id'] in important_translations:
            translations.append(item)
            if item['child_frame_id'] == 'cam_stereo_left_optical':
                T_cam = item['transform']
            elif item['child_frame_id'] == 'lidar_hdl64_s3_roof':
                T_velodyne = item['transform']
            elif item['child_frame_id'] == 'radar_ars300':
                T_radar = item['transform']

    # Use pyquaternion to setup rotation matrices properly
    R_c_quaternion = Quaternion(w=T_cam['rotation']['w'] * 360 / 2 / np.pi, x=T_cam['rotation']['x'] * 360 / 2 / np.pi,
                     y=T_cam['rotation']['y'] * 360 / 2 / np.pi, z=T_cam['rotation']['z'] * 360 / 2 / np.pi)
    R_v_quaternion = Quaternion(w=T_velodyne['rotation']['w'] * 360 / 2 / np.pi, x=T_velodyne['rotation']['x'] * 360 / 2 / np.pi,
                     y=T_velodyne['rotation']['y'] * 360 / 2 / np.pi, z=T_velodyne['rotation']['z'] * 360 / 2 / np.pi)

    # Setup quaternion values as 3x3 orthogonal rotation matrices
    R_c_matrix = R_c_quaternion.rotation_matrix
    R_v_matrix = R_v_quaternion.rotation_matrix

    # Setup translation Vectors
    Tr_cam = np.asarray([T_cam['translation']['x'], T_cam['translation']['y'], T_cam['translation']['z']])
    Tr_velodyne = np.asarray([T_velodyne['translation']['x'], T_velodyne['translation']['y'], T_velodyne['translation']['z']])
    Tr_radar = np.asarray([T_radar['translation']['x'], T_radar['translation']['y'], T_radar['translation']['z']])

    # Setup Translation Matrix camera to lidar -> ROS spans transformation from its children to its parents
    # Therefore one inversion step is needed for zero_to_camera -> <parent_child>
    zero_to_camera = np.zeros((3, 4))
    zero_to_camera[0:3, 0:3] = R_c_matrix
    zero_to_camera[0:3, 3] = Tr_cam
    zero_to_camera = np.vstack((zero_to_camera, np.array([0, 0, 0, 1])))

    zero_to_velodyne = np.zeros((3, 4))
    zero_to_velodyne[0:3, 0:3] = R_v_matrix
    zero_to_velodyne[0:3, 3] = Tr_velodyne
    zero_to_velodyne = np.vstack((zero_to_velodyne, np.array([0, 0, 0, 1])))

    zero_to_radar = zero_to_velodyne.copy()
    zero_to_radar[0:3, 3] = Tr_radar

    # Calculate total extrinsic transformation to camera
    velodyne_to_camera = np.matmul(np.linalg.inv(zero_to_camera), zero_to_velodyne)
    camera_to_velodyne = np.matmul(np.linalg.inv(zero_to_velodyne), zero_to_camera)
    radar_to_camera = np.matmul(np.linalg.inv(zero_to_camera), zero_to_radar)

    # Read projection matrix P and camera rectification matrix R
    P = np.reshape(data_camera['P'], [3, 4])

    # In our case rectification matrix R has to be equal to the identity as the projection matrix P contains the
    # R matrix w.r.t KITTI definition
    R = np.identity(4)

    # Calculate total transformation matrix from velodyne to camera
    vtc = np.matmul(np.matmul(P, R), velodyne_to_camera)

    return velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera


def project_3d_to_2d(points3d, P):
    points2d = np.matmul(P, np.vstack((points3d, np.ones([1, np.shape(points3d)[1]]))))

    # scale projected points
    points2d[0][:] = points2d[0][:] / points2d[2][:]
    points2d[1][:] = points2d[1][:] / points2d[2][:]

    points2d = points2d[0:2]
    return points2d.transpose()

def project_points_to_2d(points3d, P):
    points2d = np.dot(P[:3, :3], points3d.T).T + P[:3, 3]
    points2d = points2d[:, :2] / points2d[:, 2][:, np.newaxis]
    points2d = points2d.astype(np.int32)
    return points2d

def lidar_project_vtc():
    path_total_dataset = "/data/datasets/saket/SeeingThroughFogData"
    name_camera_calib = "calib_cam_stereo_left.json"
    tf_tree = "calib_tf_tree_full.json"
    #scan = load_velodyne_scan(scan_path)
    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(path_total_dataset, name_camera_calib, tf_tree)

    return vtc

def lidar_project_points(pointcloud, vtc):
    ps = project_3d_to_2d(pointcloud[:,:3].transpose(), vtc)
    return ps

def lidar_points_image(points,
                           ps,
                           test_image,
                           cmap="jet",
                           saveto = "/home/saket/Dense/lidar_proj.png",
                           ):

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    # PLOT THE IMAGE
    from scipy import ndimage
    import matplotlib as mpl
    import matplotlib.cm as cm
    from matplotlib import pyplot as plt
    
    test_image = np.zeros((1024, 1920, 3))

    fig = plt.figure(dpi=200)
    ax = plt.gca()

    #d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    pixel_values = -d_lidar
    #plt.figure(dpi=200)
    ax.scatter(ps[:, 0], ps[:, 1], s=1, c=pixel_values, linewidths=0, alpha=0.5, cmap=cmap)

    #z_lidar = pointcloud[:, 2]
    #pixel_values = z_lidar
    #fig, ax = plt.subplots(dpi=200)
    #plt.figure(dpi=200)
    #ax.scatter(ps[:, 0], ps[:, 1], s=1, c=pixel_values, linewidths=0, alpha=0.5, cmap=cmap)


    #r_lidar = pointcloud[:, 3]
    #pixel_values = r_lidar
    #ax.figure(dpi=200)
    #ax.scatter(ps[:, 0], ps[:, 1], s=1, c=pixel_values, linewidths=0, alpha=0.5, cmap=cmap)
    ax.imshow(test_image)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    dpi = 200
    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    #else:
    #    fig.show()

    feature = open(saveto, 'rb').read()

    return feature

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class labelStruct(object):

    def __init__(self):
        self.classes = []
        self.truncation = []
        self.occlusion = []
        self.angle = []
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.height = []
        self.width = []
        self.length = []
        self.posx = []
        self.posy = []
        self.posz = []
        self.orient3d = []
        self.rotx = []
        self.roty = []
        self.rotz = []
        self.score = []
        self.qx = []
        self.qy = []
        self.qz = []
        self.qw = []
        self.height_box = []
        self.width_box = []

    def __truediv__(self, image_shape):
        self.xmin = [max(0.0, min(i / image_shape[1], 1.0)) for i in self.xmin]
        self.xmax = [max(0.0, min(i / image_shape[1], 1.0)) for i in self.xmax]
        self.ymin = [max(0.0, min(i / image_shape[0], 1.0)) for i in self.ymin]
        self.ymax = [max(0.0, min(i / image_shape[0], 1.0)) for i in self.ymax]

        return self

    def print_labelStruct(self):
        print('classes:', self.classes)
        print('xmin:', self.xmin)
        print('ymin:', self.ymin)
        print('xmax:', self.xmax)
        print('ymax:', self.ymax)
        #print('label:', self.difficults)
        #print('ymax:', len(self.ymax))
        #print('ymin:', len(self.ymin))


class dataStruct(object):
    images = None
    lidar = None
    image_shape = None
    lidar_shape = None
    calib_dict = None


class ExampleCreator(object):
    source_dir = None

    DIFFICULTY_TO_INT = {
        'easy': 0,
        'moderate': 1,
        'hard': 1
    }

    MAX_TRUNCATION = {
        "easy": 0.15,
        "moderate": 0.30,
        "hard": 0.50
    }

    MAX_OCCLUSION = {
        "easy": 0,
        "moderate": 1,
        "hard": 2
    }

    MIN_BBOX_HEIGHT = {
        "easy": 40,
        "moderate": 25,
        "hard": 25
    }
    calib = 'calib'

    def __init__(self):
        pass

    def load_velo_scan(self, file):
        """Load and parse a velodyne binary file."""
        scan = np.fromfile(file, dtype=np.float32)
        return scan.reshape((-1, 5))

    def read_radar_file(self, path):

        with open(path, 'r') as f:
            data = json.load(f)

        # print data
        data_list = [[0, 0, 0, 0, 0]]
        for target in data['targets']:
            data_list.append([target['x_sc'], target['y_sc'], 0, target['rVelOverGroundOdo_sc'], target['rDist_sc']])

        targets = np.asarray(data_list)

        return targets

    def return_simple_calib_dict(self, base_dir, image_name):
        P, P1 = self.read_calibration_file(os.path.join(base_dir, self.calib), image_name.split('.png')[0])
        P = np.array(P.astype(dtype=np.float32))
        calibration_matrices = {}
        calibration_matrices['P'] = P
        return calibration_matrices

    def proces_label(self, entry_id, image_shape):

        object_list = self.get_kitti_object_list(
            os.path.join(self.source_dir, 'gt_labels_cmore_copied_together/cam_left_labels_TMP', entry_id + '.txt'))

        o = labelStruct()
        for object in object_list:
            for key in object:
                o.__getattribute__(key).append(object[key])
        o = o.__truediv__(image_shape)
        o.print_labelStruct()
        o = self.process_label_val(o)
        o = self.process_class_val(o)
        o = self.process_xmin(o)
        o = self.process_xmax(o)
        o = self.process_ymin(o)
        o = self.process_ymax(o)
        o = self.process_occ(o)
        o = self.process_tr(o)
        o = self.process_hb(o)
        o = self.process_edh(o)
        return o

    def get_kitti_object_list(slef, label_file, camera_to_velodyne=None):
        """Create dict for all objects of the label file, objects are labeled w.r.t KITTI definition"""
        kitti_object_list = []
        # print(label_file)
        try:
            with open(label_file, 'r') as file:
                for line in file:
                    line = line.replace('\n', '')  # remove '\n'
                    kitti_properties = line.split(' ')
                    all_classes.append(kitti_properties[0])
                    object_dict = {
                        'classes': kitti_properties[0].encode('utf-8'),
                        'truncation': float(kitti_properties[1]),
                        'occlusion': int(kitti_properties[2]),
                        'angle': float(kitti_properties[3]),
                        'xmin': int(round(float(kitti_properties[4]))),
                        'ymin': int(round(float(kitti_properties[5]))),
                        'xmax': int(round(float(kitti_properties[6]))),
                        'ymax': int(round(float(kitti_properties[7]))),
                        'height': float(kitti_properties[8]),
                        'width': float(kitti_properties[9]),
                        'length': float(kitti_properties[10]),
                        'posx': float(kitti_properties[11]),
                        'posy': float(kitti_properties[12]),
                        'posz': float(kitti_properties[13]),
                        'orient3d': float(kitti_properties[14]),
                        'rotx': float(kitti_properties[15]),
                        'roty': float(kitti_properties[16]),
                        'rotz': float(kitti_properties[17]),
                        'score': float(kitti_properties[18]),
                        'qx': float(kitti_properties[19]),
                        'qy': float(kitti_properties[20]),
                        'qz': float(kitti_properties[21]),
                        'qw': float(kitti_properties[22]),
                        'height_box': int(round(float(kitti_properties[7]))) - int(round(float(kitti_properties[5]))),
                        'width_box': int(round(float(kitti_properties[6]))) - int(round(float(kitti_properties[4]))),
                    }

                    if camera_to_velodyne is not None:
                        pos = np.asarray([object_dict['posx'], object_dict['posy'], object_dict['posz'], 1])
                        pos_lidar = np.matmul(camera_to_velodyne, pos.T)
                        object_dict['posx_lidar'] = pos_lidar[0]
                        object_dict['posy_lidar'] = pos_lidar[1]
                        object_dict['posz_lidar'] = pos_lidar[2]

                    kitti_object_list.append(object_dict)
                #x = set(all_classes)
                #print("****")
                #print("Unique Classes:",x)
                return kitti_object_list

        except:
            print('Problem occurred when reading label file!')
            return []

    def process_edh(self, parsed_labels):
        """
        :return: label object with filled out diffcult level
        """

        height_box = parsed_labels.height_box
        difficults = []
        for i in range(len(parsed_labels.xmin_)):
            truncation_lvl_item = parsed_labels.tr_[i]
            occlusion_lvl_item = parsed_labels.occ_[i]
            bbox_height_item = parsed_labels.hb_[i]
            level = 1
            for mode in ['hard', 'moderate', 'easy']:
                if truncation_lvl_item <= self.MAX_TRUNCATION[mode] and occlusion_lvl_item <= self.MAX_OCCLUSION[
                    mode] and bbox_height_item >= self.MIN_BBOX_HEIGHT[mode]:
                    level = self.DIFFICULTY_TO_INT[mode]
            difficults.append(level)
        parsed_labels.difficults = difficults
        print('difficults:', parsed_labels.difficults)

        return parsed_labels

    def process_label_val(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        la = []
        for i in parsed_labels.classes:
            #if i == b'Pedestrian':
            #    la.append(1)
            #if i == b'Pedestrian_is_group':
            #    la.append(1)
            #if i == b'person':
            #    la.append(1)
            if i == b'PassengerCar':
                la.append(1)
            #if i == b'PassengerCar_is_group':
            #    la.append(2)
            if i == b'RidableVehicle':
                la.append(1)
            #if i == b'RidableVehicle_is_group':
            #    la.append(3)
            if i == b'LargeVehicle':
                la.append(1)
            #if i == b'LargeVehicle_is_group':
            #    la.append(4)
            #if i == b'train':
            #    la.append(4)
            #if i == b'Obstacle':
            #    la.append(5)
            if i == b'Vehicle':
                la.append(1)
            #if i == b'Vehicle_is_group':
            #    la.append(6)
            if i == b'DontCare':
                la.append(0)
        parsed_labels.la = la
        print('Labels:', parsed_labels.la)

        return parsed_labels

    def process_label_val(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        la = []
        for i in parsed_labels.classes:
            #if i == b'Pedestrian':
            #    la.append(2)
            #if i == b'Pedestrian_is_group':
            #    la.append(1)
            #if i == b'person':
            #    la.append(2)
            if i == b'PassengerCar':
                la.append(1)
            #if i == b'PassengerCar_is_group':
            #    la.append(2)
            if i == b'RidableVehicle':
                la.append(1)
            #if i == b'RidableVehicle_is_group':
            #    la.append(3)
            if i == b'LargeVehicle':
                la.append(1)
            #if i == b'LargeVehicle_is_group':
            #    la.append(4)
            #if i == b'train':
            #    la.append(1)
            #if i == b'Obstacle':
            #    la.append(5)
            if i == b'Vehicle':
                la.append(1)
            #if i == b'Vehicle_is_group':
            #    la.append(6)
            if i == b'DontCare':
                la.append(0)
        parsed_labels.la = la
        print('Labels:', parsed_labels.la)
        return parsed_labels

    def process_class_val(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        ca = []
        for i in parsed_labels.classes:
            #if i == b'Pedestrian':
            #    ca.append(b'Pedestrian')
            #if i == b'Pedestrian_is_group':
            #    ca.append(b'Pedestrian')
            #if i == b'person':
            #    ca.append(b'Pedestrian')
            if i == b'PassengerCar':
                ca.append(b'Vehicle')
            #if i == b'PassengerCar_is_group':
            #    ca.append(b'Vehicle')
            if i == b'RidableVehicle':
                ca.append(b'Vehicle')
            #if i == b'RidableVehicle_is_group':
            #    ca.append(b'Vehicle')
            if i == b'LargeVehicle':
                ca.append(b'Vehicle')
            #if i == b'LargeVehicle_is_group':
            #    ca.append(b'Vehicle')
            #if i == b'train':
            #    ca.append(b'Vehicle')
            #if i == b'Obstacle':
            #    ca.append(b'Obstacle')
            if i == b'Vehicle':
                ca.append(b'Vehicle')
            #if i == b'Vehicle_is_group':
            #    ca.append(b'Vehicle')
            if i == b'DontCare':
                ca.append(b'DontCare')
        parsed_labels.ca = ca
        print('New Classes:', parsed_labels.ca)
        return parsed_labels

    def process_occ(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        occ_ = []
        for idx,xm in enumerate(parsed_labels.occlusion):
            #for i in parsed_labels.xmin:
            if parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'Vehicle':
                occ_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                occ_.append(xm)
        parsed_labels.occ_ = occ_
        print('New OCC:', parsed_labels.occ_)
        return parsed_labels

    def process_tr(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        tr_ = []
        for idx,xm in enumerate(parsed_labels.truncation):
            #for i in parsed_labels.xmin:
            if parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'Vehicle':
                tr_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                tr_.append(xm)
        parsed_labels.tr_ = tr_
        print('New TR:', parsed_labels.tr_)
        return parsed_labels

    def process_hb(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        hb_ = []
        for idx,xm in enumerate(parsed_labels.height_box):
            #for i in parsed_labels.xmin:
            if parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'Vehicle':
                hb_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                hb_.append(xm)
        parsed_labels.hb_ = hb_
        print('New Hb:', parsed_labels.hb_)
        return parsed_labels

    def process_xmin(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        xmin_ = []
        for idx,xm in enumerate(parsed_labels.xmin):
            #for i in parsed_labels.xmin:
            #if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
            #    xmin_.append(xm)
            if parsed_labels.classes[idx] == b'PassengerCar':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'Vehicle':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                xmin_.append(xm)
        parsed_labels.xmin_ = xmin_
        print('New Xmin:', parsed_labels.xmin_)
        return parsed_labels

    def process_xmax(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        xmax_ = []
        for idx,xm in enumerate(parsed_labels.xmax):
            #for i in parsed_labels.xmin:
            #if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
            #    xmax_.append(xm)
            if parsed_labels.classes[idx] == b'PassengerCar':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'Vehicle':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                xmax_.append(xm)
        parsed_labels.xmax_ = xmax_
        print('New xmax:', parsed_labels.xmax_)
        return parsed_labels

    def process_ymin(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        ymin_ = []
        for idx,xm in enumerate(parsed_labels.ymin):
            #for i in parsed_labels.xmin:
            #if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
            #    ymin_.append(xm)
            if parsed_labels.classes[idx] == b'PassengerCar':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'Vehicle':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                ymin_.append(xm)
        parsed_labels.ymin_ = ymin_
        print('New ymin:', parsed_labels.ymin_)
        return parsed_labels

    def process_ymax(self, parsed_labels):
        """
        :return: label object with filled out labels value
        """
        ymax_ = []
        for idx,xm in enumerate(parsed_labels.ymax):
            #for i in parsed_labels.xmin:
            #if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
            #    ymax_.append(xm)
            if parsed_labels.classes[idx] == b'PassengerCar':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'Vehicle':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'DontCare':
                ymax_.append(xm)
        parsed_labels.ymax_ = ymax_
        print('New ymax:', parsed_labels.ymax_)
        return parsed_labels

    def create_example(self, *args):
        raise NotImplementedError

    def read_data(self, *args):
        raise NotImplementedError


class SwedenImagesv2(ExampleCreator):
    gated_keys = ['gated1_rect8']
    image_keys = ['cam_stereo_left_lut']
    point_keys = ['lidar_hdl64_last', 'lidar_hdl64_strongest']
    radar_keys = ['radar_ars300_tfl']

    def __init__(self, source_dir=None):
        self.source_dir = source_dir

    def read_data(self, entry_id, total_id):

        dist_images = {}
        gated_images = {}
        dist_images_shape = {}
        dist_images_height = {}
        dist_images_width = {}
        #gated_images_shape = {}
        dist_lidar = {}
        dist_lidar_shape = {}
        dist_lidar_height = {}
        dist_lidar_width = {}
        gated_images_shape = {}
        gated_images_height = {}
        gated_images_width = {}
        # @ TODO add if needed
        # radar = {}
        # radar_shape = {}
        # read disturbed files

        for folder in self.image_keys:
            file_path = os.path.join(self.source_dir, folder, entry_id + '.png')
            feature = open(file_path, 'rb').read()
            img = PIL.Image.open(file_path, mode="r")
            img_width, img_height = img.size
            dist_images[folder] = feature
            dist_images_shape[folder] = ([img_height, img_width, 3])
            dist_images_height[folder] = ([img_height])
            dist_images_width[folder] = ([img_width])

        for folder in self.point_keys:
            velodyne_name = entry_id + '.bin'
            velodyne_path = os.path.join(self.source_dir, folder, velodyne_name)
            pointcloud = self.load_velo_scan(velodyne_path)
            numpy_lidar_shape = pointcloud.shape
            #dist_lidar[folder] = numpy_lidar
            #dist_lidar_shape[folder] = numpy_lidar_shape
            assert len(numpy_lidar_shape) == 2
            vtc = lidar_project_vtc()
            ps = lidar_project_points(pointcloud, vtc)
            fig = lidar_points_image(points=pointcloud,ps=ps,test_image=img,cmap="jet")
            dist_lidar[folder] = fig
            img_width, img_height = img.size
            dist_lidar_shape[folder] = ([img_height, img_width, 3])
            dist_lidar_height[folder] = ([img_height])
            dist_lidar_width[folder] = ([img_width])
        
        #
        # @ TODO add if needed
        # for folder in self.radar_keys:
        #     velodyne_name = entry_id + '.json'
        #     # velodyne_image = os.path.join(lidar_root, 'point_projected_image_yzI', 'projected', velodyne_name)
        #     velodyne_path = os.path.join(self.source_dir, folder, velodyne_name)
        #     numpy_radar = self.read_radar_file(velodyne_path)
        #     radar[folder] = numpy_radar
        #     radar_shape[folder] = numpy_radar.shape
        #     assert len(numpy_radar.shape) == 2

        for folder in self.gated_keys:
            file_path = os.path.join(self.source_dir, folder, entry_id + '.png')
            feature = open(file_path, 'rb').read()
            img = PIL.Image.open(file_path, mode="r")
            # padding (Converting to Camera Frame)
            img = img.resize((1280, 768), Image.ANTIALIAS).convert('RGB')
            img = add_margin(img, 210, 360, 46, 280, (0,0,0))
            img_width, img_height = img.size
            # save and load
            saveto = "/home/saket/Dense/gated_pad.png"
            img1 = img.save(saveto)
            feature = open(saveto, 'rb').read()
            gated_images[folder] = feature
            gated_images_shape[folder] = ([img_height, img_width, 3])
            gated_images_height[folder] = ([img_height])
            gated_images_width[folder] = ([img_width])


        o = self.proces_label(entry_id, dist_images_shape[self.image_keys[0]])

        data = {}
        data['image_data'] = dist_images
        data['gated_data'] = gated_images
        data['lidar_data'] = dist_lidar
        # @ TODO add if needed
        # data['radar_data'] = radar
        # data['radar_shape'] = radar_shape
        data['image_shape'] = dist_images_shape
        data['image_height'] = dist_images_height
        data['image_width'] = dist_images_width

        data['lidar_shape'] = dist_lidar_shape
        data['lidar_height'] = dist_lidar_height
        data['lidar_width'] = dist_lidar_width
        
        data['gated_shape'] = gated_images_shape
        data['gated_height'] = gated_images_height
        data['gated_width'] = gated_images_width

        #data['gated_shape'] = gated_images_shape
        data['label'] = o
        # data['calibration_matrices'] = self.return_calib_dict(self.source_dir, 'calib_sweden')
        data['name'] = entry_id
        data['total_id'] = total_id

        return data

    def create_example(self, data):

        # print 'Doing the right stuff'
        label = data['label']
        #print("Label:",label)

        lidar_data = data['lidar_data']
        # radar_data = data['radar_data']
        image_data = data['image_data']
        gated_data = data['gated_data']
        image_shape = data['image_shape']
        image_height = data['image_height']
        image_width = data['image_width']

        lidar_shape = data['lidar_shape']
        lidar_height = data['lidar_height']
        lidar_width = data['lidar_width']

        gated_shape = data['gated_shape']
        gated_height = data['gated_height']
        gated_width = data['gated_width']

        #gated_shape = data['gated_shape']
        name = data['name']
        #print('Name:', name)
        total_id = data['total_id']
        # calibration_matrices = data['calibration_matrices']
        # print 'Doing the right stuff'
        image_format = b'png'

        # debugging
        print("Class length:", len(label.ca))
        #print("Class BoundingBox:",len(label.xmin))
        print("New Class BoundingBox:",len(label.xmin_))
        print("New OCC:", len(label.occ_))
        print("New TR:", len(label.tr_))
        print("Name:", name)
        
        feature_dict = {
            #'key': int64_feature(int(total_id)),
            'image/filename': bytes_feature(name.encode("utf8")),
            'image/format': bytes_feature(image_format),
            'image/object/class/text': bytes_feature(label.ca),
            'image/object/class/label': int64_feature(label.la),
            'image/object/bbox/xmin': float_feature(label.xmin_),
            'image/object/bbox/xmax': float_feature(label.xmax_),
            'image/object/bbox/ymin': float_feature(label.ymin_),
            'image/object/bbox/ymax': float_feature(label.ymax_),
            #'image/object/bbox/angle': float_feature(label.angle),
            #'image/object/truncation': float_feature(label.truncation),
            'image/object/difficult': int64_feature(label.difficults),
            #'image/object/occlusion': int64_feature(label.occlusion),
            #'image/object/object/bbox3d/height': float_feature(label.height),
            #'image/object/bbox3d/width': float_feature(label.width),
            #'image/object/bbox3d/length': float_feature(label.length),
            #'image/object/bbox3d/x': float_feature(label.posx),
            #'image/object/bbox3d/y': float_feature(label.posy),
            #'image/object/bbox3d/z': float_feature(label.posz),
            #'image/object/bbox3d/alpha3d': float_feature(label.orient3d),
            #'label': int64_feature(label),
            }

        for key in self.image_keys:
            feature_dict['image/encoded'] = bytes_feature(image_data[key])
            feature_dict['image/height'] = int64_feature([x for x in image_height[key]])
            feature_dict['image/width'] = int64_feature([x for x in image_width[key]])
        
        for key in self.point_keys:
            feature_dict['lidar/encoded'] = bytes_feature(lidar_data[key])
            feature_dict['lidar/height'] = int64_feature([x for x in lidar_height[key]])
            feature_dict['lidar/width'] = int64_feature([x for x in lidar_width[key]])
        
        #for idx, point_key in enumerate(self.point_keys):
            #feature_dict['lidar/encoded'] = float_feature(lidar_data[point_key].flatten().tolist())
            #feature_dict['lidar/shape'] = int64_feature([x for x in lidar_data[point_key].shape])
        # for idx, radar_key in enumerate(self.radar_keys):
        #     feature_dict['radar/'+radar_key] = float_feature(radar_data[radar_key].flatten().tolist())
        #     feature_dict['radar/shape/'+radar_key] = int64_feature([x for x in radar_data[radar_key].shape])
        for key in self.gated_keys:
            feature_dict['gated/encoded'] = bytes_feature(gated_data[key])
            feature_dict['gated/height'] = int64_feature([x for x in gated_height[key]])
            feature_dict['gated/width'] = int64_feature([x for x in gated_width[key]])
        #    feature_dict['gated/' + key] = bytes_feature(gated_data[key])
        #    feature_dict['gated/shape/' + key] = int64_feature([x for x in gated_shape[key]])

        tf_train_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return tf_train_example

    def get_output_filename(self, output_dir, name, idx):
        return '%s/%s_%06d.swedentfrecord' % (output_dir, name, idx)
