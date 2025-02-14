import tensorflow as tf
import numpy as np
import os
import PIL.Image
import json

all_classes = []


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
            if i == b'Pedestrian':
                la.append(1)
            #if i == b'Pedestrian_is_group':
            #    la.append(1)
            if i == b'person':
                la.append(1)
            if i == b'PassengerCar':
                la.append(2)
            #if i == b'PassengerCar_is_group':
            #    la.append(2)
            if i == b'RidableVehicle':
                la.append(3)
            #if i == b'RidableVehicle_is_group':
            #    la.append(3)
            if i == b'LargeVehicle':
                la.append(4)
            #if i == b'LargeVehicle_is_group':
            #    la.append(4)
            if i == b'train':
                la.append(4)
            #if i == b'Obstacle':
            #    la.append(5)
            #if i == b'Vehicle':
            #    la.append(2)
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
            if i == b'Pedestrian':
                ca.append(b'Pedestrian')
            #if i == b'Pedestrian_is_group':
            #    ca.append(b'Pedestrian')
            if i == b'person':
                ca.append(b'Pedestrian')
            if i == b'PassengerCar':
                ca.append(b'PassengerCar')
            #if i == b'PassengerCar_is_group':
            #    ca.append(b'Vehicle')
            if i == b'RidableVehicle':
                ca.append(b'Cyclist')
            #if i == b'RidableVehicle_is_group':
            #    ca.append(b'Vehicle')
            if i == b'LargeVehicle':
                ca.append(b'LargeVehicle')
            #if i == b'LargeVehicle_is_group':
            #    ca.append(b'Vehicle')
            if i == b'train':
                ca.append(b'LargeVehicle')
            #if i == b'Obstacle':
            #    ca.append(b'Obstacle')
            #if i == b'Vehicle':
            #    ca.append(b'Vehicle')
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person' or parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person' or parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person' or parsed_labels.classes[idx] == b'RidableVehicle'  or parsed_labels.classes[idx] == b'PassengerCar' or parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'PassengerCar':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                xmin_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'PassengerCar':
                xmax_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                xmax_.append(xm)            
            elif parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'PassengerCar':    
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                ymin_.append(xm)
            elif parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':
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
            if parsed_labels.classes[idx] == b'Pedestrian' or parsed_labels.classes[idx] == b'person':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'PassengerCar':
                ymax_.append(xm)
            elif parsed_labels.classes[idx] == b'RidableVehicle':
                ymax_.append(xm)    
            elif parsed_labels.classes[idx] == b'LargeVehicle' or parsed_labels.classes[idx] == b'train':    
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
    gated_keys = ['gated0_rect8', 'gated1_rect8', 'gated2_rect8', 'gated_full_acc_rect8']
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
        gated_images_shape = {}
        dist_lidar = {}
        dist_lidar_shape = {}
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
            numpy_lidar = self.load_velo_scan(velodyne_path)
            numpy_lidar_shape = numpy_lidar.shape
            dist_lidar[folder] = numpy_lidar
            dist_lidar_shape[folder] = numpy_lidar_shape
            assert len(numpy_lidar_shape) == 2
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
            img_width, img_height = img.size
            gated_images[folder] = feature
            gated_images_shape[folder] = ([img_height, img_width, 3])



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
        data['gated_shape'] = gated_images_shape
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
        gated_shape = data['gated_shape']
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
            #'image/object/truncated': float_feature(label.truncation),
            'image/object/difficult': int64_feature(label.difficults),
            #'image/object/occluded': int64_feature(label.occlusion),
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
        #for idx, point_key in enumerate(self.point_keys):
            #feature_dict['lidar/encoded'] = float_feature(lidar_data[point_key].flatten().tolist())
            #feature_dict['lidar/shape'] = int64_feature([x for x in lidar_data[point_key].shape])
        # for idx, radar_key in enumerate(self.radar_keys):
        #     feature_dict['radar/'+radar_key] = float_feature(radar_data[radar_key].flatten().tolist())
        #     feature_dict['radar/shape/'+radar_key] = int64_feature([x for x in radar_data[radar_key].shape])
        #for key in self.gated_keys:
        #    feature_dict['gated/' + key] = bytes_feature(gated_data[key])
        #    feature_dict['gated/shape/' + key] = int64_feature([x for x in gated_shape[key]])

        tf_train_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return tf_train_example

    def get_output_filename(self, output_dir, name, idx):
        return '%s/%s_%06d.swedentfrecord' % (output_dir, name, idx)
