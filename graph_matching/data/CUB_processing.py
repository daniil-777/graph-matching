import numpy as np
import random
import os
import re
import pandas as pd
from distutils.dir_util import copy_tree
from sklearn.model_selection import train_test_split
import pickle


# CUB_path = cfg.CUB_path
# img_path = cfg.CUB_img_path
# annoth_path = cfg.CUB_annoth_path

img_path = "CUB/"
annoth_path = "CUB/parts/"

class CUB_data:
    def __init__(self):
        self.working_dir = "CUB/"
        self.img_path = self.working_dir 
        self.xml_cub_part = self.working_dir + "CUB_xml/"
        self.xml_cub_group_names = self.working_dir + "CUB_xml_groups_names/"
        self.parts_path = self.working_dir  + "parts/"
        self.cache_path = self.working_dir + 'cache/'
        self.annotations_path = self.working_dir + 'annotations/'
        self.read_files()

    def read_files(self):
        self.images = pd.read_csv((img_path + "images.txt"), header=None, delimiter=' ').values
        self.bounding_boxes = pd.read_csv((self.img_path + "bounding_boxes.txt"), header=None, delimiter=' ',
                                          dtype=int).values
        self.image_class_labels = pd.read_csv((self.img_path + "image_class_labels.txt"), header=None, delimiter=' ',
                                              dtype=int).values
        self.part_locs = pd.read_csv((self.parts_path + "part_locs.txt"), header=None, delimiter=' ', dtype=int).values
        self.classes = pd.read_csv((self.img_path + "classes.txt"), header=None, delimiter=' ').values

    def get_dict_parts(self, name_file):  # annotations of birds' parts
        file = open(name_file, "r")
        dictionary = {}
        i = 1
        for line in file:
            dictionary[i] = str("".join(re.split("[^a-zA-Z]*", line)))
            i = i + 1
        file.close()
        return dictionary

    def image_name_to_key(self, image_name):  # extract name of a bird
        array = image_name.split('_')
        return array[-2] + '_' + array[-1].strip('.jpg')

    def image_name_to_dir_name(self, image_name):  # extract code for a folder
        array = image_name.split('.')
        return array[1].split('/')[0]

    def convert_to_PascalVOC(self, label, category, b_box, parts_array, xml_out_dir, fileName):

        xml = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" + "<annotation>\n"
        xml = xml + "\t<image>" + fileName + "</image>\n"
        xml = xml + "\t<voc_id>" + str(label) + "</voc_id>\n"
        xml = xml + "\t<category>" + category + "</category>\n"
        xml = xml + "\t<visible_bounds height=\"" + str(b_box[3]) + "\" width=\"" + str(b_box[2]) + "\" xmin=\"" + str(
            b_box[0]) + "\" ymin=\"" + str(b_box[1]) + "\"/>\n"
        xml = xml + "\t<keypoints>\n"
        for i in range(parts_array.shape[0]):
            index = parts_array[i][0]
            xml = xml + "\t\t<keypoint name=\"" + self.get_dict_parts(self.parts_path + "parts.txt")[
                index] + "\" visible=\"1\" x=\"" + str(
                parts_array[i][1]) + "\" y=\"" + str(parts_array[i][2]) + "\" z=\"" + str(0) + "\"/>\n"
        xml = xml + "\t</keypoints>\n" + "</annotation>"
        if not os.path.exists(xml_out_dir):
            os.makedirs(xml_out_dir)
        # output to a file.
        xmlFilePath = os.path.join(xml_out_dir, fileName + ".xml")

        with open(xmlFilePath, 'w') as f:
            f.write(xml)

    def convert_to_txt(self, label, category, b_box, parts_array, file_out_dir, fileName):
        FilePath = os.path.join(file_out_dir, fileName + ".txt")
        outF = open(FilePath, "w")
        outF.write(fileName + "\n")
        outF.write(str(label) + "\n")
        outF.write(str(category) + "\n")
        string_b_box = str(b_box[3]) + " " + str(b_box[2]) + " " + str(b_box[0]) + " " + str(b_box[1]) + "\n"
        outF.write(string_b_box)
        for i in range(parts_array.shape[0]):
            index = parts_array[i][0]
            str = self.parts[index] + ' ' + str(parts_array[i][1]) + ' ' + str(parts_array[i][2])
            outF.write(str)
        outF.close()

    def get_common_names(self):
        unique_names_list = []
        for i in range(len(self.classes)):
            array = self.classes[i][1].split("_")
            if (any(char.isdigit() for char in array[-1]) == True):
                array[-1] = (array[-1].split("."))[1]
            unique_names_list.append(array[-1])
        unique_names_array = np.array(unique_names_list)
        return np.unique(unique_names_array)

    def group_folders_names(self):  # unite folders with the same final name eg "Albatros"
        names = self.get_common_names()
        files = os.listdir(self.xml_cub_part)
        for i in names:
            if not os.path.exists(self.xml_cub_group_names + i):
                os.makedirs(self.xml_cub_group_names + i)
            # os.makedirs(self.xml_cub_group_names + i) #start with creating folders for each name
            for j in files:
                if (j.split("_")[-1] == i):
                    copy_tree(os.path.join(self.xml_cub_part + j), os.path.join(self.xml_cub_group_names + i))

    def augmentation_birds(self):
        raise NotImplementedError

    def birds_kinds_split(self, path_directories, arg):
        raise NotImplementedError

    def train_test_split(self, threshhold, ratio):
        array_names = []
        list_attributes_all = [self.get_dict_parts('CUB/parts/parts.txt')[i + 1] for i in range(15)]
        dict_classes = {}  # dictionary attributes initialisation
        names_attributes = self.get_dict_parts
        names = self.get_common_names()
        train_array, test_array = np.empty(names.shape[0], dtype=object), np.empty(names.shape[0], dtype=object)
        number = 0
        for i in names:
            # print(i)
            name_current_file = self.xml_cub_group_names + i
            files = os.listdir(name_current_file)
            # print(files)
            number_files = len(files)
            if (number_files > threshhold):
                array_names.append(i)
                dict_classes[i] = list_attributes_all  # fill dictionary
                # print('ok')
                if not os.path.exists(os.path.join(self.annotations_path + i)):
                    os.makedirs(os.path.join(self.annotations_path + i))
                copy_tree(os.path.join(name_current_file),
                          os.path.join(self.annotations_path + i))  # put folder to annotations folder
                X1, X2 = train_test_split(np.arange(number_files), test_size=ratio)

                list_train = [i + '/' + files[j] for j in X1]
                list_test = [i + '/' + files[j] for j in X2]

                train_array[number] = list_train
                test_array[number] = list_test
                number = number + 1
        train_array = train_array[train_array != None]
        test_array = test_array[test_array != None]
        args = {'train': train_array, "test": test_array}
        np.savez(os.path.join(self.working_dir + 'voc_cub_pairs'), **args)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        with open((self.cache_path + 'voc_db_train.pkl'), 'wb') as f:
            pickle.dump(train_array, f)
        with open((self.cache_path + 'voc_db_test.pkl'), 'wb') as f:
            pickle.dump(test_array, f)
        print(array_names)
        with open('dict_annot.txt', 'w') as f:
            print(dict_classes, file=f)

    #         with open('cool_dict.txt','w') as file:
    #             s = "{"
    #             for k in sorted (dict_classes.keys()):
    #                 s = s + str(k) + ": " + "[" + []', '.join(dict_classes[k]) + "]" + "\n"
    #             s = s + "}"
    #             file.write(s)

    # then train/test as a random shaffle of 3/4 indexes and 1/4 respectively
    # then make a final file of all train/test npz
    # also try distribue images over just special kinds without generalisation
    # DATASET_FULL_NAME: PascalVOC!!! creste new file with class CUB_Voc and call this new dataset, in the worst

    #             #case just replace Fulldataset in train_evel with real word
    #         raise NotImplementedError

    def run_xml(self, args):

        for i in range(len(self.images)):
            parts_specific = self.part_locs[(self.part_locs[:, 0] == i) & (self.part_locs[:, 4]) == 1,
                             1:]  # select visible parts with special label
            file_name_ = self.image_name_to_key(self.images[i][1])
            xml_out_dir_ = self.xml_cub_part + self.image_name_to_dir_name(self.images[i][1])
            b_box_ = self.bounding_boxes[i][1:]
            category_ = self.image_name_to_dir_name(self.images[i][1])  # bag should be fixed
            label = self.image_class_labels[i][1]
            if (args == "xml"):
                self.convert_to_PascalVOC(label, category_, b_box_, parts_specific, xml_out_dir_, file_name_)
            elif (args == "txt"):

                self.convert_to_txt(self, label, category_, b_box_, parts_specific, xml_out_dir_, file_name_)
            else:
                raise ValueError('Cannot handle this order {}'.format(args))
        self.group_folders_names()



if __name__ == "__main__":
    CUB = CUB_data()
    CUB.run_xml('xml')
    CUB.train_test_split(340, 0.2)


