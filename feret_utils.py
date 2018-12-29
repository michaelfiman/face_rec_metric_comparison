'''
    Author: 
        Michaelfi
        
    Date: 
        10.7.18
    
    Description: 
        Utils used to get and process data from FERET databse and images
    
    Python Version:
        3.5
'''

import pickle
import cv2 as cv
import numpy as np
from random import shuffle

def get_feret_files_and_tags_dict():
    '''
    This function will unpickle the feret pickle file containing the files and tags from the database.
        
    retunrs:
        dictionary with names of files as keys and id of person in photo as values
    
    '''
    with open('feret_dict.pickle', 'rb') as handle:
        unpickled_files_dict = pickle.load(handle)
    return unpickled_files_dict

def run_face_detection(image_file, min_size=100):
    '''
    This function will run the face detection algorith, used by opencv2 to get all faces in the image
    
    param image_file:
        The path to the file which will be used for detection and cropping
    param min_size:
        The minimum size of pixels (min_size * min_size) which we allow to "recognize" as a face
    
    returns a numpy array with the size (N, 4) where for wach face in the N faces detected we get x, y, w, h 
    will return an empty list of no faces were detected
    '''
    face_cascade = cv.CascadeClassifier('/home/shared/anaconda3/pkgs/opencv-3.3.1-py36h0a11808_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    img = cv.imread(image_file)
    
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.09,
        minNeighbors=3,
        minSize=(min_size, min_size)
    )
    return faces

def get_keys_by_value(dict_t, value):
    keys = list()
    item_list = dict_t.items()
    for item  in item_list:
        if item[1] == value:
            keys.append(item[0])
    return  keys

def create_dataset(feret_dict, percent_of_train = 0.85, subject_count=725, flip=False):
    '''
    This function will create a dataset based on the feret_dict which will contaion <num_of_train> images for train
    and <num_of_val> images for validation. function will change sizes of all images to 128*128
    
    param feret_dict:
        Dictionary which is the output of run_face_detection function (see above)
    param percent_of_train:
        Sets how many images will be used for train
    param_subject_count:
        Number of subjects which will be used for training
    
    returns a tuple of a list of subject ids according to subject count and a dictionay holding the following:
        data_dict['X_train'] - size of (<num_of_train>, 128, 128, 3)
        data_dict['y_train'] - size of (<num_of_train>)
        data_dict['X_eval'] - size of (<num_of_val>, 128, 128, 3)
        data_dict['y_eval'] - - size of (<num_of_val>)
        
    '''
    
    data_dict = {}
    data_dict['X_train'] = np.zeros((1, 96, 96, 3), dtype=np.float32)
    data_dict['y_train'] = np.zeros((1))
    data_dict['X_eval'] = np.zeros((1, 96, 96, 3), dtype=np.float32)
    data_dict['y_eval'] = np.zeros((1))
    
    # Turn the dictionary so that we have a key which the subject ID, and value will be a list of pictures of the subject
    rev_feret_dict = {}
    list_of_ids = list(set(feret_dict.values()))
    for ids in list_of_ids:
        rev_feret_dict[ids] = get_keys_by_value(feret_dict, ids)
    num_of_sample = 0
    
    # For each id and up to the amount of subject count, lets detect faces and create samples for the data set
    for i in range(subject_count):
        subject_id = list_of_ids[i]
        list_of_files = rev_feret_dict[subject_id]
        shuffle(list_of_files)
        num_of_pics_of_subject = len(list_of_files)
        max_train_idx = int(round(num_of_pics_of_subject * percent_of_train))
        for idx, pic_file in enumerate(list_of_files):
            faces = run_face_detection('pics/%s' % (pic_file))
            img = cv.imread('pics/%s' % (pic_file))

            if len(faces) != 0:
                (x, y, w, h) = faces[0]    
                crop_img = img[y:y+h, x:x+w]
                crop_img = cv.resize(crop_img, dsize=(96, 96), interpolation=cv.INTER_NEAREST)
                crop_img = np.expand_dims(crop_img, axis=0)
                if (idx <= max_train_idx):
                    data_dict['X_train'] = np.append(data_dict['X_train'], crop_img, axis=0)
                    data_dict['y_train'] = np.append(data_dict['y_train'], [i], axis=0)
                    if flip:
                        data_dict['X_train'] = np.append(data_dict['X_train'], np.fliplr(crop_img), axis=0)
                        data_dict['y_train'] = np.append(data_dict['y_train'], [i], axis=0)
                else:
                    data_dict['X_eval'] = np.append(data_dict['X_eval'], crop_img, axis=0)
                    data_dict['y_eval'] = np.append(data_dict['y_eval'], [i], axis=0)
    
    mean_image = np.mean(data_dict['X_train'], axis=0)
    std_image = np.std(data_dict['X_train'], axis=0)
    data_dict['X_train'] =  (data_dict['X_train'] - mean_image)/std_image
    data_dict['X_eval'] =  (data_dict['X_eval'] - mean_image)/std_image

    
    data_dict['X_train'] = data_dict['X_train'][1:,:,:,:]
    data_dict['y_train'] = data_dict['y_train'][1:]     
    data_dict['X_eval'] = data_dict['X_eval'][1:,:,:,:]
    data_dict['y_eval'] = data_dict['y_eval'][1:]
    
    return list_of_ids[:subject_count], data_dict, mean_image, std_image

def create_dataset_gs(feret_dict, percent_of_train = 0.85, subject_count=725, flip=False):
    '''
    This function will create a dataset based on the feret_dict which will contaion <num_of_train> images for train
    and <num_of_val> images for validation. function will change sizes of all images to 96*96
    
    param feret_dict:
        Dictionary which is the output of run_face_detection function (see above)
    param percent_of_train:
        Sets how many images will be used for train
    param_subject_count:
        Number of subjects which will be used for training
    
    returns a tuple of a list of subject ids according to subject count and a dictionay holding the following:
        data_dict['X_train'] - size of (<num_of_train>, 96, 96)
        data_dict['y_train'] - size of (<num_of_train>)
        data_dict['X_eval'] - size of (<num_of_val>, 96, 96)
        data_dict['y_eval'] - - size of (<num_of_val>)
        
    '''
    
    data_dict = {}
    data_dict['X_train'] = np.zeros((1, 96, 96, 1), dtype=np.float32)
    data_dict['y_train'] = np.zeros((1))
    data_dict['X_eval'] = np.zeros((1, 96, 96, 1), dtype=np.float32)
    data_dict['y_eval'] = np.zeros((1))
    
    # Turn the dictionary so that we have a key which the subject ID, and value will be a list of pictures of the subject
    rev_feret_dict = {}
    list_of_ids = list(set(feret_dict.values()))
    for ids in list_of_ids:
        rev_feret_dict[ids] = get_keys_by_value(feret_dict, ids)
    num_of_sample = 0
    
    # For each id and up to the amount of subject count, lets detect faces and create samples for the data set
    for i in range(subject_count):
        subject_id = list_of_ids[i]
        list_of_files = rev_feret_dict[subject_id]
        shuffle(list_of_files)
        num_of_pics_of_subject = len(list_of_files)
        max_train_idx = int(round(num_of_pics_of_subject * percent_of_train))
        for idx, pic_file in enumerate(list_of_files):
            faces = run_face_detection('pics/%s' % (pic_file))
            img = cv.imread('pics/%s' % (pic_file))

            if len(faces) != 0:
                (x, y, w, h) = faces[0]    
                crop_img = img[y:y+h, x:x+w]
                crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
                crop_img = cv.resize(crop_img, dsize=(96, 96), interpolation=cv.INTER_NEAREST)
                crop_img = np.expand_dims(crop_img, axis=0)
                crop_img = np.expand_dims(crop_img, axis=3)
                if (idx <= max_train_idx):
                    data_dict['X_train'] = np.append(data_dict['X_train'], crop_img, axis=0)
                    data_dict['y_train'] = np.append(data_dict['y_train'], [i], axis=0)
                    if flip:
                        data_dict['X_train'] = np.append(data_dict['X_train'], np.flip(crop_img, axis=2), axis=0)
                        data_dict['y_train'] = np.append(data_dict['y_train'], [i], axis=0)
                else:
                    data_dict['X_eval'] = np.append(data_dict['X_eval'], crop_img, axis=0)
                    data_dict['y_eval'] = np.append(data_dict['y_eval'], [i], axis=0)
    
    mean_image = np.mean(data_dict['X_train'])
    std_image = np.std(data_dict['X_train'])
    data_dict['X_train'] =  (data_dict['X_train'] - mean_image)/std_image
    data_dict['X_eval'] =  (data_dict['X_eval'] - mean_image)/std_image

    
    data_dict['X_train'] = data_dict['X_train'][1:,:,:,:]
    data_dict['y_train'] = data_dict['y_train'][1:]     
    data_dict['X_eval'] = data_dict['X_eval'][1:,:,:,:]
    data_dict['y_eval'] = data_dict['y_eval'][1:]
    
    return list_of_ids[:subject_count], data_dict, mean_image, std_image

def extract_faces(pic_file):
    '''
    This function will extract faces from a given picture file if faces were recognized in the file and return them as a np array.
    
    param pic_file:
        Name of file to be used (should be in directory dl_ph)
    
    returns None if no face were detected or numpy array size of (N, 96, 96 ,3) where N is the amount of faces recognized
    '''
    output_array = np.zeros((1, 96, 96, 3), dtype=np.float32)
    faces = run_face_detection('%s' % (pic_file), min_size = 20)
    img = cv.imread('%s' % (pic_file))
    
    for x, y, w, h in faces:
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv.resize(crop_img, dsize=(96, 96), interpolation=cv.INTER_NEAREST)
        crop_img = np.expand_dims(crop_img, axis=0)
        output_array = np.append(output_array, crop_img, axis=0)
           
    return None if len(faces) == 0 else output_array[1:,:,:,:]

def extract_faces_gs(pic_file):
    '''
    This function will extract faces from a given picture file if faces were recognized in the file and return them as a np array.
    
    param pic_file:
        Name of file to be used (should be in directory dl_ph)
    
    returns None if no face were detected or numpy array size of (N, 96, 96) where N is the amount of faces recognized
    '''
    output_array = np.zeros((1, 96, 96, 1), dtype=np.float32)
    faces = run_face_detection('%s' % (pic_file), min_size = 20)
    img = cv.imread('%s' % (pic_file))
    
    for x, y, w, h in faces:
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        crop_img = cv.resize(crop_img, dsize=(96, 96), interpolation=cv.INTER_NEAREST)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = np.expand_dims(crop_img, axis=3)
        output_array = np.append(output_array, crop_img, axis=0)
           
    return None if len(faces) == 0 else output_array[1:,:,:,:]
        
    
    
    
    