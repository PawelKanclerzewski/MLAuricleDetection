import os
import glob
import cv2
import random
import numpy as np


def get_images(path, ext='jpg'):
    img_files = glob.glob(f'{path}/*.{ext}')
    print(f"Loaded {len(img_files)} images.")

    return img_files


def save_array_to_file(file_path, data):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    np.save(file_path, np.asarray(data))

    print(f">>> The array is saved in the file {file_path}.npy")


def load_array_from_file(file_path):
    array = np.load(file_path, allow_pickle=True)

    return array


def gen_rand_indexes(num_range=(1, 101), n_num=2, iters=10):
    return [random.sample(range(num_range[0], num_range[1]), n_num) for i in range(iters)]


def gen_sets_elem_indexes():
    rand_indexes = gen_rand_indexes(num_range=(0, 7), n_num=2, iters=25)

    val_set_indexes = np.array(rand_indexes)[:,
                      0]  # rand_indexes is a nested list, in Python accessing a nested list cannot be done by multi-dimensional slicing (must convert to numpy array - support multi-dimensional slicing)
    test_set_indexes = np.array(rand_indexes)[:, 1]

    save_array_to_file(file_path='../data/val_set_indexes', data=val_set_indexes)
    save_array_to_file(file_path='../data/test_set_indexes', data=test_set_indexes)


def split_data_into_sets(array, indexes=None, sets_proportion=[5, 1, 1]):
    if indexes:
        all_indexes = {x for x in range(len(array))}  # set -> {}
        diff_idx = list(all_indexes.difference(list(indexes)))  # set has 'difference' func

        train_set = list(array[diff_idx])

        if len(indexes) > 1:
            val_set = list([array[indexes[0]]])
            test_set = list([array[indexes[1]]])
        else:
            test_set = list([array[indexes[0]]])
    else:
        random.shuffle(array)

        array = np.array(array)
        train_set = list(array[:sets_proportion[0]])

        if len(sets_proportion) > 1:
            val_set = list(array[sets_proportion[0]:sets_proportion[0] + sets_proportion[1]])
            test_set = list(array[sets_proportion[0] + sets_proportion[1]:])
        else:
            test_set = list(array[sets_proportion[0]:])

    if len(indexes) > 1:
        return train_set, val_set, test_set
    else:
        return train_set, test_set


def prepare_sets(all_data, classes, indexes):
    global_train_set = []
    global_val_set = []
    global_test_set = []

    for idx, class_id in enumerate(classes):
        single_class_set = np.array([x for x in all_data if x[1] == class_id])

        train_set, val_set, test_set = split_data_into_sets(array=single_class_set,
                                                            indexes=(indexes[0][idx], indexes[1][idx]))
        global_train_set.extend(train_set)
        global_val_set.extend(val_set)
        global_test_set.extend(test_set)

    print(
        f"Prepared sets: train_set={len(global_train_set)}, val_set={len(global_val_set)}, test_set={len(global_test_set)}")

    return global_train_set, global_val_set, global_test_set


def load_and_prepare_image(img_path, img_resize_shape, img_type, display_images=False):
    input_image = cv2.imread(img_path)  # BGR channels
    if img_type == 'greyscale':
        output_image = cv2.cvtColor(input_image,
                                    cv2.COLOR_BGR2GRAY)  # in the case of color images, the decoded images will have (default) the channels stored in B G R order (in OpenCV!) [cv2.COLOR_BGR2GRAY: grayscale space conversion code]
    else:
        output_image = input_image

    if img_resize_shape:
        output_image = cv2.resize(output_image, img_resize_shape)

    if display_images:
        cv2.imshow('Input image', input_image)
        cv2.imshow('Output image', output_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # normalize inputs from 0-255 to 0.0-1.0
    output_image = output_image.astype('float32')
    output_image /= 255

    return output_image


def create_dataset(img_list, img_resize_shape, img_type):
    dataset = []
    classes = []

    for img_path in img_list:
        image = load_and_prepare_image(img_path=img_path, img_resize_shape=img_resize_shape, img_type=img_type)

        class_idx = int(os.path.basename(img_path).split('_')[0])
        if class_idx not in classes:
            classes.append(class_idx)

        dataset.append([image, class_idx])

    # random.shuffle(dataset)
    return dataset, classes


def main():
    data_split_type = 'standard_split'  # 'standard_split' / 'cv'

    img_type = 'color'  # 'color' / 'greyscale'
    img_resize_shape = (30, 43)  # None / (X, Y)

    if img_resize_shape:
        folder_to_save = f"../data/datasets/{data_split_type}/{img_type}/{img_resize_shape[0]}x{img_resize_shape[1]}"
    else:
        folder_to_save = f"../data/datasets/{data_split_type}/{img_type}/oryginal_size"

    img_files = get_images(path='../data/subset-1', ext='jpg')

    dataset, classes = create_dataset(img_list=img_files, img_resize_shape=img_resize_shape, img_type=img_type)

    if data_split_type == 'standard_split':

        val_set_indexes = load_array_from_file(file_path='../data/val_set_indexes.npy')
        test_set_indexes = load_array_from_file(file_path='../data/test_set_indexes.npy')

        train_set, val_set, test_set = prepare_sets(all_data=dataset, classes=classes,
                                                    indexes=(val_set_indexes, test_set_indexes))

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        save_array_to_file(file_path=f'{folder_to_save}/train_set', data=train_set)
        save_array_to_file(file_path=f'{folder_to_save}/val_set', data=val_set)
        save_array_to_file(file_path=f'{folder_to_save}/test_set', data=test_set)

        array = load_array_from_file(file_path=f'{folder_to_save}/train_set.npy')
        print(array)
    else:
        random.shuffle(dataset)

        save_array_to_file(file_path=f'{folder_to_save}/dataset', data=dataset)

        array = load_array_from_file(file_path=f'{folder_to_save}/dataset.npy')
        print(array)


if __name__ == "__main__":
    # gen_sets_elem_indexes()       # -> ZAD.1
    main()  # -> ZAD.2









