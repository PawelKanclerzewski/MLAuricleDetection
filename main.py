import os
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Input, InputLayer
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def load_array_from_file(file_name):
    array = np.load(file_name, allow_pickle=True)

    return array


def load_dataset_cv(main_path_sets):
    dataset = load_array_from_file(file_name=f'{main_path_sets}/dataset.npy')
    return dataset


def load_dataset(main_path_sets):
    train_set = load_array_from_file(file_name=f'{main_path_sets}/train_set.npy')
    val_set = load_array_from_file(file_name=f'{main_path_sets}/val_set.npy')
    test_set = load_array_from_file(file_name=f'{main_path_sets}/test_set.npy')

    return train_set, val_set, test_set


def unpacking_data(data):
    X = data[:, 0]
    y = data[:, 1]

    X = np.array(list(X), np.float32)
    y = np_utils.to_categorical(y)  # one hot encode outputs

    return X, y


def gen_test_folder_name():
    now = datetime.now()

    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)

    if len(month) == 1:
        month = '0' + month[0]
    if len(day) == 1:
        day = '0' + day[0]
    if len(hour) == 1:
        hour = '0' + hour[0]
    if len(minute) == 1:
        minute = '0' + minute[0]

    folder_name = f"test_{now.year}{month}{day}_{hour}{minute}"

    return folder_name


def flatten_data(dataset, img_type):
    if img_type == 'color':
        n_samples, nh, nw, nch = dataset.shape
        size = nh * nw * nch
    else:
        n_samples, nh, nw = dataset.shape
        size = nh * nw

    flatten_set = dataset.reshape((n_samples, size))

    return flatten_set


def reshape_data(dataset):
    pass
    # TODO


def train_evaluate(X_train, y_train, X_test, y_test, model, epochs, batch_size, callbacks):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
              callbacks=callbacks)
    test_scores = model.evaluate(X_test, y_test, verbose=0)

    # pred = model.predict(X_test)
    # y_pred = np.argmax(pred, axis=1)
    # y_val_classes = np.argmax(y_test, axis=1)
    # print(f"Metric func: acc = {metrics.accuracy_score(y_true=y_val_classes, y_pred=y_pred)}")
    # print(metrics.classification_report(y_val_classes, y_pred))

    return round(test_scores[1] * 100, 2)


def cross_validation(n_splits, X, y, model, epochs, batch_size, callbacks):
    kFold = StratifiedKFold(n_splits=n_splits)

    y = np.argmax(y, axis=1)

    test_scores = np.zeros(n_splits)
    idx = 0

    model.save_weights('reference_model.h5')

    print(f"{n_splits}Fold Cross validation")

    for train, test in kFold.split(X, y):
        model.load_weights('reference_model.h5')  # reset NN model before training process in next fold
        test_scores[idx] = train_evaluate(X[train], y[train], X[test], y[test], model, epochs, batch_size, callbacks)

        print(f"Fold {idx + 1}: test acc = {test_scores[idx]}")
        idx += 1

    print("\nFinal results (all folds):")
    for idx in range(len(test_scores)):
        print(f"Fold {idx + 1}: test acc = {test_scores[idx]}")

    print(f"MEAN value [test acc] = {round(test_scores.mean(), 2)}")
    print(f"STD value = {round(test_scores.std(), 2)}")


def create_model_mlp(n_classes, input_shape, optimizer='adam', func_activation='relu',
                     kernel_initializer='random_uniform', units=256, layers=1):
    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(
        Dense(units=256, input_shape=input_shape, activation=func_activation, kernel_initializer=kernel_initializer))
    # network.add(Activation(func_activation))

    # network.add(BatchNormalization())

    # network.add(Dropout(rate=0.3))

    # model.add(Dense(units=256, activation=func_activation, kernel_initializer=kernel_initializer))
    for i in range(0, layers):
        model.add(Dense(units=units, activation=func_activation, kernel_initializer=kernel_initializer))

    # model.add(Dense(units=1024, activation=func_activation, kernel_initializer=kernel_initializer))

    # model.add(Dense(units=1024, activation=func_activation, kernel_initializer=kernel_initializer))

    # model.add(Dense(units=512, activation=func_activation, kernel_initializer=kernel_initializer))

    # Adding the output
    model.add(Dense(units=n_classes, kernel_initializer='random_uniform', activation='softmax'))

    # Compiling ANN
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
        'accuracy'])  # metrics = [keras.metrics.Accuracy(name='acc'), keras.metrics.Precision(name='prec'), keras.metrics.Recall(name='recall')])

    # Displaying model summary
    model.summary()

    # Return compiled network
    return model


def create_model_cnn(n_classes, input_shape, optimizer='adam', func_activation='relu',
                     kernel_initializer='random_uniform', units=256, layers=1):
    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation=func_activation,
                     kernel_initializer=kernel_initializer))

    # model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation=func_activation, kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D())

    model.add(Flatten())

    # model.add(Dense(512, activation=func_activation, kernel_initializer=kernel_initializer))

    for i in range(0, layers):
        model.add(Dense(units, activation=func_activation, kernel_initializer=kernel_initializer))
    # model.add(Dropout(0.2))

    # Adding the output
    model.add(Dense(n_classes, activation='softmax'))

    # Compiling ANN
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Displaying model summary
    model.summary()

    # Return compiled network
    return model


def testing(_data_split_type, _neural_network, _img_type, _optimizer, _func_activation, _kernel_initializer,
            _reshape_data_method='data_flattening', _epochs=2000, _batch_size=16, _X=492, _Y=702, _units=256,
            _layers=1):
    # data_split_type = 'standard_split'             # 'standard_split' / 'cv'
    # neural_network = 'mlp'                         # 'mlp' / 'cnn'
    #
    # img_type = 'greyscale'                         # 'color' / 'greyscale'
    # img_resize_shape = (32,32)                        # None / (X, Y)
    # reshape_data_method = 'data_flattening'        # 'data_flattening' / 'data_reshape'    -- only for MLP

    data_split_type = _data_split_type  # 'standard_split' / 'cv'
    neural_network = _neural_network  # 'mlp' / 'cnn'

    img_type = _img_type  # 'color' / 'greyscale'
    img_resize_shape = (_X, _Y)  # None / (X, Y)
    reshape_data_method = _reshape_data_method  # 'data_flattening' / 'data_reshape'    -- only for MLP

    if img_resize_shape:
        dataset_folder = f"../data/datasets/{data_split_type}/{img_type}/{img_resize_shape[0]}x{img_resize_shape[1]}"
    else:
        dataset_folder = f"../data/datasets/{data_split_type}/{img_type}/oryginal_size"

    # epochs = 2000
    # batch_size = 16
    #
    # optimizer='adam'                               # tf.keras.optimizers.Adam(learning_rate=0.001)
    # func_activation='relu'
    # kernel_initializer='random_uniform'

    epochs = _epochs
    batch_size = _batch_size

    optimizer = _optimizer  # tf.keras.optimizers.Adam(learning_rate=0.001)
    func_activation = _func_activation
    kernel_initializer = _kernel_initializer

    test_folder_name = gen_test_folder_name()
    if not os.path.exists(f"./logs/{test_folder_name}"):
        os.makedirs(f"./logs/{test_folder_name}")

    # Load data
    if data_split_type == 'standard_split':
        train_set, val_set, test_set = load_dataset(main_path_sets=dataset_folder)
    else:
        dataset = load_dataset_cv(main_path_sets=dataset_folder)  # TODO

    if data_split_type == 'standard_split':
        X_train, y_train = unpacking_data(data=train_set)
        X_test, y_test = unpacking_data(data=test_set)
        X_val, y_val = unpacking_data(data=val_set)
    else:
        X, y = unpacking_data(data=dataset)

    if data_split_type == 'standard_split':
        if img_type == 'greyscale':
            images_shape = (X_train.shape[1], X_train.shape[2], 1)
        elif img_type == 'color':
            images_shape = (X_train.shape[1:])

        n_classes = y_train.shape[1]
    else:
        if img_type == 'greyscale':
            images_shape = (X.shape[1], X.shape[2], 1)
        elif img_type == 'color':
            images_shape = (X.shape[1:])

        n_classes = y.shape[1]

    # Compile model
    if neural_network == 'mlp':
        if reshape_data_method == 'data_flattening':
            if data_split_type == 'standard_split':
                X_train = flatten_data(dataset=X_train, img_type=img_type)
                X_val = flatten_data(dataset=X_val, img_type=img_type)
                X_test = flatten_data(dataset=X_test, img_type=img_type)
            else:
                X_train = X = flatten_data(dataset=X, img_type=img_type)

        elif reshape_data_method == 'data_reshape':  # TODO
            if data_split_type == 'standard_split':
                X_train = reshape_data(X_train)
                X_val = reshape_data(X_val)
                X_test = reshape_data(X_test)
            else:
                X_train = reshape_data(X)

        model = create_model_mlp(n_classes=n_classes, input_shape=(X_train.shape[1],), optimizer=optimizer,
                                 func_activation=func_activation, kernel_initializer=kernel_initializer,
                                 units=_units, layers=_layers)

    elif neural_network == 'cnn':
        model = create_model_cnn(n_classes=n_classes, input_shape=images_shape, optimizer=optimizer,
                                 func_activation=func_activation, kernel_initializer=kernel_initializer,
                                 units=_units, layers=_layers)

    # Init callbacks
    early_stopping_callback = EarlyStopping(monitor="val_accuracy", patience=100,
                                            restore_best_weights=True)  # patience: number of epochs with no improvement after which training will be stopped
    # tensorboard_callback = TensorBoard(log_dir=f'./logs/{test_folder_name}/events')                                 # tensorboard --logdir=logs
    # csv_logger_callback = CSVLogger(f"./logs/{test_folder_name}/model_history_log.csv", append=True)
    # model_checkpoint_callback = ModelCheckpoint(filepath=f"./logs/{test_folder_name}" + "/checkpoints/epoch{epoch:03d}_val-acc{val_accuracy:.3f}", monitor="val_accuracy", save_best_only=True, verbose=1)

    callbacks = [early_stopping_callback]  # , tensorboard_callback, csv_logger_callback, model_checkpoint_callback]

    if data_split_type == 'standard_split':
        # # Train model
        # model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=32, callbacks=[tensorboard_callback])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                  callbacks=callbacks)

        # Evaluate model
        # pred = model.predict(X_val)
        # y_pred = np.argmax(pred, axis=1)
        # y_val_classes = np.argmax(y_val, axis=1)
        # print(f"Metric func: acc = {metrics.accuracy_score(y_true=y_val_classes, y_pred=y_pred)}")
        # print(metrics.classification_report(y_val_classes, y_pred))

        epochs_stop = len(model.history.history['loss'])
        print(f"Epochs_stop: {epochs_stop}")

        scores = model.evaluate(X_train, y_train, verbose=0)
        print("Train acc: %.2f%%" % (scores[1] * 100))
        train_acc = (scores[1] * 100)

        scores = model.evaluate(X_val, y_val, verbose=0)
        print("Val acc: %.2f%%" % (scores[1] * 100))
        val_acc = (scores[1] * 100)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Test acc: %.2f%%" % (scores[1] * 100))
        test_acc = (scores[1] * 100)

        '''
        data_save_row = {'data_split_type': [_data_split_type],
                         'neural_network': [_neural_network],
                         'img_type': [_img_type],
                         'reshape_data_method': [_reshape_data_method],
                         'optimizer': [_optimizer],
                         'func_activation': [_func_activation],
                         'kernel_initializer': [_kernel_initializer],
                         'epochs': [_epochs],
                         'batch_size': [_batch_size],
                         'X': [_X],
                         'Y': [_Y],
                         'units': [_units],
                         'layers': [_layers],
                         'epochs_stop': [epochs_stop],
                         'train_acc': [train_acc],
                         'val_acc': [val_acc],
                         'test_acc': [test_acc]}
        '''

        data_save_row = [_data_split_type, _neural_network, _img_type, _reshape_data_method, _optimizer,
                         _func_activation, _kernel_initializer,
                         _epochs, _batch_size, _X, _Y, _units, _layers, epochs_stop, train_acc, val_acc, test_acc]

        # DF = pd.DataFrame(data_save_row)
        return data_save_row

        # csv_name="Test01_2024_04_12"
        # test_folder_name = gen_test_folder_name()
        # DF.to_csv(f"{csv_name}.csv")
        # model.save('model.h5')

    else:
        cross_validation(n_splits=5, X=X, y=y, model=model, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        data_save_row = [_data_split_type, _neural_network, _img_type, _reshape_data_method, _optimizer,
                         _func_activation, _kernel_initializer,
                         _epochs, _batch_size, _X, _Y, _units, _layers, epochs_stop, train_acc, val_acc, test_acc]

        return data_save_row


def main():  # color/greyscale/cv/32x32/64x64/61x87

    # 'standard_split' / 'cv'
    # 'mlp' / 'cnn'
    # 'color' / 'greyscale'
    networks = ['mlp', 'cnn']
    colors = ['color', 'greyscale']
    optimizers = ['adam', 'nadam']
    activations = ['relu', 'softmax']
    initializers = ['random_uniform', 'truncated_normal']
    # sizes = [(32,32),(64,64),(61,87)]
    neurony = [256, 512, 1024]
    warstwy = [1, 2, 4]

    # Create dataframe
    df = pd.DataFrame(
        columns=['data_split_type', 'neural_network', 'img_type', 'reshape_data_method', 'optimizer', 'func_activation',
                 'kernel_initializer',
                 'epochs', 'batch_size', 'Y', 'Y', 'units', 'layers', 'epochs_stop', 'train_acc', 'val_acc',
                 'test_acc'])

    split = 'standard_split'
    xx = 30
    yy = 43
    incr = 0
    # df.loc[len(df.index)] = testing(_data_split_type='standard_split', _neural_network=networks[0], _img_type=colors[0], _optimizer=optimizers[1], _func_activation=activations[0],
    #        _kernel_initializer=initializers[1],_reshape_data_method='data_flattening', _epochs=2000, _batch_size=16, _X=xx, _Y=yy, _units=unn, _layers=lay)

    for net in networks:
        for col in colors:
            for optim in optimizers:
                for activ in activations:
                    for init in initializers:
                        for neur in neurony:
                            for wars in warstwy:
                                df.loc[len(df.index)] = testing(_data_split_type=split, _neural_network=net,
                                                                _img_type=col, _optimizer=optim, _func_activation=activ,
                                                                _kernel_initializer=init,
                                                                _reshape_data_method='data_flattening', _epochs=2000,
                                                                _batch_size=16, _X=xx, _Y=yy, _units=neur, _layers=wars)
                                incr += 1
                                print(f"Iteration: {incr}/288. Percent: {((incr / 288) * 100)}%")
            folder_name = gen_test_folder_name()
            df.to_csv(f"{folder_name}_{incr}_{((incr / 288) * 100)}%_{xx}x{yy}_{split}.csv")

    # df.loc[len(df.index)] = testing(_data_split_type='standard_split', _neural_network='mlp', _img_type='greyscale', _optimizer='adam', _func_activation='softmax',
    #        _kernel_initializer='random_uniform',_reshape_data_method='data_flattening', _epochs=2000, _batch_size=16, _X=32, _Y=32, _units=256, _layers=1)

    # Save dataframe to csv
    folder_name = gen_test_folder_name()
    df.to_csv(f"{folder_name}_{xx}x{yy}_{split}.csv")

    # testing(_data_split_type='standard_split', _neural_network='mlp', _img_type='greyscale', _optimizer='adam', _func_activation='relu',
    #        _kernel_initializer='random_uniform',_reshape_data_method='data_flattening', _epochs=2000, _batch_size=16, _X=64, _Y=64, _units=256, _layers=1)


if __name__ == "__main__":
    main()