import tensorflow as tf
import keras
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D

def MLP(trn, cfg):
    input_layer = Input(shape=(trn.X.shape[1],))
    clayer = input_layer
    for i in range(len(cfg['width'])):
        if len(cfg['dropout']) > 0:
            clayer = Dense(cfg['width'][i], activation='relu')(clayer)
            clayer = Dropout(cfg['dropout'][i])(clayer)
        elif len(cfg['weight_decay']) > 0:
            clayer = Dense(cfg['width'][i], activation='relu', kernel_regularizer=l2(cfg['weight_decay'][i]))(clayer)
        else:
            clayer = Dense(cfg['width'][i], activation='relu')(clayer)
    output_layer = Dense(trn.Y.shape[1], activation='linear')(clayer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model  

def CNN_Global(trn, cfg):
    input_layer = Input(shape=trn.X.shape[1:])
    clayer = input_layer
    for i in range(len(cfg['width'])):
        if i%2 == 0:
            strides=2
        else:
            strides=1
        if len(cfg['weight_decay']) > 0:
            clayer = Conv2D(filters=cfg['width'][i], kernel_size=3, strides=strides, kernel_regularizer=l2(cfg['weight_decay'][i]))(clayer)
        else:
            clayer = Conv2D(filters=cfg['width'][i], kernel_size=3, strides=strides)(clayer)
        if len(cfg['batch_norm']) > 0 and cfg['batch_norm'][i]:
            clayer = BatchNormalization()(clayer)
        clayer = Activation('relu')(clayer)
        if len(cfg['dropout']) > 0:
            clayer = Dropout(cfg['dropout'][i])(clayer)        
    clayer = Conv2D(filters=trn.Y.shape[1], kernel_size=1)(clayer)
    output_layer = GlobalAveragePooling2D()(clayer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def VGG16(trn, cfg):
    input_tensor = Input(shape=trn.X.shape[1:])
    base_model = keras.applications.VGG16(include_top=False,
                                          weights='imagenet',
                                          input_tensor=input_tensor,
                                          input_shape=(trn.X.shape[1:]))
    output = base_model.layers[-1].output
    output = Flatten()(output)
    output = Dense(4096, activation='relu')(output)
    if cfg['dropout'] > 0:
        output = Dropout(cfg['dropout'])(output)  
    output = Dense(4096, activation='relu')(output)
    if cfg['dropout'] > 0:
        output= Dropout(cfg['dropout'])(output)  
    output = Dense(trn.Y.shape[1], activation='linear')(output)

    model = Model(inputs=input_tensor, outputs=output)
    return model

def ResNet50(trn, cfg):
    input_tensor = Input(shape=trn.X.shape[1:])
    base_model = keras.applications.ResNet50(include_top=False,
                                          weights='imagenet',
                                          input_tensor=input_tensor,
                                          input_shape=(trn.X.shape[1:]))
    output = base_model.layers[-1].output
    output = Flatten()(output)
    output = Dense(4096, activation='relu')(output)
    if cfg['dropout'] > 0:
        output = Dropout(cfg['dropout'])(output)  
    output = Dense(4096, activation='relu')(output)
    if cfg['dropout'] > 0:
        output= Dropout(cfg['dropout'])(output)  
    output = Dense(trn.Y.shape[1], activation='linear')(output)

    model = Model(inputs=input_tensor, outputs=output)
    return model

def get_model(cfg, trn):
    if cfg['model'] == 'MLP':
        model = MLP(trn, cfg)
    elif cfg['model'] == 'CNN':
        model = CNN(trn, cfg)
    elif cfg['model'] == 'CNN_Global':
        model = CNN_Global(trn, cfg)
    elif cfg['model'] == 'VGG16':
        model = VGG16(trn, cfg)
    elif cfg['model'] == 'ResNet50':
        model = ResNet50(trn, cfg)
    if cfg['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=cfg['learning_rate'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = keras.optimizers.SGD(lr=cfg['learning_rate'], momentum=0.9)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    return model



