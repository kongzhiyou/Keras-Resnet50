import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import layers
from keras.models import load_model, model_from_json, Model
from keras.callbacks import ModelCheckpoint
import cv2

class_dict = ['其他_其他_其他', '旺哥_巧克力味_促销装YY2枚', '旺哥_巧克力味_散装', '旺哥_巧克力味_盒装YY12枚', '旺哥_巧克力味_盒装YY2枚', '旺哥_巧克力味_盒装YY6枚',
              '旺哥_柠檬味_促销装YY2枚', '旺哥_柠檬味_散装', '旺哥_柠檬味_盒装YY2枚', '旺哥_柠檬味_盒装YY6枚', '旺哥_牛奶味_促销装YY2枚', '旺哥_牛奶味_散装',
              '旺哥_牛奶味_盒装YY12枚', '旺哥_牛奶味_盒装YY2枚', '旺哥_牛奶味_盒装YY6枚', '旺哥_牛奶味_袋装YY4枚', '旺哥_红丝绒味_促销装YY2枚', '旺哥_红丝绒味_散装',
              '旺哥_红丝绒味_盒装YY2枚', '旺哥_红丝绒味_盒装YY6枚', '旺哥_芝士味_盒装YY18枚', '旺哥_芝士味_盒装YY6枚', '旺哥_芝士味_袋装YY16枚', '旺哥_草莓味_盒装YY18枚',
              '旺哥_草莓味_盒装YY6枚']

# keras.__version__
#
# train_datagen = ImageDataGenerator(
#     shear_range=10,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     preprocessing_function=preprocess_input)
#
# train_generator = train_datagen.flow_from_directory(
#     'data/train',
#     batch_size=32,
#     class_mode='binary',
#     target_size=(224, 224))

# validation_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input)
#
# validation_generator = validation_datagen.flow_from_directory(
#     'data/validation',
#     shuffle=False,
#     class_mode='binary',
#     target_size=(224,224))

conv_base = ResNet50(
    include_top=False,
    weights='imagenet')
#
# for layer in conv_base.layers:
#     layer.trainable = False
#
#
#
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(25, activation='softmax')(x)
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath,verbose=0, save_weights_only=True, mode="max")
model = Model(conv_base.input, predictions)
#
# optimizer = keras.optimizers.Adam()
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])
#
# history = model.fit_generator(generator=train_generator,
#                               epochs=50,
#                               steps_per_epoch=13550//32,
#                               validation_data=validation_generator,
#                               validation_steps=125//32,
#                               callbacks=[checkpoint]
#                               )
#
# # architecture and weights to HDF5
# model.save('my_model.h5')

#architecture to JSON, weights to HDF5
# model.save_weights('my_weights.h5')
# with open('models/keras/architecture.json', 'w') as f:
#         f.write(model.to_json())

#architecture and weights from HDF5
from keras.utils.vis_utils import plot_model

model = load_model('my_model.h5')
plot_model(model=model, to_file='model.png',show_shapes=True)


# architecture from JSON, weights from HDF5
# with open('models/keras/architecture.json') as f:
#     model = model_from_json(f.read())
# model.load_weights('model-ep028-loss0.158-val_loss1.297.h5')


# validation_img_paths = ["data/validation/旺哥_巧克力味_盒装YY2枚/2.jpg",
#                         "/home/pinlan/Keras-Resnet50/data/validation/旺哥_巧克力味_盒装YY6枚/0.jpg",
#                         "/home/pinlan/Keras-Resnet50/data/validation/旺哥_巧克力味_盒装YY6枚/4.jpg",
#                         "/home/pinlan/Keras-Resnet50/data/validation/旺哥_柠檬味_盒装YY2枚/2.jpg"
#                         ]
# img_list = [Image.open(img_path) for img_path in validation_img_paths]

# validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
#                              for img in img_list])

# img_list = glob.glob('data/validation/*/*.jp*g')
# for image in img_list:
#     img = Image.open(image)
#     validation_batch = np.stack([preprocess_input(np.array(img.resize((224, 224))))])
#     pred_probs = model.predict(validation_batch)
#     res1 = np.argmax(pred_probs[0, :])
#     # print(res1)
#     print(image, class_dict[res1])
#
#     img = cv2.imread(image)
#     cv2.putText(img, str(res1), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     # cv2.imshow('classification', img)
#     #
#     # cv2.waitKey(0)
#     image_name = image.split('/')[-1].split('.')[0]
#     cv2.imwrite('data/predict/'+image.split('/')[-2]+'_predict_'+class_dict[res1]+'.jpg', img)

    # print(len(class_dict))
    # print(class_dict[res1])

# fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))

# for i, img in enumerate(img_list):
#     ax = axs[i]
#     ax.axis('off')
#     ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
#                                                             100*pred_probs[i,1]))
#     ax.imshow(img)
