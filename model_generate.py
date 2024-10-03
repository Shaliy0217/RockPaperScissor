import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # 關掉 onednn 加速
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd


dataset_path = '' # 這邊填資料集的資料夾
images = []
labels = []

# 把輸入圖片跟 label 對應
for subfolder in os.listdir(dataset_path):
    subfolder_path = os.path.join(dataset_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    # subfolder 當作 label 名稱
    for image_filename in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_filename)
        images.append(image_path)
        labels.append(subfolder)
        
df = pd.DataFrame({'image': images, 'label': labels})

# 0.8 for 訓練，0.2 for 驗證
X_train, X_val, y_train, y_val = train_test_split(df['image'], df['label'], test_size=0.2, random_state=42)
df_train = pd.DataFrame({'image': X_train, 'label': y_train})
df_val = pd.DataFrame({'image': X_val, 'label': y_val})

# 把 label 轉換成數值，以進行機器學習
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train) # 訓練集
y_val = encoder.fit_transform(y_val) # 驗證集


# 加強訓練資料集
train_datagen = ImageDataGenerator(
    rescale=1./255, # 像素值調成 0~1
    rotation_range=45, # 旋轉
    width_shift_range=0.2,  #左右移
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# 用來在訓練過程產生隨機影響圖片的生成器
train = train_datagen.flow_from_dataframe(
    df_train,
    x_col='image',
    y_col='label',
    target_size=(50, 50),
    batch_size=64,
    class_mode='categorical', # 多(三)分類
    shuffle=True
)


# 測試集則不特別做翻轉那些
val_datagen = ImageDataGenerator(
    rescale=1./255)

val = val_datagen.flow_from_dataframe(
    df_val,
    x_col='image',
    y_col='label',
    target_size=(50, 50),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# 加各種層
model = Sequential()
# 模型不大，做 4 次卷積池化就差不多夠了
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3))) # 提取特徵
model.add(MaxPooling2D(pool_size=(2, 2))) # 壓縮特徵圖尺寸成 1/2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu')) # 512個神經元，做影像分類也有人會用 256 or 1024
model.add(Dense(3, activation='softmax'))


# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 依照正確率選擇最好的模型留下
callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='model_mon.keras',  # 模型名稱
                                               verbose=2, 
                                               save_best_only=True, 
                                               monitor='val_accuracy',
                                               mode='max')
# 訓練 50 次
history = model.fit(train, epochs=50, validation_data=val, verbose=2, callbacks=[callbacks])


