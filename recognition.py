import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 不要顯示警告

import tensorflow as tf
from tensorflow.keras.preprocessing import image as preprocess_image
import numpy as np

class RPSRecognizer:
    def __init__(self, model_name: str):
        self.model = tf.keras.models.load_model(model_name)
        self.class_labels = ['paper','scissor', 'rock'] # 與主程式相同順序
        
    def preprocess_image(self, img_path): # 把圖片先改成RGB三維，再多加上一維作為批次
        img = preprocess_image.load_img(img_path, target_size=(50, 50))
        img_array = preprocess_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def recognize(self, img_path: str) -> str:      
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0) # 預測
        
        predicted_label_index = np.argmax(predictions[0])  # 取信心值最大的標籤
        predicted_label = self.class_labels[predicted_label_index]
        # 回傳 label 名稱、index(方便主程式使用)
        return predicted_label, predicted_label_index 

# 測試用
if __name__ == '__main__':
    recognizer = RPSRecognizer('model_all_psr.keras')
    result = recognizer.recognize('frame.jpg')
    print(result)

