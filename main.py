import sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from recognition import RPSRecognizer
from rembg import remove as rmbg
from random import randint

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.recognizer = RPSRecognizer('model_all_psr.keras')

        self.setWindowTitle("Rock Paper Scissors")
        self.setGeometry(100, 100, 1000, 640)

        # 左邊畫面
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 640)
        self.label = ['布 !', '石頭', '剪刀']

        # 右上方文字
        self.prompt_label = QLabel("準備好了嗎？")
        self.prompt_label.setFont(QFont('微軟正黑體', 32))
        self.prompt_label.setAlignment(Qt.AlignCenter)
        
        #中間的自己跟電腦出拳物件
        self.computer_label = QLabel()
        self.computer_label.setAlignment(Qt.AlignLeft)
        
        self.text_vs = QLabel('vs')
        self.text_vs.setVisible(False)
        self.text_vs.setAlignment(Qt.AlignCenter)
        self.text_vs.setStyleSheet("""
                text-align: center;
                font-family: 微軟正黑體;
                font-size: 40px;
        """)

        self.player_label = QLabel()
        self.player_label.setAlignment(Qt.AlignRight)
        
        self.pla_and_com = QLabel('你的出拳                電腦的出拳')
        self.pla_and_com.setVisible(False)
        self.pla_and_com.setAlignment(Qt.AlignCenter)
        self.pla_and_com.setStyleSheet("""
                width: 45px;
                text-align: center;
                font-family: 微軟正黑體;
                font-size: 20px;
        """)

        self.icon_path = ['paper.png', 'scissor.png', 'rock.png']

        self.capture_button = QPushButton("準備好了！")
        self.capture_button.clicked.connect(self.start_game)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* green */
                color: white;
                padding: 50px 10px;
                text-align: center;
                font-family: 微軟正黑體;
                font-size: 30px;
                border: none;
                border-radius: 12px;
            }
            QPushButton:disabled {
                background-color: #828282; /* gray */
            }
            QPushButton:hover {
                background-color: #45a049; /* dark green */
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)

        vs_box = QHBoxLayout()
        vs_box.addWidget(self.player_label)
        vs_box.addWidget(self.text_vs)
        vs_box.addWidget(self.computer_label)

        side_box = QVBoxLayout()
        top_spacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        side_box.addItem(top_spacer)
        side_box.addWidget(self.prompt_label)
        side_box.addStretch(1)
        side_box.addLayout(vs_box)
        side_box.addWidget(self.pla_and_com)
        side_box.addStretch(8)

        side_box.addWidget(self.capture_button)
        
        layout = QHBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(side_box)
        
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 15)  # 15 FPS

    
    def show_vs(self, show: bool): # 遊戲結果出來前顯示問號
        self.player_label.setVisible(show)
        self.computer_label.setVisible(show)
        self.text_vs.setVisible(show)
        if show:
            self.computer_label.setPixmap(QPixmap('question.png'))
            self.player_label.setPixmap(QPixmap('question.png'))
        self.text_vs.setVisible(show)
        self.pla_and_com.setVisible(show)
        

    def show_computer_choice(self): # 顯示電腦出拳
        com = self.computer_choice
        self.computer_label.setPixmap(QPixmap(self.icon_path[com]))


    def show_result(self): # 顯示玩家出拳和輸贏
        com, pla = self.computer_choice, self.player_choice

        self.player_label.setPixmap(QPixmap(self.icon_path[pla]))
        self.computer_label.setPixmap(QPixmap(self.icon_path[com]))
        
        if com == pla:
            self.prompt_label.setText('平手')
        elif (pla == 2 and com == 0) or com == pla + 1:
            self.prompt_label.setText('你輸了 QQ')
        else:
            self.prompt_label.setText('你贏了 好欸')
        self.capture_button.setEnabled(True)

        
    def update_frame(self): # 更新預覽畫面
        ret, frame = self.webcam.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            
            # 只取攝影機中間方形畫面
            height, width, _ = frame.shape
            min_dim = min(height, width)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
            
            # 更新畫面
            frame = cv2.resize(frame, (640, 640))
            height, width, _ = frame.shape
            bytes_per_line = width * 3
            q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.image_label.setPixmap(q_img)


    def start_game(self):
        self.capture_button.setEnabled(False)
        self.show_vs(True)

        self.countdown = 3
        self.prompt_label.setText(self.label[2])
        self.capture_button.setText('再玩一次')
        self.timer_countdown = QTimer()
        self.timer_countdown.setInterval(900)
        self.timer_countdown.timeout.connect(self.update_countdown)
        self.timer_countdown.start()


    def update_countdown(self): # 倒數三秒
        self.countdown -= 1
        self.prompt_label.setText(self.label[self.countdown-1])

        if self.countdown == 1:
            self.computer_choice = randint(0, 2) # 配合模型 label: 0 paper, 1 scissor, 2 rock
            self.show_computer_choice()

        if self.countdown == 0:
            self.timer_countdown.stop()
            self.player_text, self.player_choice = self.recognize_image()
            self.prompt_label.setText(self.player_text)
            self.show_result()


    def recognize_image(self): # 讀取 webcam，辨識手勢
        ret, frame = self.webcam.read()
        height, width, _ = frame.shape
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # 辨識前做去背
        frame_rmbg = rmbg(frame)

        # 儲存圖片作為辨識用
        if ret:
            cv2.imwrite("frame.jpg", frame_rmbg)
        
        return self.recognizer.recognize("frame.jpg")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
