import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QVBoxLayout, QPushButton, QHBoxLayout, QLineEdit
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence

PATH='gruj.h5'
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #self.lbl1 = QLabel('Enter your sentence:')
        self.te = QTextEdit()
        self.te.setAcceptRichText(False)
        self.te.setStyleSheet(("color: black;"
            "background-color: white;"
            "border-style: solid;"
            "border-width: 4px;"
            "border-radius: 10px;"
            "border-color: black;"
            "font: bold 14px;"
            "min-width: 10em;"
             "padding: 6px;"))
        #self.lbl2 = QLabel('The number of words is 0')

        okButton = QPushButton('Detect',self)
        okButton.clicked.connect(self.text_changed)

        okButton.setStyleSheet(
            "color: black;"
            "background-color: white;"
            "border-style: dashed;"
            "border-width: 4px;"
            "border-radius: 10px;"
            "border-color: black;"
            "font: bold 14px;"
            "min-width: 10em;"
             "padding: 6px;")
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addStretch(1)
        vbox = QVBoxLayout()
        #self.te.textChanged.connect(self.text_changed)

        #vbox.addWidget(self.lbl1)
        vbox.addWidget(self.te)
        #vbox.addWidget(self.lbl2)
        vbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        self.le = QLineEdit(self)
        vbox.addWidget(self.le)
        vbox.addStretch(30)
        self.le.setStyleSheet("color: black;"
            "background-color: white;"
            "border-style: solid;"
            "border-width: 4px;"
            "border-radius: 10px;"
            "border-color: black;"
            "font: bold 14px;"
            "min-width: 10em;"
             "padding: 6px;")

        self.setLayout(vbox)

        self.setWindowTitle('QTextEdit')
        self.setGeometry(400, 300, 500, 300)
        self.show()

    def text_changed(self):
        text = self.te.toPlainText()
        X = []
        loaded_model = tf.keras.models.load_model(PATH)
        max_features = 4500

        t = Tokenizer()
        t.fit_on_texts([text])

        sequences = t.texts_to_sequences([text])[0]
        X=pad_sequences([sequences,[1]], maxlen=max_features, padding='pre')
        yhat = loaded_model.predict_classes(X)
        if yhat[0] == 1:
            self.le.setText("False")
        else:
            self.le.setText(("True"))
        # self.le.setText(self, str(yhat))
        # print(yhat)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())