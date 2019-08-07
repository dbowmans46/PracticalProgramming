
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(0, 85, 0);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 20, 711, 531))
        self.widget.setObjectName("widget")

        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.playerDeckMain = QtWidgets.QLabel(self.widget)
        self.playerDeckMain.setObjectName("playerDeckMain")

        self.gridLayout.addWidget(self.playerDeckMain, 11, 2, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)

        self.computerNameLabel = QtWidgets.QLabel(self.widget)
        self.computerNameLabel.setObjectName("computerNameLabel")
        
        self.gridLayout.addWidget(self.computerNameLabel, 4, 3, 1, 1)
        
        self.warTitle = QtWidgets.QLabel(self.widget)
        self.warTitle.setStyleSheet("background-image: url(:/Main/q02th.jpg);")
        self.warTitle.setObjectName("warTitle")
        
        self.gridLayout.addWidget(self.warTitle, 2, 3, 1, 1)
        
        self.playerDeckDiscard = QtWidgets.QLabel(self.widget)
        self.playerDeckDiscard.setObjectName("playerDeckDiscard")
        
        self.gridLayout.addWidget(self.playerDeckDiscard, 11, 4, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.compDeckDiscard = QtWidgets.QLabel(self.widget)
        self.compDeckDiscard.setObjectName("compDeckDiscard")
        
        self.gridLayout.addWidget(self.compDeckDiscard, 10, 4, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.compDeckActive = QtWidgets.QLabel(self.widget)
        self.compDeckActive.setObjectName("compDeckActive")
        
        self.gridLayout.addWidget(self.compDeckActive, 10, 3, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.palyerNameLabel = QtWidgets.QLabel(self.widget)
        self.palyerNameLabel.setObjectName("palyerNameLabel")
        
        self.verticalLayout.addWidget(self.palyerNameLabel)
        
        self.gridLayout.addLayout(self.verticalLayout, 14, 3, 1, 1)
        
        self.compDeckMain = QtWidgets.QLabel(self.widget)
        self.compDeckMain.setObjectName("compDeckMain")
        
        self.gridLayout.addWidget(self.compDeckMain, 10, 2, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.autoCompletPushButton = QtWidgets.QPushButton(self.widget)
        self.autoCompletPushButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.autoCompletPushButton.setObjectName("autoCompletPushButton")
        
        self.gridLayout.addWidget(self.autoCompletPushButton, 16, 2, 1, 1)
        
        self.playerDeckActive = QtWidgets.QLabel(self.widget)
        self.playerDeckActive.setObjectName("playerDeckActive")
        
        self.gridLayout.addWidget(self.playerDeckActive, 11, 3, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        
        self.DesignedByLabel = QtWidgets.QLabel(self.widget)
        self.DesignedByLabel.setObjectName("DesignedByLabel")
        
        self.gridLayout.addWidget(self.DesignedByLabel, 18, 1, 1, 6, QtCore.Qt.AlignHCenter)
        
        self.quitPushButton = QtWidgets.QPushButton(self.widget)
        self.quitPushButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.quitPushButton.setObjectName("quitPushButton")
        
        self.gridLayout.addWidget(self.quitPushButton, 16, 4, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.dealPushButton = QtWidgets.QPushButton(self.widget)
        self.dealPushButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.dealPushButton.setObjectName("dealPushButton")
        
        self.gridLayout.addWidget(self.dealPushButton, 16, 3, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        
        self.gridLayout.addLayout(self.verticalLayout_3, 10, 1, 1, 1)
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.gridLayout.addLayout(self.verticalLayout_2, 2, 1, 1, 1)
        
        self.cardCountCompLabel = QtWidgets.QLabel(self.widget)
        self.cardCountCompLabel.setObjectName("cardCountCompLabel")
        
        self.gridLayout.addWidget(self.cardCountCompLabel, 5, 3, 1, 1)
        
        self.cardCountPlayerLabel = QtWidgets.QLabel(self.widget)
        self.cardCountPlayerLabel.setObjectName("cardCountPlayerLabel")
        
        self.gridLayout.addWidget(self.cardCountPlayerLabel, 15, 3, 1, 1)
        
        self.autoCompletPushButton.raise_()
        
        self.dealPushButton.raise_()
        
        self.warTitle.raise_()
        
        self.playerDeckDiscard.raise_()
        
        self.DesignedByLabel.raise_()
        
        self.compDeckActive.raise_()
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.playerDeckMain.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.computerNameLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">&lt;comp_name&gt;</span></p></body></html>"))
        self.warTitle.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:20pt; color:#ffffff;\">War!!!</span></p></body></html>"))
        self.playerDeckDiscard.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.compDeckDiscard.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.compDeckActive.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.palyerNameLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">&lt;player_name&gt;</span></p></body></html>"))
        self.compDeckMain.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.autoCompletPushButton.setText(_translate("MainWindow", "Auto Complete"))
        self.playerDeckActive.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.DesignedByLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))
        self.quitPushButton.setText(_translate("MainWindow", "Quit"))
        self.dealPushButton.setText(_translate("MainWindow", "Deal"))
        self.cardCountCompLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">Card Count: + &lt;int&gt;</span></p></body></html>"))
        self.cardCountPlayerLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#ffffff;\">Card Count: + &lt;int&gt;</span></p></body></html>"))