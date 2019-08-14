
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        
        Dialog.setObjectName("Dialog")
        Dialog.resize(800, 600)
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setStyleSheet("background-color: rgb(0, 85, 0);")
        
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(220, 30, 332, 532))
        self.layoutWidget.setObjectName("layoutWidget")
        
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        
        self.gridLayout.addItem(spacerItem, 6, 0, 1, 1)
        
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        
        self.gridLayout.addItem(spacerItem1, 10, 0, 1, 1)
        
        self.StartButton = QtWidgets.QPushButton(self.layoutWidget)
        self.StartButton.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.StartButton.setObjectName("StartButton")
        
        self.gridLayout.addWidget(self.StartButton, 9, 0, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.DesignedByLabel = QtWidgets.QLabel(self.layoutWidget)
        self.DesignedByLabel.setObjectName("DesignedByLabel")
        
        self.gridLayout.addWidget(self.DesignedByLabel, 11, 0, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.PlayerNameLabel = QtWidgets.QLabel(self.layoutWidget)
        self.PlayerNameLabel.setObjectName("PlayerNameLabel")
        
        self.verticalLayout.addWidget(self.PlayerNameLabel)
        
        self.PlayerNameInputBox = QtWidgets.QLineEdit(self.layoutWidget)
        self.PlayerNameInputBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.PlayerNameInputBox.setObjectName("PlayerNameInputBox")
        
        self.verticalLayout.addWidget(self.PlayerNameInputBox)
        
        self.horizontalLayout.addLayout(self.verticalLayout)
        
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        
        self.horizontalLayout.addItem(spacerItem2)
        
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        
        self.ComputerNameLabel = QtWidgets.QLabel(self.layoutWidget)
        self.ComputerNameLabel.setObjectName("ComputerNameLabel")
        
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.ComputerNameLabel)
        
        self.ComputerNameInputBox = QtWidgets.QLineEdit(self.layoutWidget)
        self.ComputerNameInputBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.ComputerNameInputBox.setObjectName("ComputerNameInputBox")
        
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.ComputerNameInputBox)
        
        self.horizontalLayout.addLayout(self.formLayout)
        
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        
        self.gridLayout.addItem(spacerItem3, 1, 0, 1, 1)
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.DeckCountLabel = QtWidgets.QLabel(self.layoutWidget)
        self.DeckCountLabel.setObjectName("DeckCountLabel")
        
        self.verticalLayout_2.addWidget(self.DeckCountLabel, 0, QtCore.Qt.AlignHCenter)
        
        self.DeckCountLineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.DeckCountLineEdit.setStyleSheet("color: rgb(255, 255, 255);")
        self.DeckCountLineEdit.setObjectName("DeckCountLineEdit")
        
        self.verticalLayout_2.addWidget(self.DeckCountLineEdit, 0, QtCore.Qt.AlignHCenter)
        
        self.gridLayout.addLayout(self.verticalLayout_2, 3, 0, 1, 1)
        
        self.DeckColorPixMap = QtWidgets.QLabel(self.layoutWidget)
        self.DeckColorPixMap.setObjectName("DeckColorPixMap")
        
        self.gridLayout.addWidget(self.DeckColorPixMap, 5, 0, 1, 1)
        
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        
        self.gridLayout.addItem(spacerItem4, 8, 0, 1, 1)
        
        self.DeckColorInputBox = QtWidgets.QComboBox(self.layoutWidget)
        self.DeckColorInputBox.setStyleSheet("color: rgb(255, 255, 255); background-color: rgb(138, 138, 138);")
        self.DeckColorInputBox.setObjectName("DeckColorInputBox")
        self.DeckColorInputBox.addItem("")
        self.DeckColorInputBox.addItem("")
        self.DeckColorInputBox.addItem("")
        
        self.gridLayout.addWidget(self.DeckColorInputBox, 7, 0, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.DeckColorLabel = QtWidgets.QLabel(self.layoutWidget)
        self.DeckColorLabel.setObjectName("DeckColorLabel")
        
        self.gridLayout.addWidget(self.DeckColorLabel, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.wagerWarTitle = QtWidgets.QLabel(self.layoutWidget)
        self.wagerWarTitle.setStyleSheet("color: rgb(255, 255, 255);")
        self.wagerWarTitle.setObjectName("wagerWarTitle")
        
        self.gridLayout.addWidget(self.wagerWarTitle, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.StartButton.setText(_translate("Dialog", "START"))
        self.DesignedByLabel.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; font-style:italic; color:#ffffff;\">Designed By: Peoples\'</span></p></body></html>"))
        self.PlayerNameLabel.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Player Name</span></p></body></html>"))
        self.ComputerNameLabel.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Computer Name</span></p></body></html>"))
        self.DeckCountLabel.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">Deck Count</span></p></body></html>"))
        self.DeckColorPixMap.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><img src=\":/Main/production/blueBackVert.bmp\"/></p></body></html>"))
        self.DeckColorInputBox.setItemText(0, _translate("Dialog", "BLUE"))
        self.DeckColorInputBox.setItemText(1, _translate("Dialog", "CAMO"))
        self.DeckColorInputBox.setItemText(2, _translate("Dialog", "RED"))
        self.DeckColorLabel.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">Deck Color</span></p></body></html>"))
        self.wagerWarTitle.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; color:#ffffff;\">WAGER WAR !!!</span></p></body></html>"))