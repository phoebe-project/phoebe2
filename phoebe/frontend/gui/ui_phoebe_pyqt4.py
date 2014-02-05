# Generated using gui_update.sh and pyuic4 from reading files
# ./dialogParamEdit.ui
# ./dialogPriorEdit.ui
# ./fileEntryWidget.ui
# ./phoebe.ui
# ./popAbout.ui
# ./popFileEntryColWidget.ui
# ./popFileEntry.ui
# ./popHelp.ui
# ./popLock.ui
# ./popObsOptions.ui
# ./popPlot.ui
# ./popPrefs.ui
# ./popTimeSelect.ui
# ./tree_datasetWidget.ui
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_popParamEdit_Dialog(object):
    def setupUi(self, popParamEdit_Dialog):
        popParamEdit_Dialog.setObjectName(_fromUtf8("popParamEdit_Dialog"))
        popParamEdit_Dialog.resize(727, 412)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(popParamEdit_Dialog.sizePolicy().hasHeightForWidth())
        popParamEdit_Dialog.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(popParamEdit_Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.descriptionLongToggle = QtGui.QPushButton(popParamEdit_Dialog)
        self.descriptionLongToggle.setMinimumSize(QtCore.QSize(24, 24))
        self.descriptionLongToggle.setMaximumSize(QtCore.QSize(24, 24))
        self.descriptionLongToggle.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/ellipsis.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.descriptionLongToggle.setIcon(icon)
        self.descriptionLongToggle.setCheckable(True)
        self.descriptionLongToggle.setObjectName(_fromUtf8("descriptionLongToggle"))
        self.gridLayout_3.addWidget(self.descriptionLongToggle, 2, 2, 1, 1)
        self.descriptionLabel = QtGui.QLabel(popParamEdit_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.descriptionLabel.sizePolicy().hasHeightForWidth())
        self.descriptionLabel.setSizePolicy(sizePolicy)
        self.descriptionLabel.setObjectName(_fromUtf8("descriptionLabel"))
        self.gridLayout_3.addWidget(self.descriptionLabel, 2, 1, 1, 1)
        self.label_4 = QtGui.QLabel(popParamEdit_Dialog)
        self.label_4.setMinimumSize(QtCore.QSize(90, 0))
        self.label_4.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 1)
        self.parameterLabel = QtGui.QLabel(popParamEdit_Dialog)
        self.parameterLabel.setObjectName(_fromUtf8("parameterLabel"))
        self.gridLayout_3.addWidget(self.parameterLabel, 1, 1, 1, 1)
        self.label_5 = QtGui.QLabel(popParamEdit_Dialog)
        self.label_5.setMinimumSize(QtCore.QSize(90, 24))
        self.label_5.setMaximumSize(QtCore.QSize(90, 16777215))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_3.addWidget(self.label_5, 1, 0, 1, 1)
        self.kindLabel = QtGui.QLabel(popParamEdit_Dialog)
        self.kindLabel.setObjectName(_fromUtf8("kindLabel"))
        self.gridLayout_3.addWidget(self.kindLabel, 0, 0, 1, 1)
        self.objectLabel = QtGui.QLabel(popParamEdit_Dialog)
        self.objectLabel.setObjectName(_fromUtf8("objectLabel"))
        self.gridLayout_3.addWidget(self.objectLabel, 0, 1, 1, 1)
        self.descriptionLongLabel = QtGui.QLabel(popParamEdit_Dialog)
        self.descriptionLongLabel.setObjectName(_fromUtf8("descriptionLongLabel"))
        self.gridLayout_3.addWidget(self.descriptionLongLabel, 3, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.stackedWidget = QtGui.QStackedWidget(popParamEdit_Dialog)
        self.stackedWidget.setObjectName(_fromUtf8("stackedWidget"))
        self.page = QtGui.QWidget()
        self.page.setObjectName(_fromUtf8("page"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.page)
        self.verticalLayout_4.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.frame = QtGui.QFrame(self.page)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setVerticalSpacing(20)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.float_value = QtGui.QDoubleSpinBox(self.frame)
        self.float_value.setObjectName(_fromUtf8("float_value"))
        self.gridLayout_2.addWidget(self.float_value, 0, 2, 1, 1)
        self.float_units = QtGui.QLabel(self.frame)
        self.float_units.setObjectName(_fromUtf8("float_units"))
        self.gridLayout_2.addWidget(self.float_units, 0, 3, 1, 1)
        self.float_infobutton = QtGui.QPushButton(self.frame)
        self.float_infobutton.setEnabled(True)
        self.float_infobutton.setMaximumSize(QtCore.QSize(24, 24))
        self.float_infobutton.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/info.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.float_infobutton.setIcon(icon1)
        self.float_infobutton.setDefault(False)
        self.float_infobutton.setFlat(False)
        self.float_infobutton.setObjectName(_fromUtf8("float_infobutton"))
        self.gridLayout_2.addWidget(self.float_infobutton, 0, 5, 1, 1)
        self.label_3 = QtGui.QLabel(self.frame)
        self.label_3.setMinimumSize(QtCore.QSize(80, 0))
        self.label_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.float_resetbutton = QtGui.QPushButton(self.frame)
        self.float_resetbutton.setMinimumSize(QtCore.QSize(24, 24))
        self.float_resetbutton.setMaximumSize(QtCore.QSize(24, 24))
        self.float_resetbutton.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/refresh.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.float_resetbutton.setIcon(icon2)
        self.float_resetbutton.setObjectName(_fromUtf8("float_resetbutton"))
        self.gridLayout_2.addWidget(self.float_resetbutton, 0, 4, 1, 1)
        self.check = QtGui.QCheckBox(self.frame)
        self.check.setMaximumSize(QtCore.QSize(24, 24))
        self.check.setText(_fromUtf8(""))
        self.check.setObjectName(_fromUtf8("check"))
        self.gridLayout_2.addWidget(self.check, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.verticalLayout_4.addWidget(self.frame)
        self.frame_2 = QtGui.QFrame(self.page)
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(self.frame_2)
        self.label.setMinimumSize(QtCore.QSize(80, 0))
        self.label.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.presetLabel = QtGui.QLabel(self.frame_2)
        self.presetLabel.setMinimumSize(QtCore.QSize(80, 0))
        self.presetLabel.setMaximumSize(QtCore.QSize(80, 16777215))
        self.presetLabel.setObjectName(_fromUtf8("presetLabel"))
        self.gridLayout.addWidget(self.presetLabel, 1, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.constraintQual = QtGui.QLabel(self.frame_2)
        self.constraintQual.setObjectName(_fromUtf8("constraintQual"))
        self.horizontalLayout.addWidget(self.constraintQual)
        self.constraintText = QtGui.QLineEdit(self.frame_2)
        self.constraintText.setObjectName(_fromUtf8("constraintText"))
        self.horizontalLayout.addWidget(self.constraintText)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.constraintReset = QtGui.QPushButton(self.frame_2)
        self.constraintReset.setMinimumSize(QtCore.QSize(0, 24))
        self.constraintReset.setMaximumSize(QtCore.QSize(24, 24))
        self.constraintReset.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/bin-3.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.constraintReset.setIcon(icon3)
        self.constraintReset.setObjectName(_fromUtf8("constraintReset"))
        self.gridLayout.addWidget(self.constraintReset, 0, 2, 1, 1)
        self.constraintHelp = QtGui.QPushButton(self.frame_2)
        self.constraintHelp.setEnabled(True)
        self.constraintHelp.setMaximumSize(QtCore.QSize(24, 24))
        self.constraintHelp.setText(_fromUtf8(""))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/help.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.constraintHelp.setIcon(icon4)
        self.constraintHelp.setObjectName(_fromUtf8("constraintHelp"))
        self.gridLayout.addWidget(self.constraintHelp, 0, 3, 1, 1)
        self.presetHelp = QtGui.QPushButton(self.frame_2)
        self.presetHelp.setEnabled(True)
        self.presetHelp.setMaximumSize(QtCore.QSize(24, 24))
        self.presetHelp.setText(_fromUtf8(""))
        self.presetHelp.setIcon(icon4)
        self.presetHelp.setObjectName(_fromUtf8("presetHelp"))
        self.gridLayout.addWidget(self.presetHelp, 1, 3, 1, 1)
        self.presetWidget = QtGui.QWidget(self.frame_2)
        self.presetWidget.setObjectName(_fromUtf8("presetWidget"))
        self.gridLayout.addWidget(self.presetWidget, 1, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.verticalLayout_4.addWidget(self.frame_2)
        self.stackedWidget.addWidget(self.page)
        self.page_4 = QtGui.QWidget()
        self.page_4.setObjectName(_fromUtf8("page_4"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.page_4)
        self.verticalLayout_5.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.frame_5 = QtGui.QFrame(self.page_4)
        self.frame_5.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_5.setObjectName(_fromUtf8("frame_5"))
        self.gridLayout_8 = QtGui.QGridLayout(self.frame_5)
        self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
        self.strlist_other_value = QtGui.QLineEdit(self.frame_5)
        self.strlist_other_value.setEnabled(False)
        self.strlist_other_value.setObjectName(_fromUtf8("strlist_other_value"))
        self.gridLayout_8.addWidget(self.strlist_other_value, 0, 1, 1, 1)
        self.label_8 = QtGui.QLabel(self.frame_5)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_8.addWidget(self.label_8, 0, 0, 1, 1)
        self.verticalLayout_5.addWidget(self.frame_5)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem1)
        self.stackedWidget.addWidget(self.page_4)
        self.page_2 = QtGui.QWidget()
        self.page_2.setObjectName(_fromUtf8("page_2"))
        self.gridLayout_4 = QtGui.QGridLayout(self.page_2)
        self.gridLayout_4.setContentsMargins(0, -1, 0, -1)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.frame_3 = QtGui.QFrame(self.page_2)
        self.frame_3.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_3.setObjectName(_fromUtf8("frame_3"))
        self.gridLayout_5 = QtGui.QGridLayout(self.frame_3)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.strlist_time_custom_text = QtGui.QLineEdit(self.frame_3)
        self.strlist_time_custom_text.setEnabled(False)
        self.strlist_time_custom_text.setObjectName(_fromUtf8("strlist_time_custom_text"))
        self.gridLayout_5.addWidget(self.strlist_time_custom_text, 2, 1, 1, 1)
        self.strlist_time_auto = QtGui.QRadioButton(self.frame_3)
        self.strlist_time_auto.setChecked(True)
        self.strlist_time_auto.setObjectName(_fromUtf8("strlist_time_auto"))
        self.gridLayout_5.addWidget(self.strlist_time_auto, 0, 0, 1, 1)
        self.strlist_time_custom = QtGui.QRadioButton(self.frame_3)
        self.strlist_time_custom.setObjectName(_fromUtf8("strlist_time_custom"))
        self.gridLayout_5.addWidget(self.strlist_time_custom, 2, 0, 1, 1)
        self.strlist_time_linspace = QtGui.QRadioButton(self.frame_3)
        self.strlist_time_linspace.setObjectName(_fromUtf8("strlist_time_linspace"))
        self.gridLayout_5.addWidget(self.strlist_time_linspace, 1, 0, 1, 1)
        self.strlist_linspace_widget = QtGui.QWidget(self.frame_3)
        self.strlist_linspace_widget.setEnabled(False)
        self.strlist_linspace_widget.setObjectName(_fromUtf8("strlist_linspace_widget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.strlist_linspace_widget)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(self.strlist_linspace_widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.strlist_linspace_min = QtGui.QDoubleSpinBox(self.strlist_linspace_widget)
        self.strlist_linspace_min.setDecimals(4)
        self.strlist_linspace_min.setMaximum(1000000000.0)
        self.strlist_linspace_min.setObjectName(_fromUtf8("strlist_linspace_min"))
        self.horizontalLayout_2.addWidget(self.strlist_linspace_min)
        self.label_7 = QtGui.QLabel(self.strlist_linspace_widget)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_2.addWidget(self.label_7)
        self.strlist_linspace_max = QtGui.QDoubleSpinBox(self.strlist_linspace_widget)
        self.strlist_linspace_max.setDecimals(4)
        self.strlist_linspace_max.setMaximum(1000000000.0)
        self.strlist_linspace_max.setObjectName(_fromUtf8("strlist_linspace_max"))
        self.horizontalLayout_2.addWidget(self.strlist_linspace_max)
        self.label_6 = QtGui.QLabel(self.strlist_linspace_widget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_2.addWidget(self.label_6)
        self.strlist_linspace_num = QtGui.QDoubleSpinBox(self.strlist_linspace_widget)
        self.strlist_linspace_num.setDecimals(0)
        self.strlist_linspace_num.setMaximum(100000000.0)
        self.strlist_linspace_num.setObjectName(_fromUtf8("strlist_linspace_num"))
        self.horizontalLayout_2.addWidget(self.strlist_linspace_num)
        self.gridLayout_5.addWidget(self.strlist_linspace_widget, 1, 1, 1, 1)
        self.gridLayout_4.addWidget(self.frame_3, 2, 0, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem2, 3, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtGui.QWidget()
        self.page_3.setObjectName(_fromUtf8("page_3"))
        self.gridLayout_6 = QtGui.QGridLayout(self.page_3)
        self.gridLayout_6.setContentsMargins(0, -1, -1, -1)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.frame_4 = QtGui.QFrame(self.page_3)
        self.frame_4.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_4.setObjectName(_fromUtf8("frame_4"))
        self.gridLayout_7 = QtGui.QGridLayout(self.frame_4)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.pushButton = QtGui.QPushButton(self.frame_4)
        self.pushButton.setEnabled(False)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.gridLayout_7.addWidget(self.pushButton, 1, 1, 1, 1)
        self.strlist_refstypes_custom = QtGui.QRadioButton(self.frame_4)
        self.strlist_refstypes_custom.setMaximumSize(QtCore.QSize(80, 16777215))
        self.strlist_refstypes_custom.setObjectName(_fromUtf8("strlist_refstypes_custom"))
        self.gridLayout_7.addWidget(self.strlist_refstypes_custom, 1, 0, 1, 1)
        self.strlist_refstypes_auto = QtGui.QRadioButton(self.frame_4)
        self.strlist_refstypes_auto.setObjectName(_fromUtf8("strlist_refstypes_auto"))
        self.gridLayout_7.addWidget(self.strlist_refstypes_auto, 0, 0, 1, 1)
        self.pushButton_2 = QtGui.QPushButton(self.frame_4)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.gridLayout_7.addWidget(self.pushButton_2, 1, 2, 1, 1)
        self.strlist_refstypes_widget = QtGui.QWidget(self.frame_4)
        self.strlist_refstypes_widget.setObjectName(_fromUtf8("strlist_refstypes_widget"))
        self.gridLayout_9 = QtGui.QGridLayout(self.strlist_refstypes_widget)
        self.gridLayout_9.setMargin(0)
        self.gridLayout_9.setObjectName(_fromUtf8("gridLayout_9"))
        self.checkBox_2 = QtGui.QCheckBox(self.strlist_refstypes_widget)
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.gridLayout_9.addWidget(self.checkBox_2, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.strlist_refstypes_widget, 2, 1, 1, 2)
        self.gridLayout_6.addWidget(self.frame_4, 3, 0, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem3, 4, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        self.verticalLayout.addWidget(self.stackedWidget)
        self.defaults_frame = QtGui.QFrame(popParamEdit_Dialog)
        self.defaults_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.defaults_frame.setFrameShadow(QtGui.QFrame.Raised)
        self.defaults_frame.setObjectName(_fromUtf8("defaults_frame"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.defaults_frame)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_9 = QtGui.QLabel(self.defaults_frame)
        self.label_9.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_4.addWidget(self.label_9)
        self.default_set = QtGui.QPushButton(self.defaults_frame)
        self.default_set.setEnabled(False)
        self.default_set.setObjectName(_fromUtf8("default_set"))
        self.horizontalLayout_4.addWidget(self.default_set)
        self.default_reset = QtGui.QPushButton(self.defaults_frame)
        self.default_reset.setEnabled(False)
        self.default_reset.setObjectName(_fromUtf8("default_reset"))
        self.horizontalLayout_4.addWidget(self.default_reset)
        self.verticalLayout.addWidget(self.defaults_frame)
        self.buttonBox = QtGui.QDialogButtonBox(popParamEdit_Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(popParamEdit_Dialog)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popParamEdit_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popParamEdit_Dialog.reject)
        QtCore.QObject.connect(self.descriptionLongToggle, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.descriptionLongLabel.setVisible)
        QtCore.QObject.connect(self.strlist_time_custom, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.strlist_time_custom_text.setEnabled)
        QtCore.QObject.connect(self.strlist_time_linspace, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.strlist_linspace_widget.setEnabled)
        QtCore.QObject.connect(self.strlist_refstypes_custom, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.strlist_refstypes_widget.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(popParamEdit_Dialog)

    def retranslateUi(self, popParamEdit_Dialog):
        popParamEdit_Dialog.setWindowTitle(QtGui.QApplication.translate("popParamEdit_Dialog", "PHOEBE - Parameter Edit", None, QtGui.QApplication.UnicodeUTF8))
        self.descriptionLongToggle.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "toggle long description", None, QtGui.QApplication.UnicodeUTF8))
        self.descriptionLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Description", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Description:", None, QtGui.QApplication.UnicodeUTF8))
        self.parameterLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Parameter:", None, QtGui.QApplication.UnicodeUTF8))
        self.kindLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Object:", None, QtGui.QApplication.UnicodeUTF8))
        self.objectLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Object Name", None, QtGui.QApplication.UnicodeUTF8))
        self.descriptionLongLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Long Description", None, QtGui.QApplication.UnicodeUTF8))
        self.float_units.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "units", None, QtGui.QApplication.UnicodeUTF8))
        self.float_infobutton.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "change unit options for this parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Value:", None, QtGui.QApplication.UnicodeUTF8))
        self.float_resetbutton.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "reset parameter to defaults", None, QtGui.QApplication.UnicodeUTF8))
        self.check.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "mark for adjustment/fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Constraint:", None, QtGui.QApplication.UnicodeUTF8))
        self.presetLabel.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Presets:", None, QtGui.QApplication.UnicodeUTF8))
        self.constraintQual.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "{qual} = ", None, QtGui.QApplication.UnicodeUTF8))
        self.constraintReset.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "remove constraint", None, QtGui.QApplication.UnicodeUTF8))
        self.constraintHelp.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "constraint help", None, QtGui.QApplication.UnicodeUTF8))
        self.presetHelp.setToolTip(QtGui.QApplication.translate("popParamEdit_Dialog", "constraint preset help", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Value:", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_time_custom_text.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "[]", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_time_auto.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "auto", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_time_custom.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "custom list:", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_time_linspace.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "np.linspace:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "min:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "max:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "num:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Currently Plotted", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_refstypes_custom.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "custom", None, QtGui.QApplication.UnicodeUTF8))
        self.strlist_refstypes_auto.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "auto", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_2.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Currently Active", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_2.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "CheckBox", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Default:", None, QtGui.QApplication.UnicodeUTF8))
        self.default_set.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Set as Default", None, QtGui.QApplication.UnicodeUTF8))
        self.default_reset.setText(QtGui.QApplication.translate("popParamEdit_Dialog", "Reset from Default", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popPriorEdit_Dialog(object):
    def setupUi(self, popPriorEdit_Dialog):
        popPriorEdit_Dialog.setObjectName(_fromUtf8("popPriorEdit_Dialog"))
        popPriorEdit_Dialog.resize(400, 300)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(popPriorEdit_Dialog.sizePolicy().hasHeightForWidth())
        popPriorEdit_Dialog.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(popPriorEdit_Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(popPriorEdit_Dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.distributionCombo = QtGui.QComboBox(popPriorEdit_Dialog)
        self.distributionCombo.setObjectName(_fromUtf8("distributionCombo"))
        self.distributionCombo.addItem(_fromUtf8(""))
        self.distributionCombo.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.distributionCombo)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.floatWidget = QtGui.QWidget(popPriorEdit_Dialog)
        self.floatWidget.setObjectName(_fromUtf8("floatWidget"))
        self.gridLayout = QtGui.QGridLayout(self.floatWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout.addWidget(self.floatWidget)
        self.stackedWidget = QtGui.QStackedWidget(popPriorEdit_Dialog)
        self.stackedWidget.setObjectName(_fromUtf8("stackedWidget"))
        self.page = QtGui.QWidget()
        self.page.setObjectName(_fromUtf8("page"))
        self.gridLayout_4 = QtGui.QGridLayout(self.page)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.label_3 = QtGui.QLabel(self.page)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_4.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.page)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_4.addWidget(self.label_2, 0, 0, 1, 1)
        self.uniform_lower = QtGui.QDoubleSpinBox(self.page)
        self.uniform_lower.setObjectName(_fromUtf8("uniform_lower"))
        self.gridLayout_4.addWidget(self.uniform_lower, 0, 1, 1, 1)
        self.uniform_upper = QtGui.QDoubleSpinBox(self.page)
        self.uniform_upper.setObjectName(_fromUtf8("uniform_upper"))
        self.gridLayout_4.addWidget(self.uniform_upper, 1, 1, 1, 1)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtGui.QWidget()
        self.page_2.setObjectName(_fromUtf8("page_2"))
        self.gridLayout_5 = QtGui.QGridLayout(self.page_2)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.label_4 = QtGui.QLabel(self.page_2)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_5.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QtGui.QLabel(self.page_2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_5.addWidget(self.label_5, 1, 0, 1, 1)
        self.normal_mu = QtGui.QDoubleSpinBox(self.page_2)
        self.normal_mu.setObjectName(_fromUtf8("normal_mu"))
        self.gridLayout_5.addWidget(self.normal_mu, 0, 1, 1, 1)
        self.normal_sigma = QtGui.QDoubleSpinBox(self.page_2)
        self.normal_sigma.setObjectName(_fromUtf8("normal_sigma"))
        self.gridLayout_5.addWidget(self.normal_sigma, 1, 1, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.verticalLayout.addWidget(self.stackedWidget)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.buttonBox = QtGui.QDialogButtonBox(popPriorEdit_Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(popPriorEdit_Dialog)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popPriorEdit_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popPriorEdit_Dialog.reject)
        QtCore.QObject.connect(self.distributionCombo, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), self.stackedWidget.setCurrentIndex)
        QtCore.QMetaObject.connectSlotsByName(popPriorEdit_Dialog)

    def retranslateUi(self, popPriorEdit_Dialog):
        popPriorEdit_Dialog.setWindowTitle(QtGui.QApplication.translate("popPriorEdit_Dialog", "PHOEBE - Prior Edit", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popPriorEdit_Dialog", "Distribution", None, QtGui.QApplication.UnicodeUTF8))
        self.distributionCombo.setItemText(0, QtGui.QApplication.translate("popPriorEdit_Dialog", "uniform", None, QtGui.QApplication.UnicodeUTF8))
        self.distributionCombo.setItemText(1, QtGui.QApplication.translate("popPriorEdit_Dialog", "normal", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popPriorEdit_Dialog", "Upper", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popPriorEdit_Dialog", "Lower", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("popPriorEdit_Dialog", "Mu", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("popPriorEdit_Dialog", "Sigma", None, QtGui.QApplication.UnicodeUTF8))


class Ui_fileEntryWidget(object):
    def setupUi(self, fileEntryWidget):
        fileEntryWidget.setObjectName(_fromUtf8("fileEntryWidget"))
        fileEntryWidget.resize(1038, 300)
        self.gridLayout = QtGui.QGridLayout(fileEntryWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.bp_lc_GridLayout = QtGui.QGridLayout()
        self.bp_lc_GridLayout.setObjectName(_fromUtf8("bp_lc_GridLayout"))
        self.bp_dataTreeView = QtGui.QTreeWidget(fileEntryWidget)
        self.bp_dataTreeView.setObjectName(_fromUtf8("bp_dataTreeView"))
        self.bp_lc_GridLayout.addWidget(self.bp_dataTreeView, 0, 0, 1, 1)
        self.bp_lc_verticalLayout = QtGui.QVBoxLayout()
        self.bp_lc_verticalLayout.setObjectName(_fromUtf8("bp_lc_verticalLayout"))
        self.bp_addCommandLinkButton = QtGui.QCommandLinkButton(fileEntryWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bp_addCommandLinkButton.sizePolicy().hasHeightForWidth())
        self.bp_addCommandLinkButton.setSizePolicy(sizePolicy)
        self.bp_addCommandLinkButton.setMinimumSize(QtCore.QSize(10, 0))
        self.bp_addCommandLinkButton.setMaximumSize(QtCore.QSize(35, 35))
        self.bp_addCommandLinkButton.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("icons/add.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bp_addCommandLinkButton.setIcon(icon)
        self.bp_addCommandLinkButton.setObjectName(_fromUtf8("bp_addCommandLinkButton"))
        self.bp_lc_verticalLayout.addWidget(self.bp_addCommandLinkButton)
        self.bp_lc_GridLayout.addLayout(self.bp_lc_verticalLayout, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.bp_lc_GridLayout, 0, 0, 1, 1)

        self.retranslateUi(fileEntryWidget)
        QtCore.QMetaObject.connectSlotsByName(fileEntryWidget)

    def retranslateUi(self, fileEntryWidget):
        fileEntryWidget.setWindowTitle(QtGui.QApplication.translate("fileEntryWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_dataTreeView.headerItem().setText(0, QtGui.QApplication.translate("fileEntryWidget", "Item", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_dataTreeView.headerItem().setText(1, QtGui.QApplication.translate("fileEntryWidget", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_dataTreeView.headerItem().setText(2, QtGui.QApplication.translate("fileEntryWidget", "Plotting", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_dataTreeView.headerItem().setText(3, QtGui.QApplication.translate("fileEntryWidget", "Sigma", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_dataTreeView.headerItem().setText(4, QtGui.QApplication.translate("fileEntryWidget", "Actions", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_addCommandLinkButton.setToolTip(QtGui.QApplication.translate("fileEntryWidget", "add data", None, QtGui.QApplication.UnicodeUTF8))


class Ui_PHOEBE_MainWindow(object):
    def setupUi(self, PHOEBE_MainWindow):
        PHOEBE_MainWindow.setObjectName(_fromUtf8("PHOEBE_MainWindow"))
        PHOEBE_MainWindow.resize(1193, 1903)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PHOEBE_MainWindow.sizePolicy().hasHeightForWidth())
        PHOEBE_MainWindow.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/phoebe-gui.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PHOEBE_MainWindow.setWindowIcon(icon)
        PHOEBE_MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        PHOEBE_MainWindow.setDockOptions(QtGui.QMainWindow.AllowTabbedDocks|QtGui.QMainWindow.AnimatedDocks|QtGui.QMainWindow.VerticalTabs)
        PHOEBE_MainWindow.setUnifiedTitleAndToolBarOnMac(True)
        self.centralwidget = QtGui.QWidget(PHOEBE_MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_11 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.mp_stackedWidget = QtGui.QStackedWidget(self.centralwidget)
        self.mp_stackedWidget.setObjectName(_fromUtf8("mp_stackedWidget"))
        self.splash = QtGui.QWidget()
        self.splash.setObjectName(_fromUtf8("splash"))
        self.gridLayout_6 = QtGui.QGridLayout(self.splash)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.mp_splash_binaryPushButton = QtGui.QPushButton(self.splash)
        self.mp_splash_binaryPushButton.setObjectName(_fromUtf8("mp_splash_binaryPushButton"))
        self.gridLayout_6.addWidget(self.mp_splash_binaryPushButton, 2, 0, 1, 1)
        self.mp_splash_triplePushButton = QtGui.QPushButton(self.splash)
        self.mp_splash_triplePushButton.setObjectName(_fromUtf8("mp_splash_triplePushButton"))
        self.gridLayout_6.addWidget(self.mp_splash_triplePushButton, 3, 0, 1, 1)
        self.mp_splash_openPushButton = QtGui.QPushButton(self.splash)
        self.mp_splash_openPushButton.setObjectName(_fromUtf8("mp_splash_openPushButton"))
        self.gridLayout_6.addWidget(self.mp_splash_openPushButton, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.splash)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_6.addWidget(self.label_3, 0, 0, 1, 1)
        self.mp_stackedWidget.addWidget(self.splash)
        self.overview = QtGui.QWidget()
        self.overview.setObjectName(_fromUtf8("overview"))
        self.gridLayout_11 = QtGui.QGridLayout(self.overview)
        self.gridLayout_11.setObjectName(_fromUtf8("gridLayout_11"))
        self.mp_gridWidget = QtGui.QWidget(self.overview)
        self.mp_gridWidget.setObjectName(_fromUtf8("mp_gridWidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.mp_gridWidget)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.mp_plotGridLayout = QtGui.QGridLayout()
        self.mp_plotGridLayout.setSpacing(3)
        self.mp_plotGridLayout.setObjectName(_fromUtf8("mp_plotGridLayout"))
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.mp_plotGridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.verticalLayout_4.addLayout(self.mp_plotGridLayout)
        self.gridLayout_11.addWidget(self.mp_gridWidget, 0, 0, 1, 1)
        self.mp_stackedWidget.addWidget(self.overview)
        self.plotexpand = QtGui.QWidget()
        self.plotexpand.setObjectName(_fromUtf8("plotexpand"))
        self.gridLayout_12 = QtGui.QGridLayout(self.plotexpand)
        self.gridLayout_12.setObjectName(_fromUtf8("gridLayout_12"))
        self.frame = QtGui.QFrame(self.plotexpand)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtGui.QFrame.NoFrame)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.mp_expandWidget = QtGui.QWidget(self.frame)
        self.mp_expandWidget.setObjectName(_fromUtf8("mp_expandWidget"))
        self.gridLayout = QtGui.QGridLayout(self.mp_expandWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mp_expandLayout = QtGui.QGridLayout()
        self.mp_expandLayout.setObjectName(_fromUtf8("mp_expandLayout"))
        self.gridLayout.addLayout(self.mp_expandLayout, 1, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.mp_expandWidget)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.gridLayout_12.addWidget(self.frame, 0, 0, 1, 1)
        self.mp_stackedWidget.addWidget(self.plotexpand)
        self.openglexpand = QtGui.QWidget()
        self.openglexpand.setObjectName(_fromUtf8("openglexpand"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.openglexpand)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.widget = QtGui.QWidget(self.openglexpand)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout_14 = QtGui.QGridLayout(self.widget)
        self.gridLayout_14.setMargin(0)
        self.gridLayout_14.setSpacing(2)
        self.gridLayout_14.setMargin(0)
        self.gridLayout_14.setObjectName(_fromUtf8("gridLayout_14"))
        self.pushButton_5 = QtGui.QPushButton(self.widget)
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.gridLayout_14.addWidget(self.pushButton_5, 3, 0, 1, 1)
        self.mp_glLayout = QtGui.QGridLayout()
        self.mp_glLayout.setObjectName(_fromUtf8("mp_glLayout"))
        self.gridLayout_14.addLayout(self.mp_glLayout, 2, 0, 1, 1)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setSpacing(2)
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.pushButton_6 = QtGui.QPushButton(self.widget)
        self.pushButton_6.setEnabled(False)
        self.pushButton_6.setMaximumSize(QtCore.QSize(24, 24))
        self.pushButton_6.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/pop.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_6.setIcon(icon1)
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        self.horizontalLayout_9.addWidget(self.pushButton_6)
        self.mpgl_gridPushButton = QtGui.QPushButton(self.widget)
        self.mpgl_gridPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.mpgl_gridPushButton.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/grid.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.mpgl_gridPushButton.setIcon(icon2)
        self.mpgl_gridPushButton.setObjectName(_fromUtf8("mpgl_gridPushButton"))
        self.horizontalLayout_9.addWidget(self.mpgl_gridPushButton)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem1)
        self.gridLayout_14.addLayout(self.horizontalLayout_9, 1, 0, 1, 1)
        self.verticalLayout_5.addWidget(self.widget)
        self.mp_stackedWidget.addWidget(self.openglexpand)
        self.systemEdit = QtGui.QWidget()
        self.systemEdit.setObjectName(_fromUtf8("systemEdit"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.systemEdit)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        spacerItem2 = QtGui.QSpacerItem(44, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem2)
        self.mpsys_gridPushButton = QtGui.QPushButton(self.systemEdit)
        self.mpsys_gridPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.mpsys_gridPushButton.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/grid.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.mpsys_gridPushButton.setIcon(icon3)
        self.mpsys_gridPushButton.setObjectName(_fromUtf8("mpsys_gridPushButton"))
        self.horizontalLayout_10.addWidget(self.mpsys_gridPushButton)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.webviewwidget = QtGui.QWidget(self.systemEdit)
        self.webviewwidget.setObjectName(_fromUtf8("webviewwidget"))
        self.verticalLayout_6.addWidget(self.webviewwidget)
        spacerItem4 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem4)
        self.mp_stackedWidget.addWidget(self.systemEdit)
        self.verticalLayout_11.addWidget(self.mp_stackedWidget)
        PHOEBE_MainWindow.setCentralWidget(self.centralwidget)
        self.tb_MenuBar = QtGui.QMenuBar(PHOEBE_MainWindow)
        self.tb_MenuBar.setGeometry(QtCore.QRect(0, 0, 1193, 25))
        self.tb_MenuBar.setObjectName(_fromUtf8("tb_MenuBar"))
        self.tb_fileMenu = QtGui.QMenu(self.tb_MenuBar)
        self.tb_fileMenu.setObjectName(_fromUtf8("tb_fileMenu"))
        self.menuImport = QtGui.QMenu(self.tb_fileMenu)
        self.menuImport.setObjectName(_fromUtf8("menuImport"))
        self.tb_editMenu = QtGui.QMenu(self.tb_MenuBar)
        self.tb_editMenu.setObjectName(_fromUtf8("tb_editMenu"))
        self.tb_viewMenu = QtGui.QMenu(self.tb_MenuBar)
        self.tb_viewMenu.setObjectName(_fromUtf8("tb_viewMenu"))
        self.tb_helpMenu = QtGui.QMenu(self.tb_MenuBar)
        self.tb_helpMenu.setObjectName(_fromUtf8("tb_helpMenu"))
        self.tb_advancedMenu = QtGui.QMenu(self.tb_MenuBar)
        self.tb_advancedMenu.setObjectName(_fromUtf8("tb_advancedMenu"))
        PHOEBE_MainWindow.setMenuBar(self.tb_MenuBar)
        self.sb_StatusBar = QtGui.QStatusBar(PHOEBE_MainWindow)
        self.sb_StatusBar.setEnabled(True)
        self.sb_StatusBar.setObjectName(_fromUtf8("sb_StatusBar"))
        PHOEBE_MainWindow.setStatusBar(self.sb_StatusBar)
        self.lp_DockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lp_DockWidget.sizePolicy().hasHeightForWidth())
        self.lp_DockWidget.setSizePolicy(sizePolicy)
        self.lp_DockWidget.setMinimumSize(QtCore.QSize(373, 522))
        self.lp_DockWidget.setFloating(False)
        self.lp_DockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable)
        self.lp_DockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.lp_DockWidget.setObjectName(_fromUtf8("lp_DockWidget"))
        self.lp_DockWidgetContents = QtGui.QWidget()
        self.lp_DockWidgetContents.setObjectName(_fromUtf8("lp_DockWidgetContents"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.lp_DockWidgetContents)
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setContentsMargins(2, 2, 2, 0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.lp_verticalLayout = QtGui.QVBoxLayout()
        self.lp_verticalLayout.setSpacing(2)
        self.lp_verticalLayout.setObjectName(_fromUtf8("lp_verticalLayout"))
        self.lp_mainWidget = QtGui.QWidget(self.lp_DockWidgetContents)
        self.lp_mainWidget.setObjectName(_fromUtf8("lp_mainWidget"))
        self.verticalLayout_14 = QtGui.QVBoxLayout(self.lp_mainWidget)
        self.verticalLayout_14.setSpacing(2)
        self.verticalLayout_14.setMargin(0)
        self.verticalLayout_14.setMargin(0)
        self.verticalLayout_14.setObjectName(_fromUtf8("verticalLayout_14"))
        self.lp_orbitPushButton = QtGui.QPushButton(self.lp_mainWidget)
        self.lp_orbitPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/menu-2.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_orbitPushButton.setIcon(icon4)
        self.lp_orbitPushButton.setCheckable(True)
        self.lp_orbitPushButton.setChecked(True)
        self.lp_orbitPushButton.setObjectName(_fromUtf8("lp_orbitPushButton"))
        self.verticalLayout_14.addWidget(self.lp_orbitPushButton)
        self.lp_orbitTreeView = ParameterTreeWidget(self.lp_mainWidget)
        self.lp_orbitTreeView.setMinimumSize(QtCore.QSize(0, 80))
        self.lp_orbitTreeView.setIndentation(2)
        self.lp_orbitTreeView.setObjectName(_fromUtf8("lp_orbitTreeView"))
        self.verticalLayout_14.addWidget(self.lp_orbitTreeView)
        self.lp_compPushButton = QtGui.QPushButton(self.lp_mainWidget)
        self.lp_compPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.lp_compPushButton.setIcon(icon4)
        self.lp_compPushButton.setCheckable(True)
        self.lp_compPushButton.setChecked(True)
        self.lp_compPushButton.setObjectName(_fromUtf8("lp_compPushButton"))
        self.verticalLayout_14.addWidget(self.lp_compPushButton)
        self.lp_compWidget = QtGui.QWidget(self.lp_mainWidget)
        self.lp_compWidget.setObjectName(_fromUtf8("lp_compWidget"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.lp_compWidget)
        self.verticalLayout_7.setSpacing(2)
        self.verticalLayout_7.setMargin(0)
        self.verticalLayout_7.setMargin(0)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.lp_compTreeView = ParameterTreeWidget(self.lp_compWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lp_compTreeView.sizePolicy().hasHeightForWidth())
        self.lp_compTreeView.setSizePolicy(sizePolicy)
        self.lp_compTreeView.setMinimumSize(QtCore.QSize(0, 80))
        self.lp_compTreeView.setIndentation(2)
        self.lp_compTreeView.setObjectName(_fromUtf8("lp_compTreeView"))
        self.verticalLayout_7.addWidget(self.lp_compTreeView)
        self.lp_meshPushButton = QtGui.QPushButton(self.lp_compWidget)
        self.lp_meshPushButton.setMinimumSize(QtCore.QSize(0, 20))
        self.lp_meshPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.lp_meshPushButton.setIcon(icon4)
        self.lp_meshPushButton.setCheckable(True)
        self.lp_meshPushButton.setChecked(True)
        self.lp_meshPushButton.setObjectName(_fromUtf8("lp_meshPushButton"))
        self.verticalLayout_7.addWidget(self.lp_meshPushButton)
        self.lp_meshTreeView = ParameterTreeWidget(self.lp_compWidget)
        self.lp_meshTreeView.setIndentation(2)
        self.lp_meshTreeView.setObjectName(_fromUtf8("lp_meshTreeView"))
        self.verticalLayout_7.addWidget(self.lp_meshTreeView)
        self.verticalLayout_14.addWidget(self.lp_compWidget)
        self.lp_verticalLayout.addWidget(self.lp_mainWidget)
        self.lp_observeoptionsWidget = QtGui.QWidget(self.lp_DockWidgetContents)
        self.lp_observeoptionsWidget.setMinimumSize(QtCore.QSize(0, 100))
        self.lp_observeoptionsWidget.setObjectName(_fromUtf8("lp_observeoptionsWidget"))
        self.verticalLayout_10 = QtGui.QVBoxLayout(self.lp_observeoptionsWidget)
        self.verticalLayout_10.setSpacing(2)
        self.verticalLayout_10.setMargin(0)
        self.verticalLayout_10.setMargin(0)
        self.verticalLayout_10.setObjectName(_fromUtf8("verticalLayout_10"))
        self.lp_optionsPushButton2 = QtGui.QPushButton(self.lp_observeoptionsWidget)
        self.lp_optionsPushButton2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.lp_optionsPushButton2.setIcon(icon4)
        self.lp_optionsPushButton2.setCheckable(True)
        self.lp_optionsPushButton2.setObjectName(_fromUtf8("lp_optionsPushButton2"))
        self.verticalLayout_10.addWidget(self.lp_optionsPushButton2)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.lp_observeoptionsReset = QtGui.QPushButton(self.lp_observeoptionsWidget)
        self.lp_observeoptionsReset.setEnabled(True)
        self.lp_observeoptionsReset.setMaximumSize(QtCore.QSize(16777215, 24))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/return.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_observeoptionsReset.setIcon(icon5)
        self.lp_observeoptionsReset.setObjectName(_fromUtf8("lp_observeoptionsReset"))
        self.horizontalLayout_6.addWidget(self.lp_observeoptionsReset)
        self.lp_observeoptionsDelete = QtGui.QPushButton(self.lp_observeoptionsWidget)
        self.lp_observeoptionsDelete.setEnabled(True)
        self.lp_observeoptionsDelete.setMaximumSize(QtCore.QSize(16777215, 24))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/bin-3.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_observeoptionsDelete.setIcon(icon6)
        self.lp_observeoptionsDelete.setObjectName(_fromUtf8("lp_observeoptionsDelete"))
        self.horizontalLayout_6.addWidget(self.lp_observeoptionsDelete)
        self.lp_observeoptionsAdd = QtGui.QPushButton(self.lp_observeoptionsWidget)
        self.lp_observeoptionsAdd.setEnabled(True)
        self.lp_observeoptionsAdd.setMaximumSize(QtCore.QSize(16777215, 24))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/add.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_observeoptionsAdd.setIcon(icon7)
        self.lp_observeoptionsAdd.setObjectName(_fromUtf8("lp_observeoptionsAdd"))
        self.horizontalLayout_6.addWidget(self.lp_observeoptionsAdd)
        self.verticalLayout_10.addLayout(self.horizontalLayout_6)
        self.lp_observeoptionsTreeView = ParameterTreeWidget(self.lp_observeoptionsWidget)
        self.lp_observeoptionsTreeView.setIndentation(2)
        self.lp_observeoptionsTreeView.setObjectName(_fromUtf8("lp_observeoptionsTreeView"))
        self.verticalLayout_10.addWidget(self.lp_observeoptionsTreeView)
        self.lp_verticalLayout.addWidget(self.lp_observeoptionsWidget)
        self.lp_spacerWidget = QtGui.QWidget(self.lp_DockWidgetContents)
        self.lp_spacerWidget.setObjectName(_fromUtf8("lp_spacerWidget"))
        self.lp_verticalLayout.addWidget(self.lp_spacerWidget)
        self.lp_progressStackedWidget = QtGui.QStackedWidget(self.lp_DockWidgetContents)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lp_progressStackedWidget.sizePolicy().hasHeightForWidth())
        self.lp_progressStackedWidget.setSizePolicy(sizePolicy)
        self.lp_progressStackedWidget.setMaximumSize(QtCore.QSize(16777215, 24))
        self.lp_progressStackedWidget.setObjectName(_fromUtf8("lp_progressStackedWidget"))
        self.page_5 = QtGui.QWidget()
        self.page_5.setObjectName(_fromUtf8("page_5"))
        self.horizontalLayout_12 = QtGui.QHBoxLayout(self.page_5)
        self.horizontalLayout_12.setSpacing(2)
        self.horizontalLayout_12.setMargin(0)
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.lp_optionsPushButton = QtGui.QPushButton(self.page_5)
        self.lp_optionsPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.lp_optionsPushButton.setText(_fromUtf8(""))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/settings.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_optionsPushButton.setIcon(icon8)
        self.lp_optionsPushButton.setCheckable(True)
        self.lp_optionsPushButton.setObjectName(_fromUtf8("lp_optionsPushButton"))
        self.horizontalLayout_12.addWidget(self.lp_optionsPushButton)
        self.lp_methodComboBox = QtGui.QComboBox(self.page_5)
        self.lp_methodComboBox.setObjectName(_fromUtf8("lp_methodComboBox"))
        self.horizontalLayout_12.addWidget(self.lp_methodComboBox)
        self.lp_serverComboBox = QtGui.QComboBox(self.page_5)
        self.lp_serverComboBox.setObjectName(_fromUtf8("lp_serverComboBox"))
        self.horizontalLayout_12.addWidget(self.lp_serverComboBox)
        self.lp_computePushButton = QtGui.QPushButton(self.page_5)
        self.lp_computePushButton.setMaximumSize(QtCore.QSize(60, 24))
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/play.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_computePushButton.setIcon(icon9)
        self.lp_computePushButton.setObjectName(_fromUtf8("lp_computePushButton"))
        self.horizontalLayout_12.addWidget(self.lp_computePushButton)
        self.lp_progressStackedWidget.addWidget(self.page_5)
        self.page_6 = QtGui.QWidget()
        self.page_6.setObjectName(_fromUtf8("page_6"))
        self.horizontalLayout_13 = QtGui.QHBoxLayout(self.page_6)
        self.horizontalLayout_13.setSpacing(2)
        self.horizontalLayout_13.setMargin(0)
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.lp_progressBar = QtGui.QProgressBar(self.page_6)
        self.lp_progressBar.setMaximumSize(QtCore.QSize(16777215, 24))
        self.lp_progressBar.setProperty("value", 0)
        self.lp_progressBar.setObjectName(_fromUtf8("lp_progressBar"))
        self.horizontalLayout_13.addWidget(self.lp_progressBar)
        self.lp_progressQuit = QtGui.QPushButton(self.page_6)
        self.lp_progressQuit.setMinimumSize(QtCore.QSize(24, 24))
        self.lp_progressQuit.setMaximumSize(QtCore.QSize(24, 24))
        self.lp_progressQuit.setText(_fromUtf8(""))
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/delete.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lp_progressQuit.setIcon(icon10)
        self.lp_progressQuit.setObjectName(_fromUtf8("lp_progressQuit"))
        self.horizontalLayout_13.addWidget(self.lp_progressQuit)
        self.lp_progressStackedWidget.addWidget(self.page_6)
        self.lp_verticalLayout.addWidget(self.lp_progressStackedWidget)
        spacerItem5 = QtGui.QSpacerItem(20, 0, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.lp_verticalLayout.addItem(spacerItem5)
        self.horizontalLayout.addLayout(self.lp_verticalLayout)
        self.lp_DockWidget.setWidget(self.lp_DockWidgetContents)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.lp_DockWidget)
        self.rp_fittingDockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        self.rp_fittingDockWidget.setMinimumSize(QtCore.QSize(371, 503))
        self.rp_fittingDockWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.rp_fittingDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable)
        self.rp_fittingDockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.rp_fittingDockWidget.setObjectName(_fromUtf8("rp_fittingDockWidget"))
        self.rp_DockWidgetContents = QtGui.QWidget()
        self.rp_DockWidgetContents.setObjectName(_fromUtf8("rp_DockWidgetContents"))
        self.gridLayout_3 = QtGui.QGridLayout(self.rp_DockWidgetContents)
        self.gridLayout_3.setSpacing(2)
        self.gridLayout_3.setContentsMargins(2, 2, 2, 0)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.rp_stackedWidget = QtGui.QStackedWidget(self.rp_DockWidgetContents)
        self.rp_stackedWidget.setObjectName(_fromUtf8("rp_stackedWidget"))
        self.page_9 = QtGui.QWidget()
        self.page_9.setObjectName(_fromUtf8("page_9"))
        self.verticalLayout_12 = QtGui.QVBoxLayout(self.page_9)
        self.verticalLayout_12.setSpacing(2)
        self.verticalLayout_12.setMargin(0)
        self.verticalLayout_12.setObjectName(_fromUtf8("verticalLayout_12"))
        self.rp_fitinTreeView = AdjustableTreeWidget(self.page_9)
        self.rp_fitinTreeView.setIndentation(2)
        self.rp_fitinTreeView.setObjectName(_fromUtf8("rp_fitinTreeView"))
        self.verticalLayout_12.addWidget(self.rp_fitinTreeView)
        self.rp_savedFeedbackPushButton = QtGui.QPushButton(self.page_9)
        self.rp_savedFeedbackPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.rp_savedFeedbackPushButton.setIcon(icon4)
        self.rp_savedFeedbackPushButton.setCheckable(True)
        self.rp_savedFeedbackPushButton.setObjectName(_fromUtf8("rp_savedFeedbackPushButton"))
        self.verticalLayout_12.addWidget(self.rp_savedFeedbackPushButton)
        self.rp_savedFeedbackTreeView = FeedbacksTreeWidget(self.page_9)
        self.rp_savedFeedbackTreeView.setEnabled(True)
        self.rp_savedFeedbackTreeView.setIndentation(2)
        self.rp_savedFeedbackTreeView.setObjectName(_fromUtf8("rp_savedFeedbackTreeView"))
        self.rp_savedFeedbackTreeView.header().setVisible(False)
        self.verticalLayout_12.addWidget(self.rp_savedFeedbackTreeView)
        self.rp_savedFeedbackAutoSaveCheck = QtGui.QCheckBox(self.page_9)
        self.rp_savedFeedbackAutoSaveCheck.setEnabled(True)
        self.rp_savedFeedbackAutoSaveCheck.setObjectName(_fromUtf8("rp_savedFeedbackAutoSaveCheck"))
        self.verticalLayout_12.addWidget(self.rp_savedFeedbackAutoSaveCheck)
        self.rp_fitoptionsWidget = QtGui.QWidget(self.page_9)
        self.rp_fitoptionsWidget.setObjectName(_fromUtf8("rp_fitoptionsWidget"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.rp_fitoptionsWidget)
        self.verticalLayout_8.setSpacing(2)
        self.verticalLayout_8.setMargin(0)
        self.verticalLayout_8.setMargin(0)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.rp_optionsPushButton2 = QtGui.QPushButton(self.rp_fitoptionsWidget)
        self.rp_optionsPushButton2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.rp_optionsPushButton2.setIcon(icon4)
        self.rp_optionsPushButton2.setCheckable(True)
        self.rp_optionsPushButton2.setObjectName(_fromUtf8("rp_optionsPushButton2"))
        self.verticalLayout_8.addWidget(self.rp_optionsPushButton2)
        self.horizontalLayout_16 = QtGui.QHBoxLayout()
        self.horizontalLayout_16.setObjectName(_fromUtf8("horizontalLayout_16"))
        self.rp_fitoptionsDelete = QtGui.QPushButton(self.rp_fitoptionsWidget)
        self.rp_fitoptionsDelete.setEnabled(True)
        self.rp_fitoptionsDelete.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_fitoptionsDelete.setIcon(icon6)
        self.rp_fitoptionsDelete.setObjectName(_fromUtf8("rp_fitoptionsDelete"))
        self.horizontalLayout_16.addWidget(self.rp_fitoptionsDelete)
        self.rp_fitoptionsReset = QtGui.QPushButton(self.rp_fitoptionsWidget)
        self.rp_fitoptionsReset.setEnabled(True)
        self.rp_fitoptionsReset.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_fitoptionsReset.setIcon(icon5)
        self.rp_fitoptionsReset.setObjectName(_fromUtf8("rp_fitoptionsReset"))
        self.horizontalLayout_16.addWidget(self.rp_fitoptionsReset)
        self.rp_fitoptionsAdd = QtGui.QPushButton(self.rp_fitoptionsWidget)
        self.rp_fitoptionsAdd.setEnabled(False)
        self.rp_fitoptionsAdd.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_fitoptionsAdd.setIcon(icon7)
        self.rp_fitoptionsAdd.setObjectName(_fromUtf8("rp_fitoptionsAdd"))
        self.horizontalLayout_16.addWidget(self.rp_fitoptionsAdd)
        self.verticalLayout_8.addLayout(self.horizontalLayout_16)
        self.rp_fitoptionsTreeView = ParameterTreeWidget(self.rp_fitoptionsWidget)
        self.rp_fitoptionsTreeView.setMinimumSize(QtCore.QSize(0, 150))
        self.rp_fitoptionsTreeView.setIndentation(2)
        self.rp_fitoptionsTreeView.setObjectName(_fromUtf8("rp_fitoptionsTreeView"))
        self.rp_fitoptionsTreeView.headerItem().setText(1, _fromUtf8("2"))
        self.verticalLayout_8.addWidget(self.rp_fitoptionsTreeView)
        self.verticalLayout_12.addWidget(self.rp_fitoptionsWidget)
        self.rp_progressStackedWidget = QtGui.QStackedWidget(self.page_9)
        self.rp_progressStackedWidget.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_progressStackedWidget.setObjectName(_fromUtf8("rp_progressStackedWidget"))
        self.page_7 = QtGui.QWidget()
        self.page_7.setObjectName(_fromUtf8("page_7"))
        self.horizontalLayout_14 = QtGui.QHBoxLayout(self.page_7)
        self.horizontalLayout_14.setSpacing(2)
        self.horizontalLayout_14.setMargin(0)
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.rp_optionsPushButton = QtGui.QPushButton(self.page_7)
        self.rp_optionsPushButton.setMinimumSize(QtCore.QSize(24, 24))
        self.rp_optionsPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.rp_optionsPushButton.setText(_fromUtf8(""))
        self.rp_optionsPushButton.setIcon(icon8)
        self.rp_optionsPushButton.setCheckable(True)
        self.rp_optionsPushButton.setObjectName(_fromUtf8("rp_optionsPushButton"))
        self.horizontalLayout_14.addWidget(self.rp_optionsPushButton)
        self.rp_methodComboBox = QtGui.QComboBox(self.page_7)
        self.rp_methodComboBox.setMinimumSize(QtCore.QSize(0, 24))
        self.rp_methodComboBox.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_methodComboBox.setObjectName(_fromUtf8("rp_methodComboBox"))
        self.horizontalLayout_14.addWidget(self.rp_methodComboBox)
        self.rp_serverComboBox = QtGui.QComboBox(self.page_7)
        self.rp_serverComboBox.setObjectName(_fromUtf8("rp_serverComboBox"))
        self.horizontalLayout_14.addWidget(self.rp_serverComboBox)
        self.rp_fitPushButton = QtGui.QPushButton(self.page_7)
        self.rp_fitPushButton.setEnabled(True)
        self.rp_fitPushButton.setMinimumSize(QtCore.QSize(24, 24))
        self.rp_fitPushButton.setMaximumSize(QtCore.QSize(60, 24))
        self.rp_fitPushButton.setIcon(icon9)
        self.rp_fitPushButton.setObjectName(_fromUtf8("rp_fitPushButton"))
        self.horizontalLayout_14.addWidget(self.rp_fitPushButton)
        self.rp_progressStackedWidget.addWidget(self.page_7)
        self.page_8 = QtGui.QWidget()
        self.page_8.setObjectName(_fromUtf8("page_8"))
        self.horizontalLayout_15 = QtGui.QHBoxLayout(self.page_8)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setMargin(0)
        self.horizontalLayout_15.setObjectName(_fromUtf8("horizontalLayout_15"))
        self.rp_progressBar = QtGui.QProgressBar(self.page_8)
        self.rp_progressBar.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_progressBar.setProperty("value", 0)
        self.rp_progressBar.setObjectName(_fromUtf8("rp_progressBar"))
        self.horizontalLayout_15.addWidget(self.rp_progressBar)
        self.rp_progressStackedWidget.addWidget(self.page_8)
        self.verticalLayout_12.addWidget(self.rp_progressStackedWidget)
        self.rp_stackedWidget.addWidget(self.page_9)
        self.page_10 = QtGui.QWidget()
        self.page_10.setObjectName(_fromUtf8("page_10"))
        self.verticalLayout_13 = QtGui.QVBoxLayout(self.page_10)
        self.verticalLayout_13.setSpacing(2)
        self.verticalLayout_13.setMargin(0)
        self.verticalLayout_13.setObjectName(_fromUtf8("verticalLayout_13"))
        self.rp_fitoutTreeView = FittingTreeWidget(self.page_10)
        self.rp_fitoutTreeView.setIndentation(2)
        self.rp_fitoutTreeView.setObjectName(_fromUtf8("rp_fitoutTreeView"))
        self.verticalLayout_13.addWidget(self.rp_fitoutTreeView)
        self.pushButton_4 = QtGui.QPushButton(self.page_10)
        self.pushButton_4.setEnabled(False)
        self.pushButton_4.setMaximumSize(QtCore.QSize(16777215, 24))
        self.pushButton_4.setIcon(icon7)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.verticalLayout_13.addWidget(self.pushButton_4)
        self.gridLayout_15 = QtGui.QGridLayout()
        self.gridLayout_15.setObjectName(_fromUtf8("gridLayout_15"))
        self.rp_rejectPushButton = QtGui.QPushButton(self.page_10)
        self.rp_rejectPushButton.setEnabled(True)
        self.rp_rejectPushButton.setMinimumSize(QtCore.QSize(0, 24))
        self.rp_rejectPushButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.rp_rejectPushButton.setIcon(icon6)
        self.rp_rejectPushButton.setObjectName(_fromUtf8("rp_rejectPushButton"))
        self.gridLayout_15.addWidget(self.rp_rejectPushButton, 3, 1, 1, 1)
        self.rp_acceptPushButton = QtGui.QPushButton(self.page_10)
        self.rp_acceptPushButton.setEnabled(True)
        self.rp_acceptPushButton.setMinimumSize(QtCore.QSize(0, 24))
        self.rp_acceptPushButton.setMaximumSize(QtCore.QSize(16777215, 24))
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/commit.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rp_acceptPushButton.setIcon(icon11)
        self.rp_acceptPushButton.setObjectName(_fromUtf8("rp_acceptPushButton"))
        self.gridLayout_15.addWidget(self.rp_acceptPushButton, 3, 3, 1, 1)
        self.pushButton_2 = QtGui.QPushButton(self.page_10)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setMaximumSize(QtCore.QSize(16777215, 24))
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/eye.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon12)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.gridLayout_15.addWidget(self.pushButton_2, 2, 3, 1, 1)
        self.pushButton_3 = QtGui.QPushButton(self.page_10)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setMaximumSize(QtCore.QSize(16777215, 24))
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/chart.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon13)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.gridLayout_15.addWidget(self.pushButton_3, 2, 1, 1, 1)
        self.verticalLayout_13.addLayout(self.gridLayout_15)
        self.rp_stackedWidget.addWidget(self.page_10)
        self.gridLayout_3.addWidget(self.rp_stackedWidget, 0, 0, 1, 1)
        self.rp_fittingDockWidget.setWidget(self.rp_DockWidgetContents)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.rp_fittingDockWidget)
        self.bp_pyDockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bp_pyDockWidget.sizePolicy().hasHeightForWidth())
        self.bp_pyDockWidget.setSizePolicy(sizePolicy)
        self.bp_pyDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.bp_pyDockWidget.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.bp_pyDockWidget.setObjectName(_fromUtf8("bp_pyDockWidget"))
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.gridLayout_13 = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout_13.setContentsMargins(6, 2, 6, 2)
        self.gridLayout_13.setObjectName(_fromUtf8("gridLayout_13"))
        self.bp_pyLayout = QtGui.QGridLayout()
        self.bp_pyLayout.setSpacing(0)
        self.bp_pyLayout.setObjectName(_fromUtf8("bp_pyLayout"))
        self.gridLayout_13.addLayout(self.bp_pyLayout, 0, 0, 1, 1)
        self.bp_pyDockWidget.setWidget(self.dockWidgetContents)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.bp_pyDockWidget)
        self.bp_datasetsDockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        self.bp_datasetsDockWidget.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self.bp_datasetsDockWidget.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea|QtCore.Qt.TopDockWidgetArea)
        self.bp_datasetsDockWidget.setObjectName(_fromUtf8("bp_datasetsDockWidget"))
        self.dockWidgetContents_3 = QtGui.QWidget()
        self.dockWidgetContents_3.setObjectName(_fromUtf8("dockWidgetContents_3"))
        self.bp_datasetsDockWidget.setWidget(self.dockWidgetContents_3)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.bp_datasetsDockWidget)
        self.rp_versionsDockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        self.rp_versionsDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.rp_versionsDockWidget.setObjectName(_fromUtf8("rp_versionsDockWidget"))
        self.dockWidgetContents_2 = QtGui.QWidget()
        self.dockWidgetContents_2.setObjectName(_fromUtf8("dockWidgetContents_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.dockWidgetContents_2)
        self.gridLayout_2.setSpacing(2)
        self.gridLayout_2.setContentsMargins(2, 2, 2, 0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.versions_oncompute = QtGui.QCheckBox(self.dockWidgetContents_2)
        self.versions_oncompute.setEnabled(True)
        self.versions_oncompute.setObjectName(_fromUtf8("versions_oncompute"))
        self.gridLayout_2.addWidget(self.versions_oncompute, 2, 0, 1, 1)
        self.versions_addnow = QtGui.QPushButton(self.dockWidgetContents_2)
        self.versions_addnow.setEnabled(True)
        self.versions_addnow.setIcon(icon7)
        self.versions_addnow.setObjectName(_fromUtf8("versions_addnow"))
        self.gridLayout_2.addWidget(self.versions_addnow, 2, 1, 1, 1)
        self.versions_treeView = VersionsTreeWidget(self.dockWidgetContents_2)
        self.versions_treeView.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.versions_treeView.sizePolicy().hasHeightForWidth())
        self.versions_treeView.setSizePolicy(sizePolicy)
        self.versions_treeView.setIndentation(2)
        self.versions_treeView.setObjectName(_fromUtf8("versions_treeView"))
        self.versions_treeView.header().setVisible(False)
        self.gridLayout_2.addWidget(self.versions_treeView, 0, 0, 1, 2)
        self.rp_versionsDockWidget.setWidget(self.dockWidgetContents_2)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.rp_versionsDockWidget)
        self.lp_systemDockWidget = QtGui.QDockWidget(PHOEBE_MainWindow)
        self.lp_systemDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.lp_systemDockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.lp_systemDockWidget.setObjectName(_fromUtf8("lp_systemDockWidget"))
        self.dockWidgetContents_4 = QtGui.QWidget()
        self.dockWidgetContents_4.setObjectName(_fromUtf8("dockWidgetContents_4"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.dockWidgetContents_4)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.lp_sysHierarchyPushButton = QtGui.QPushButton(self.dockWidgetContents_4)
        self.lp_sysHierarchyPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.lp_sysHierarchyPushButton.setIcon(icon4)
        self.lp_sysHierarchyPushButton.setCheckable(True)
        self.lp_sysHierarchyPushButton.setChecked(False)
        self.lp_sysHierarchyPushButton.setObjectName(_fromUtf8("lp_sysHierarchyPushButton"))
        self.verticalLayout_2.addWidget(self.lp_sysHierarchyPushButton)
        self.mp_systemSelectWebViewWidget = QtGui.QWidget(self.dockWidgetContents_4)
        self.mp_systemSelectWebViewWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.mp_systemSelectWebViewWidget.setObjectName(_fromUtf8("mp_systemSelectWebViewWidget"))
        self.verticalLayout_2.addWidget(self.mp_systemSelectWebViewWidget)
        self.mp_sysOrbitPushButton = QtGui.QPushButton(self.dockWidgetContents_4)
        self.mp_sysOrbitPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.mp_sysOrbitPushButton.setIcon(icon4)
        self.mp_sysOrbitPushButton.setCheckable(True)
        self.mp_sysOrbitPushButton.setChecked(True)
        self.mp_sysOrbitPushButton.setObjectName(_fromUtf8("mp_sysOrbitPushButton"))
        self.verticalLayout_2.addWidget(self.mp_sysOrbitPushButton)
        self.mp_sysorbitWidget = QtGui.QWidget(self.dockWidgetContents_4)
        self.mp_sysorbitWidget.setObjectName(_fromUtf8("mp_sysorbitWidget"))
        self.verticalLayout_15 = QtGui.QVBoxLayout(self.mp_sysorbitWidget)
        self.verticalLayout_15.setSpacing(2)
        self.verticalLayout_15.setMargin(0)
        self.verticalLayout_15.setMargin(0)
        self.verticalLayout_15.setObjectName(_fromUtf8("verticalLayout_15"))
        self.sys_orbitmplGridLayout = QtGui.QGridLayout()
        self.sys_orbitmplGridLayout.setObjectName(_fromUtf8("sys_orbitmplGridLayout"))
        self.verticalLayout_15.addLayout(self.sys_orbitmplGridLayout)
        self.sys_orbitOptionsWidget = QtGui.QWidget(self.mp_sysorbitWidget)
        self.sys_orbitOptionsWidget.setObjectName(_fromUtf8("sys_orbitOptionsWidget"))
        self.verticalLayout_17 = QtGui.QVBoxLayout(self.sys_orbitOptionsWidget)
        self.verticalLayout_17.setSpacing(2)
        self.verticalLayout_17.setMargin(0)
        self.verticalLayout_17.setMargin(0)
        self.verticalLayout_17.setObjectName(_fromUtf8("verticalLayout_17"))
        self.sys_orbitOptionsPushButton2 = QtGui.QPushButton(self.sys_orbitOptionsWidget)
        self.sys_orbitOptionsPushButton2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.sys_orbitOptionsPushButton2.setIcon(icon4)
        self.sys_orbitOptionsPushButton2.setCheckable(True)
        self.sys_orbitOptionsPushButton2.setObjectName(_fromUtf8("sys_orbitOptionsPushButton2"))
        self.verticalLayout_17.addWidget(self.sys_orbitOptionsPushButton2)
        self.sys_orbitOptionsTreeView = ParameterTreeWidget(self.sys_orbitOptionsWidget)
        self.sys_orbitOptionsTreeView.setIndentation(2)
        self.sys_orbitOptionsTreeView.setObjectName(_fromUtf8("sys_orbitOptionsTreeView"))
        self.sys_orbitOptionsTreeView.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout_17.addWidget(self.sys_orbitOptionsTreeView)
        self.verticalLayout_15.addWidget(self.sys_orbitOptionsWidget)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(2)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.sys_orbitOptionsPushButton = QtGui.QPushButton(self.mp_sysorbitWidget)
        self.sys_orbitOptionsPushButton.setEnabled(True)
        self.sys_orbitOptionsPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.sys_orbitOptionsPushButton.setText(_fromUtf8(""))
        self.sys_orbitOptionsPushButton.setIcon(icon8)
        self.sys_orbitOptionsPushButton.setCheckable(True)
        self.sys_orbitOptionsPushButton.setObjectName(_fromUtf8("sys_orbitOptionsPushButton"))
        self.horizontalLayout_5.addWidget(self.sys_orbitOptionsPushButton)
        self.sys_orbitPushButton = QtGui.QPushButton(self.mp_sysorbitWidget)
        self.sys_orbitPushButton.setEnabled(True)
        self.sys_orbitPushButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.sys_orbitPushButton.setIcon(icon9)
        self.sys_orbitPushButton.setObjectName(_fromUtf8("sys_orbitPushButton"))
        self.horizontalLayout_5.addWidget(self.sys_orbitPushButton)
        self.verticalLayout_15.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.addWidget(self.mp_sysorbitWidget)
        self.mp_sysMeshPushButton = QtGui.QPushButton(self.dockWidgetContents_4)
        self.mp_sysMeshPushButton.setMaximumSize(QtCore.QSize(16777215, 20))
        self.mp_sysMeshPushButton.setIcon(icon4)
        self.mp_sysMeshPushButton.setCheckable(True)
        self.mp_sysMeshPushButton.setChecked(True)
        self.mp_sysMeshPushButton.setObjectName(_fromUtf8("mp_sysMeshPushButton"))
        self.verticalLayout_2.addWidget(self.mp_sysMeshPushButton)
        self.mp_sysmplWidget = QtGui.QWidget(self.dockWidgetContents_4)
        self.mp_sysmplWidget.setMinimumSize(QtCore.QSize(0, 150))
        self.mp_sysmplWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.mp_sysmplWidget.setObjectName(_fromUtf8("mp_sysmplWidget"))
        self.verticalLayout_9 = QtGui.QVBoxLayout(self.mp_sysmplWidget)
        self.verticalLayout_9.setSpacing(2)
        self.verticalLayout_9.setMargin(0)
        self.verticalLayout_9.setMargin(0)
        self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
        self.mp_sysmplGridLayout = QtGui.QGridLayout()
        self.mp_sysmplGridLayout.setObjectName(_fromUtf8("mp_sysmplGridLayout"))
        self.verticalLayout_9.addLayout(self.mp_sysmplGridLayout)
        self.sys_meshOptionsWidget = QtGui.QWidget(self.mp_sysmplWidget)
        self.sys_meshOptionsWidget.setObjectName(_fromUtf8("sys_meshOptionsWidget"))
        self.verticalLayout_16 = QtGui.QVBoxLayout(self.sys_meshOptionsWidget)
        self.verticalLayout_16.setSpacing(2)
        self.verticalLayout_16.setMargin(0)
        self.verticalLayout_16.setMargin(0)
        self.verticalLayout_16.setObjectName(_fromUtf8("verticalLayout_16"))
        self.sys_meshOptionsPushButton2 = QtGui.QPushButton(self.sys_meshOptionsWidget)
        self.sys_meshOptionsPushButton2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.sys_meshOptionsPushButton2.setIcon(icon4)
        self.sys_meshOptionsPushButton2.setCheckable(True)
        self.sys_meshOptionsPushButton2.setObjectName(_fromUtf8("sys_meshOptionsPushButton2"))
        self.verticalLayout_16.addWidget(self.sys_meshOptionsPushButton2)
        self.sys_meshOptionsTreeView = ParameterTreeWidget(self.sys_meshOptionsWidget)
        self.sys_meshOptionsTreeView.setIndentation(2)
        self.sys_meshOptionsTreeView.setObjectName(_fromUtf8("sys_meshOptionsTreeView"))
        self.sys_meshOptionsTreeView.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout_16.addWidget(self.sys_meshOptionsTreeView)
        self.sys_meshAutoUpdate = QtGui.QCheckBox(self.sys_meshOptionsWidget)
        self.sys_meshAutoUpdate.setEnabled(True)
        self.sys_meshAutoUpdate.setMaximumSize(QtCore.QSize(80, 16777215))
        self.sys_meshAutoUpdate.setObjectName(_fromUtf8("sys_meshAutoUpdate"))
        self.verticalLayout_16.addWidget(self.sys_meshAutoUpdate)
        self.verticalLayout_9.addWidget(self.sys_meshOptionsWidget)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(2)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.sys_meshOptionsPushButton = QtGui.QPushButton(self.mp_sysmplWidget)
        self.sys_meshOptionsPushButton.setEnabled(True)
        self.sys_meshOptionsPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.sys_meshOptionsPushButton.setText(_fromUtf8(""))
        self.sys_meshOptionsPushButton.setIcon(icon8)
        self.sys_meshOptionsPushButton.setCheckable(True)
        self.sys_meshOptionsPushButton.setObjectName(_fromUtf8("sys_meshOptionsPushButton"))
        self.horizontalLayout_3.addWidget(self.sys_meshOptionsPushButton)
        self.sys_meshPushButton = QtGui.QPushButton(self.mp_sysmplWidget)
        self.sys_meshPushButton.setEnabled(True)
        self.sys_meshPushButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.sys_meshPushButton.setIcon(icon9)
        self.sys_meshPushButton.setObjectName(_fromUtf8("sys_meshPushButton"))
        self.horizontalLayout_3.addWidget(self.sys_meshPushButton)
        self.verticalLayout_9.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.mp_sysmplWidget)
        self.lp_systemDockWidget.setWidget(self.dockWidgetContents_4)
        PHOEBE_MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.lp_systemDockWidget)
        self.tb_file_newAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_newAction.setEnabled(True)
        self.tb_file_newAction.setObjectName(_fromUtf8("tb_file_newAction"))
        self.tb_file_openAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_openAction.setEnabled(True)
        self.tb_file_openAction.setObjectName(_fromUtf8("tb_file_openAction"))
        self.tb_file_saveAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_saveAction.setEnabled(False)
        self.tb_file_saveAction.setObjectName(_fromUtf8("tb_file_saveAction"))
        self.tb_file_saveasAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_saveasAction.setEnabled(True)
        self.tb_file_saveasAction.setObjectName(_fromUtf8("tb_file_saveasAction"))
        self.tb_file_exportAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_exportAction.setEnabled(False)
        self.tb_file_exportAction.setObjectName(_fromUtf8("tb_file_exportAction"))
        self.tb_file_quitAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_file_quitAction.setObjectName(_fromUtf8("tb_file_quitAction"))
        self.tb_edit_prefsAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_edit_prefsAction.setEnabled(True)
        self.tb_edit_prefsAction.setObjectName(_fromUtf8("tb_edit_prefsAction"))
        self.tb_view_rpAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_rpAction.setCheckable(True)
        self.tb_view_rpAction.setChecked(True)
        self.tb_view_rpAction.setObjectName(_fromUtf8("tb_view_rpAction"))
        self.tb_view_datasetsAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_datasetsAction.setCheckable(True)
        self.tb_view_datasetsAction.setChecked(True)
        self.tb_view_datasetsAction.setObjectName(_fromUtf8("tb_view_datasetsAction"))
        self.tb_help_aboutAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_help_aboutAction.setEnabled(True)
        self.tb_help_aboutAction.setObjectName(_fromUtf8("tb_help_aboutAction"))
        self.tb_help_helpAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_help_helpAction.setEnabled(True)
        self.tb_help_helpAction.setObjectName(_fromUtf8("tb_help_helpAction"))
        self.actionObject_Tree = QtGui.QAction(PHOEBE_MainWindow)
        self.actionObject_Tree.setCheckable(True)
        self.actionObject_Tree.setChecked(True)
        self.actionObject_Tree.setObjectName(_fromUtf8("actionObject_Tree"))
        self.actionParameters_2 = QtGui.QAction(PHOEBE_MainWindow)
        self.actionParameters_2.setCheckable(True)
        self.actionParameters_2.setChecked(True)
        self.actionParameters_2.setObjectName(_fromUtf8("actionParameters_2"))
        self.actionObject_Schematic = QtGui.QAction(PHOEBE_MainWindow)
        self.actionObject_Schematic.setCheckable(True)
        self.actionObject_Schematic.setObjectName(_fromUtf8("actionObject_Schematic"))
        self.tb_view_lpAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_lpAction.setCheckable(True)
        self.tb_view_lpAction.setChecked(True)
        self.tb_view_lpAction.setObjectName(_fromUtf8("tb_view_lpAction"))
        self.tb_view_hierAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_hierAction.setCheckable(True)
        self.tb_view_hierAction.setChecked(True)
        self.tb_view_hierAction.setObjectName(_fromUtf8("tb_view_hierAction"))
        self.tb_view_schemAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_schemAction.setCheckable(True)
        self.tb_view_schemAction.setChecked(True)
        self.tb_view_schemAction.setEnabled(True)
        self.tb_view_schemAction.setObjectName(_fromUtf8("tb_view_schemAction"))
        self.tb_view_newobjAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_newobjAction.setCheckable(True)
        self.tb_view_newobjAction.setEnabled(True)
        self.tb_view_newobjAction.setObjectName(_fromUtf8("tb_view_newobjAction"))
        self.tb_edit_undoAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_edit_undoAction.setEnabled(True)
        self.tb_edit_undoAction.setVisible(True)
        self.tb_edit_undoAction.setObjectName(_fromUtf8("tb_edit_undoAction"))
        self.tb_edit_redoAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_edit_redoAction.setEnabled(True)
        self.tb_edit_redoAction.setVisible(True)
        self.tb_edit_redoAction.setObjectName(_fromUtf8("tb_edit_redoAction"))
        self.tb_view_sysAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_sysAction.setCheckable(True)
        self.tb_view_sysAction.setChecked(True)
        self.tb_view_sysAction.setObjectName(_fromUtf8("tb_view_sysAction"))
        self.tb_view_glAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_glAction.setCheckable(True)
        self.tb_view_glAction.setChecked(True)
        self.tb_view_glAction.setObjectName(_fromUtf8("tb_view_glAction"))
        self.actionLegacy_PHOEBE = QtGui.QAction(PHOEBE_MainWindow)
        self.actionLegacy_PHOEBE.setEnabled(True)
        self.actionLegacy_PHOEBE.setObjectName(_fromUtf8("actionLegacy_PHOEBE"))
        self.tb_fileImport_libraryAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_fileImport_libraryAction.setObjectName(_fromUtf8("tb_fileImport_libraryAction"))
        self.actionBM3 = QtGui.QAction(PHOEBE_MainWindow)
        self.actionBM3.setEnabled(False)
        self.actionBM3.setObjectName(_fromUtf8("actionBM3"))
        self.actionWD_active = QtGui.QAction(PHOEBE_MainWindow)
        self.actionWD_active.setEnabled(False)
        self.actionWD_active.setObjectName(_fromUtf8("actionWD_active"))
        self.actionKeplerEB_Catalog = QtGui.QAction(PHOEBE_MainWindow)
        self.actionKeplerEB_Catalog.setEnabled(False)
        self.actionKeplerEB_Catalog.setObjectName(_fromUtf8("actionKeplerEB_Catalog"))
        self.tb_tools_pluginAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_tools_pluginAction.setObjectName(_fromUtf8("tb_tools_pluginAction"))
        self.tb_view_plotsAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_plotsAction.setCheckable(True)
        self.tb_view_plotsAction.setChecked(True)
        self.tb_view_plotsAction.setObjectName(_fromUtf8("tb_view_plotsAction"))
        self.tb_view_bpPlotsAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_bpPlotsAction.setCheckable(True)
        self.tb_view_bpPlotsAction.setChecked(True)
        self.tb_view_bpPlotsAction.setObjectName(_fromUtf8("tb_view_bpPlotsAction"))
        self.tb_tools_scriptAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_tools_scriptAction.setEnabled(False)
        self.tb_tools_scriptAction.setObjectName(_fromUtf8("tb_tools_scriptAction"))
        self.tb_view_versionsAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_versionsAction.setCheckable(True)
        self.tb_view_versionsAction.setChecked(False)
        self.tb_view_versionsAction.setObjectName(_fromUtf8("tb_view_versionsAction"))
        self.tb_view_pythonAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_pythonAction.setCheckable(True)
        self.tb_view_pythonAction.setChecked(True)
        self.tb_view_pythonAction.setObjectName(_fromUtf8("tb_view_pythonAction"))
        self.tb_view_systemAction = QtGui.QAction(PHOEBE_MainWindow)
        self.tb_view_systemAction.setCheckable(True)
        self.tb_view_systemAction.setChecked(False)
        self.tb_view_systemAction.setObjectName(_fromUtf8("tb_view_systemAction"))
        self.menuImport.addAction(self.tb_fileImport_libraryAction)
        self.menuImport.addAction(self.actionLegacy_PHOEBE)
        self.menuImport.addAction(self.actionBM3)
        self.menuImport.addAction(self.actionWD_active)
        self.tb_fileMenu.addAction(self.tb_file_newAction)
        self.tb_fileMenu.addAction(self.tb_file_openAction)
        self.tb_fileMenu.addAction(self.menuImport.menuAction())
        self.tb_fileMenu.addAction(self.tb_file_saveAction)
        self.tb_fileMenu.addAction(self.tb_file_saveasAction)
        self.tb_fileMenu.addSeparator()
        self.tb_fileMenu.addAction(self.tb_file_exportAction)
        self.tb_fileMenu.addSeparator()
        self.tb_fileMenu.addAction(self.tb_file_quitAction)
        self.tb_editMenu.addAction(self.tb_edit_undoAction)
        self.tb_editMenu.addAction(self.tb_edit_redoAction)
        self.tb_editMenu.addSeparator()
        self.tb_editMenu.addAction(self.tb_edit_prefsAction)
        self.tb_viewMenu.addSeparator()
        self.tb_viewMenu.addAction(self.tb_view_systemAction)
        self.tb_viewMenu.addAction(self.tb_view_lpAction)
        self.tb_viewMenu.addAction(self.tb_view_rpAction)
        self.tb_viewMenu.addAction(self.tb_view_versionsAction)
        self.tb_viewMenu.addAction(self.tb_view_datasetsAction)
        self.tb_viewMenu.addAction(self.tb_view_pythonAction)
        self.tb_helpMenu.addAction(self.tb_help_aboutAction)
        self.tb_helpMenu.addAction(self.tb_help_helpAction)
        self.tb_advancedMenu.addAction(self.tb_tools_scriptAction)
        self.tb_MenuBar.addAction(self.tb_fileMenu.menuAction())
        self.tb_MenuBar.addAction(self.tb_editMenu.menuAction())
        self.tb_MenuBar.addAction(self.tb_viewMenu.menuAction())
        self.tb_MenuBar.addAction(self.tb_advancedMenu.menuAction())
        self.tb_MenuBar.addAction(self.tb_helpMenu.menuAction())

        self.retranslateUi(PHOEBE_MainWindow)
        self.mp_stackedWidget.setCurrentIndex(1)
        self.lp_progressStackedWidget.setCurrentIndex(0)
        self.rp_stackedWidget.setCurrentIndex(0)
        self.rp_progressStackedWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.tb_view_rpAction, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_fittingDockWidget.setVisible)
        QtCore.QObject.connect(self.tb_view_lpAction, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_DockWidget.setVisible)
        QtCore.QObject.connect(self.tb_file_quitAction, QtCore.SIGNAL(_fromUtf8("activated()")), PHOEBE_MainWindow.close)
        QtCore.QObject.connect(self.lp_orbitPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_orbitTreeView.setVisible)
        QtCore.QObject.connect(self.lp_compPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_compWidget.setVisible)
        QtCore.QObject.connect(self.tb_view_glAction, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.mp_sysmplWidget.setVisible)
        QtCore.QObject.connect(self.lp_meshPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_meshTreeView.setVisible)
        QtCore.QObject.connect(self.lp_optionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_observeoptionsWidget.setVisible)
        QtCore.QObject.connect(self.lp_optionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_optionsPushButton2.setChecked)
        QtCore.QObject.connect(self.lp_optionsPushButton2, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_optionsPushButton.setChecked)
        QtCore.QObject.connect(self.rp_optionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_optionsPushButton2.setChecked)
        QtCore.QObject.connect(self.rp_optionsPushButton2, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_optionsPushButton.setChecked)
        QtCore.QObject.connect(self.rp_optionsPushButton2, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_fitoptionsWidget.setVisible)
        QtCore.QObject.connect(self.rp_savedFeedbackPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_savedFeedbackTreeView.setVisible)
        QtCore.QObject.connect(self.rp_savedFeedbackPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_savedFeedbackAutoSaveCheck.setVisible)
        QtCore.QObject.connect(self.tb_view_systemAction, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.lp_systemDockWidget.setVisible)
        QtCore.QObject.connect(self.tb_view_versionsAction, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.rp_versionsDockWidget.setVisible)
        QtCore.QObject.connect(self.lp_sysHierarchyPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.mp_systemSelectWebViewWidget.setVisible)
        QtCore.QObject.connect(self.mp_sysOrbitPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.mp_sysorbitWidget.setVisible)
        QtCore.QObject.connect(self.mp_sysMeshPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.mp_sysmplWidget.setVisible)
        QtCore.QObject.connect(self.sys_meshOptionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_meshOptionsWidget.setVisible)
        QtCore.QObject.connect(self.sys_meshOptionsPushButton2, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_meshOptionsPushButton.setChecked)
        QtCore.QObject.connect(self.sys_meshOptionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_meshOptionsPushButton2.setChecked)
        QtCore.QObject.connect(self.sys_orbitOptionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_orbitOptionsWidget.setVisible)
        QtCore.QObject.connect(self.sys_orbitOptionsPushButton, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_orbitOptionsPushButton2.setChecked)
        QtCore.QObject.connect(self.sys_orbitOptionsPushButton2, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.sys_orbitOptionsPushButton.setChecked)
        QtCore.QMetaObject.connectSlotsByName(PHOEBE_MainWindow)

    def retranslateUi(self, PHOEBE_MainWindow):
        PHOEBE_MainWindow.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "PHOEBE", None, QtGui.QApplication.UnicodeUTF8))
        self.mp_splash_binaryPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Create New Roche Binary", None, QtGui.QApplication.UnicodeUTF8))
        self.mp_splash_triplePushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Create New Hierarchical Triple System", None, QtGui.QApplication.UnicodeUTF8))
        self.mp_splash_openPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Open PHOEBE File", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/images/phoebe-gui.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_5.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "PushButton", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_fileMenu.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuImport.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Import", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_editMenu.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Edit", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_viewMenu.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "View", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_helpMenu.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_advancedMenu.setTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Tools", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_DockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_orbitPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Orbital Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_orbitTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_orbitTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_compPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Component Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_compTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_compTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_meshPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Mesh Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_meshTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_meshTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_optionsPushButton2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Compute Options", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsReset.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "reset to default (from preferences)", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsReset.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Reset", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsDelete.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Delete", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsAdd.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Create New", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Preview", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_observeoptionsTreeView.headerItem().setText(2, QtGui.QApplication.translate("PHOEBE_MainWindow", "Compute", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_computePushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Run", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_progressQuit.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "cancel thread", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fittingDockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitinTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Adjust", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitinTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitinTreeView.headerItem().setText(2, QtGui.QApplication.translate("PHOEBE_MainWindow", "Prior", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_savedFeedbackPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Saved Feedback", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_savedFeedbackTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Feedback Name", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_savedFeedbackAutoSaveCheck.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "auto save feedback", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_optionsPushButton2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Fitting Options", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoptionsDelete.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Delete", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoptionsReset.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "reset to default (from preferences)", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoptionsReset.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Reset", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoptionsAdd.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Create New", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoptionsTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_optionsPushButton.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "view options for the current fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitPushButton.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "run the selected fitting routine", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Fit", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoutTreeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoutTreeView.headerItem().setText(1, QtGui.QApplication.translate("PHOEBE_MainWindow", "Old", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoutTreeView.headerItem().setText(2, QtGui.QApplication.translate("PHOEBE_MainWindow", "New", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_fitoutTreeView.headerItem().setText(3, QtGui.QApplication.translate("PHOEBE_MainWindow", "Error", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_4.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "save this feedback to the bundle for future review/reference", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_4.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Save Feedback", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_rejectPushButton.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "reject the proposed parameters/model", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_rejectPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Reject", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_acceptPushButton.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "accept the proposed parameters/model", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_acceptPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Accept", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_2.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "view correlation and fitting statistics", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Examine", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_3.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "overplot the proposed model on the current model", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_3.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Overplot", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_pyDockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Console", None, QtGui.QApplication.UnicodeUTF8))
        self.bp_datasetsDockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.rp_versionsDockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "Versions", None, QtGui.QApplication.UnicodeUTF8))
        self.versions_oncompute.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "auto add after Preview/Compute", None, QtGui.QApplication.UnicodeUTF8))
        self.versions_oncompute.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "auto add", None, QtGui.QApplication.UnicodeUTF8))
        self.versions_addnow.setStatusTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "create a backup of the current version", None, QtGui.QApplication.UnicodeUTF8))
        self.versions_addnow.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Add Version", None, QtGui.QApplication.UnicodeUTF8))
        self.versions_treeView.headerItem().setText(0, QtGui.QApplication.translate("PHOEBE_MainWindow", "Version Name", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_systemDockWidget.setWindowTitle(QtGui.QApplication.translate("PHOEBE_MainWindow", "System", None, QtGui.QApplication.UnicodeUTF8))
        self.lp_sysHierarchyPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Hierarchy", None, QtGui.QApplication.UnicodeUTF8))
        self.mp_sysOrbitPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Orbit View", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_orbitOptionsPushButton2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Orbit View Options", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_orbitPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Update", None, QtGui.QApplication.UnicodeUTF8))
        self.mp_sysMeshPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Mesh View", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_meshOptionsPushButton2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Mesh View Options", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_meshAutoUpdate.setToolTip(QtGui.QApplication.translate("PHOEBE_MainWindow", "update mesh plot whenever the select time is changed", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_meshAutoUpdate.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "auto", None, QtGui.QApplication.UnicodeUTF8))
        self.sys_meshPushButton.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Update", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_newAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "New", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_newAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+N", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_openAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Open...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_openAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+O", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_saveAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_saveAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+S", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_saveasAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Save As...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_saveasAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Shift+S", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_exportAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Export...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_quitAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_file_quitAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Q", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_prefsAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Preferences", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_prefsAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Alt+P", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_rpAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_rpAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+3", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_datasetsAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Data and Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_datasetsAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+5", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_help_aboutAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "About Phoebe", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_help_helpAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Phoebe Help", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_help_helpAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "F1", None, QtGui.QApplication.UnicodeUTF8))
        self.actionObject_Tree.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Object Tree", None, QtGui.QApplication.UnicodeUTF8))
        self.actionParameters_2.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.actionObject_Schematic.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Object Schematic", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_lpAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_lpAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+2", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_hierAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Hierarchical", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_schemAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Schematic", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_newobjAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "New Objects", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_undoAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Undo", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_undoAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Z", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_redoAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Redo", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_edit_redoAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Shift+Z", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_sysAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "System View", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_sysAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+8", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_glAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "3D View", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_glAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+9", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLegacy_PHOEBE.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Legacy PHOEBE...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_fileImport_libraryAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "From Library...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_fileImport_libraryAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+Shift+L", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBM3.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "BM3...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionWD_active.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "WD .active...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionKeplerEB_Catalog.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "KeplerEB Catalog...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_tools_pluginAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Plugin Manager", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_plotsAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Middle Pane", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_plotsAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+7", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_bpPlotsAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_tools_scriptAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Run Python Script...", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_versionsAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Versions", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_versionsAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+4", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_pythonAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "Python Interface", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_pythonAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+6", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_systemAction.setText(QtGui.QApplication.translate("PHOEBE_MainWindow", "System", None, QtGui.QApplication.UnicodeUTF8))
        self.tb_view_systemAction.setShortcut(QtGui.QApplication.translate("PHOEBE_MainWindow", "Ctrl+1", None, QtGui.QApplication.UnicodeUTF8))

from phoebe_widgets import FittingTreeWidget, AdjustableTreeWidget, ParameterTreeWidget, FeedbacksTreeWidget, VersionsTreeWidget

class Ui_popAbout_Dialog(object):
    def setupUi(self, popAbout_Dialog):
        popAbout_Dialog.setObjectName(_fromUtf8("popAbout_Dialog"))
        popAbout_Dialog.resize(400, 300)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(popAbout_Dialog.sizePolicy().hasHeightForWidth())
        popAbout_Dialog.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QtGui.QGridLayout(popAbout_Dialog)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.phoebeLogo_label = QtGui.QLabel(popAbout_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phoebeLogo_label.sizePolicy().hasHeightForWidth())
        self.phoebeLogo_label.setSizePolicy(sizePolicy)
        self.phoebeLogo_label.setObjectName(_fromUtf8("phoebeLogo_label"))
        self.gridLayout_5.addWidget(self.phoebeLogo_label, 0, 0, 1, 1)
        self.phoebe_label = QtGui.QLabel(popAbout_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phoebe_label.sizePolicy().hasHeightForWidth())
        self.phoebe_label.setSizePolicy(sizePolicy)
        self.phoebe_label.setTextFormat(QtCore.Qt.AutoText)
        self.phoebe_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.phoebe_label.setOpenExternalLinks(True)
        self.phoebe_label.setObjectName(_fromUtf8("phoebe_label"))
        self.gridLayout_5.addWidget(self.phoebe_label, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.TabWidget = QtGui.QTabWidget(popAbout_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TabWidget.sizePolicy().hasHeightForWidth())
        self.TabWidget.setSizePolicy(sizePolicy)
        self.TabWidget.setObjectName(_fromUtf8("TabWidget"))
        self.info_tabWidget = QtGui.QWidget()
        self.info_tabWidget.setObjectName(_fromUtf8("info_tabWidget"))
        self.gridLayout_3 = QtGui.QGridLayout(self.info_tabWidget)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.info_label = QtGui.QLabel(self.info_tabWidget)
        self.info_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setObjectName(_fromUtf8("info_label"))
        self.gridLayout_3.addWidget(self.info_label, 0, 0, 1, 1)
        self.TabWidget.addTab(self.info_tabWidget, _fromUtf8(""))
        self.changeLog_tabWidget = QtGui.QWidget()
        self.changeLog_tabWidget.setObjectName(_fromUtf8("changeLog_tabWidget"))
        self.gridLayout_7 = QtGui.QGridLayout(self.changeLog_tabWidget)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.changeLog_TextEdit = QtGui.QTextEdit(self.changeLog_tabWidget)
        self.changeLog_TextEdit.setObjectName(_fromUtf8("changeLog_TextEdit"))
        self.gridLayout_7.addWidget(self.changeLog_TextEdit, 0, 0, 1, 1)
        self.TabWidget.addTab(self.changeLog_tabWidget, _fromUtf8(""))
        self.credits_tabWidget = QtGui.QWidget()
        self.credits_tabWidget.setObjectName(_fromUtf8("credits_tabWidget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.credits_tabWidget)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.credits_TextEdit = QtGui.QTextEdit(self.credits_tabWidget)
        self.credits_TextEdit.setObjectName(_fromUtf8("credits_TextEdit"))
        self.gridLayout_4.addWidget(self.credits_TextEdit, 0, 0, 1, 1)
        self.TabWidget.addTab(self.credits_tabWidget, _fromUtf8(""))
        self.license_tabWidget = QtGui.QWidget()
        self.license_tabWidget.setObjectName(_fromUtf8("license_tabWidget"))
        self.gridLayout_6 = QtGui.QGridLayout(self.license_tabWidget)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.license_TextEdit = QtGui.QTextEdit(self.license_tabWidget)
        self.license_TextEdit.setObjectName(_fromUtf8("license_TextEdit"))
        self.gridLayout_6.addWidget(self.license_TextEdit, 0, 0, 1, 1)
        self.TabWidget.addTab(self.license_tabWidget, _fromUtf8(""))
        self.gridLayout.addWidget(self.TabWidget, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(popAbout_Dialog)
        self.TabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(popAbout_Dialog)

    def retranslateUi(self, popAbout_Dialog):
        popAbout_Dialog.setWindowTitle(QtGui.QApplication.translate("popAbout_Dialog", "PHOEBE - About", None, QtGui.QApplication.UnicodeUTF8))
        self.phoebeLogo_label.setText(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/images/phoebe-gui.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.phoebe_label.setText(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600;\">PHOEBE</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.info_label.setText(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">PHOEBE - PHysics Of Eclipsing BinariEs</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">v 2.0</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">version/download information</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a href=\"http://phoebe.villanova.edu\"><span style=\" text-decoration: underline; color:#0000ff;\">PHOEBE Website</span></a></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.info_tabWidget), QtGui.QApplication.translate("popAbout_Dialog", "Info", None, QtGui.QApplication.UnicodeUTF8))
        self.changeLog_TextEdit.setHtml(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">do we want to include a changelog here?</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.changeLog_tabWidget), QtGui.QApplication.translate("popAbout_Dialog", "ChangeLog", None, QtGui.QApplication.UnicodeUTF8))
        self.credits_TextEdit.setHtml(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">developers and contact information</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.credits_tabWidget), QtGui.QApplication.translate("popAbout_Dialog", "Credits", None, QtGui.QApplication.UnicodeUTF8))
        self.license_TextEdit.setHtml(QtGui.QApplication.translate("popAbout_Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">PHOEBE is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">PHOEBE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">You should have received a copy of the GNU General Public License along with PHOEBE; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.license_tabWidget), QtGui.QApplication.translate("popAbout_Dialog", "License", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popFileEntryColWidget(object):
    def setupUi(self, popFileEntryColWidget):
        popFileEntryColWidget.setObjectName(_fromUtf8("popFileEntryColWidget"))
        popFileEntryColWidget.resize(400, 300)
        self.verticalLayout_2 = QtGui.QVBoxLayout(popFileEntryColWidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.type_comboBox = QtGui.QComboBox(popFileEntryColWidget)
        self.type_comboBox.setEnabled(True)
        self.type_comboBox.setObjectName(_fromUtf8("type_comboBox"))
        self.verticalLayout.addWidget(self.type_comboBox)
        self.units_comboBox = QtGui.QComboBox(popFileEntryColWidget)
        self.units_comboBox.setEnabled(False)
        self.units_comboBox.setObjectName(_fromUtf8("units_comboBox"))
        self.units_comboBox.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.units_comboBox)
        self.comp_comboBox = QtGui.QComboBox(popFileEntryColWidget)
        self.comp_comboBox.setEnabled(False)
        self.comp_comboBox.setObjectName(_fromUtf8("comp_comboBox"))
        self.comp_comboBox.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.comp_comboBox)
        self.col_comboBox = QtGui.QComboBox(popFileEntryColWidget)
        self.col_comboBox.setEnabled(True)
        self.col_comboBox.setObjectName(_fromUtf8("col_comboBox"))
        self.col_comboBox.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.col_comboBox)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(popFileEntryColWidget)
        QtCore.QMetaObject.connectSlotsByName(popFileEntryColWidget)

    def retranslateUi(self, popFileEntryColWidget):
        popFileEntryColWidget.setWindowTitle(QtGui.QApplication.translate("popFileEntryColWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.units_comboBox.setItemText(0, QtGui.QApplication.translate("popFileEntryColWidget", "--Units--", None, QtGui.QApplication.UnicodeUTF8))
        self.comp_comboBox.setItemText(0, QtGui.QApplication.translate("popFileEntryColWidget", "--Component--", None, QtGui.QApplication.UnicodeUTF8))
        self.col_comboBox.setItemText(0, QtGui.QApplication.translate("popFileEntryColWidget", "--Column--", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popFileEntry_Dialog(object):
    def setupUi(self, popFileEntry_Dialog):
        popFileEntry_Dialog.setObjectName(_fromUtf8("popFileEntry_Dialog"))
        popFileEntry_Dialog.setWindowModality(QtCore.Qt.NonModal)
        popFileEntry_Dialog.resize(779, 640)
        self.gridLayout = QtGui.QGridLayout(popFileEntry_Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.buttonBox = QtGui.QDialogButtonBox(popFileEntry_Dialog)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.plot_Widget = QtGui.QWidget(popFileEntry_Dialog)
        self.plot_Widget.setObjectName(_fromUtf8("plot_Widget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.plot_Widget)
        self.gridLayout_4.setMargin(0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.dataTextEdit = QtGui.QPlainTextEdit(self.plot_Widget)
        self.dataTextEdit.setUndoRedoEnabled(False)
        self.dataTextEdit.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.dataTextEdit.setObjectName(_fromUtf8("dataTextEdit"))
        self.gridLayout_4.addWidget(self.dataTextEdit, 3, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 5, 0, 1, 1)
        self.widget_ColWidgets = QtGui.QWidget(self.plot_Widget)
        self.widget_ColWidgets.setObjectName(_fromUtf8("widget_ColWidgets"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.widget_ColWidgets)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.horizontalLayout_ColWidgets = QtGui.QHBoxLayout()
        self.horizontalLayout_ColWidgets.setObjectName(_fromUtf8("horizontalLayout_ColWidgets"))
        self.horizontalLayout_2.addLayout(self.horizontalLayout_ColWidgets)
        self.gridLayout_4.addWidget(self.widget_ColWidgets, 2, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.plot_gridLayout = QtGui.QGridLayout()
        self.plot_gridLayout.setObjectName(_fromUtf8("plot_gridLayout"))
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label = QtGui.QLabel(self.plot_Widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.plot_Widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 3, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.pfe_fileChooserButton = QtGui.QPushButton(self.plot_Widget)
        self.pfe_fileChooserButton.setObjectName(_fromUtf8("pfe_fileChooserButton"))
        self.horizontalLayout_3.addWidget(self.pfe_fileChooserButton)
        self.pfe_fileReloadButton = QtGui.QCommandLinkButton(self.plot_Widget)
        self.pfe_fileReloadButton.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pfe_fileReloadButton.sizePolicy().hasHeightForWidth())
        self.pfe_fileReloadButton.setSizePolicy(sizePolicy)
        self.pfe_fileReloadButton.setMaximumSize(QtCore.QSize(90, 30))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/refresh.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pfe_fileReloadButton.setIcon(icon)
        self.pfe_fileReloadButton.setIconSize(QtCore.QSize(16, 16))
        self.pfe_fileReloadButton.setObjectName(_fromUtf8("pfe_fileReloadButton"))
        self.horizontalLayout_3.addWidget(self.pfe_fileReloadButton)
        self.pfe_synChooserButton = QtGui.QPushButton(self.plot_Widget)
        self.pfe_synChooserButton.setEnabled(True)
        self.pfe_synChooserButton.setObjectName(_fromUtf8("pfe_synChooserButton"))
        self.horizontalLayout_3.addWidget(self.pfe_synChooserButton)
        self.gridLayout_3.addLayout(self.horizontalLayout_3, 1, 2, 1, 1)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.name = QtGui.QLineEdit(self.plot_Widget)
        self.name.setObjectName(_fromUtf8("name"))
        self.horizontalLayout_4.addWidget(self.name)
        self.label_3 = QtGui.QLabel(self.plot_Widget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        self.doubleSpinBox = QtGui.QDoubleSpinBox(self.plot_Widget)
        self.doubleSpinBox.setEnabled(False)
        self.doubleSpinBox.setObjectName(_fromUtf8("doubleSpinBox"))
        self.horizontalLayout_4.addWidget(self.doubleSpinBox)
        self.gridLayout_3.addLayout(self.horizontalLayout_4, 3, 2, 1, 1)
        self.label_4 = QtGui.QLabel(self.plot_Widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 4, 0, 1, 1)
        self.pfe_filterComboBox = QtGui.QComboBox(self.plot_Widget)
        self.pfe_filterComboBox.setObjectName(_fromUtf8("pfe_filterComboBox"))
        self.pfe_filterComboBox.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.pfe_filterComboBox, 4, 2, 1, 1)
        self.label_5 = QtGui.QLabel(self.plot_Widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.pfe_categoryComboBox = QtGui.QComboBox(self.plot_Widget)
        self.pfe_categoryComboBox.setObjectName(_fromUtf8("pfe_categoryComboBox"))
        self.pfe_categoryComboBox.addItem(_fromUtf8(""))
        self.pfe_categoryComboBox.addItem(_fromUtf8(""))
        self.pfe_categoryComboBox.addItem(_fromUtf8(""))
        self.pfe_categoryComboBox.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.pfe_categoryComboBox, 0, 2, 1, 1)
        self.plot_gridLayout.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.plot_gridLayout)
        self.gridLayout_4.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.syn_timeWidget = QtGui.QWidget(self.plot_Widget)
        self.syn_timeWidget.setObjectName(_fromUtf8("syn_timeWidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.syn_timeWidget)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_10 = QtGui.QLabel(self.syn_timeWidget)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.gridLayout_2.addWidget(self.label_10, 3, 4, 1, 1)
        self.label_9 = QtGui.QLabel(self.syn_timeWidget)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.gridLayout_2.addWidget(self.label_9, 3, 1, 1, 1)
        self.linspace_num = QtGui.QSpinBox(self.syn_timeWidget)
        self.linspace_num.setMinimum(1)
        self.linspace_num.setMaximum(100000)
        self.linspace_num.setSingleStep(10)
        self.linspace_num.setProperty("value", 100)
        self.linspace_num.setObjectName(_fromUtf8("linspace_num"))
        self.gridLayout_2.addWidget(self.linspace_num, 3, 8, 1, 1)
        self.times_match = QtGui.QRadioButton(self.syn_timeWidget)
        self.times_match.setEnabled(True)
        self.times_match.setChecked(False)
        self.times_match.setObjectName(_fromUtf8("times_match"))
        self.gridLayout_2.addWidget(self.times_match, 1, 0, 1, 1)
        self.times_list = QtGui.QRadioButton(self.syn_timeWidget)
        self.times_list.setEnabled(True)
        self.times_list.setChecked(True)
        self.times_list.setObjectName(_fromUtf8("times_list"))
        self.gridLayout_2.addWidget(self.times_list, 4, 0, 1, 1)
        self.times_linspace = QtGui.QRadioButton(self.syn_timeWidget)
        self.times_linspace.setEnabled(True)
        self.times_linspace.setObjectName(_fromUtf8("times_linspace"))
        self.gridLayout_2.addWidget(self.times_linspace, 3, 0, 1, 1)
        self.times_arange = QtGui.QRadioButton(self.syn_timeWidget)
        self.times_arange.setEnabled(True)
        self.times_arange.setObjectName(_fromUtf8("times_arange"))
        self.gridLayout_2.addWidget(self.times_arange, 2, 0, 1, 1)
        self.linspace_max = QtGui.QDoubleSpinBox(self.syn_timeWidget)
        self.linspace_max.setDecimals(6)
        self.linspace_max.setMaximum(9999999.0)
        self.linspace_max.setObjectName(_fromUtf8("linspace_max"))
        self.gridLayout_2.addWidget(self.linspace_max, 3, 5, 1, 1)
        self.linspace_min = QtGui.QDoubleSpinBox(self.syn_timeWidget)
        self.linspace_min.setDecimals(6)
        self.linspace_min.setMaximum(9999999.0)
        self.linspace_min.setObjectName(_fromUtf8("linspace_min"))
        self.gridLayout_2.addWidget(self.linspace_min, 3, 2, 1, 1)
        self.label_13 = QtGui.QLabel(self.syn_timeWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.gridLayout_2.addWidget(self.label_13, 5, 0, 1, 1)
        self.label_11 = QtGui.QLabel(self.syn_timeWidget)
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.gridLayout_2.addWidget(self.label_11, 3, 7, 1, 1)
        self.label_6 = QtGui.QLabel(self.syn_timeWidget)
        self.label_6.setMaximumSize(QtCore.QSize(30, 16777215))
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_2.addWidget(self.label_6, 2, 1, 1, 1)
        self.timeselect_linspace_min = QtGui.QPushButton(self.syn_timeWidget)
        self.timeselect_linspace_min.setMaximumSize(QtCore.QSize(24, 16777215))
        self.timeselect_linspace_min.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/ellipsis.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.timeselect_linspace_min.setIcon(icon1)
        self.timeselect_linspace_min.setObjectName(_fromUtf8("timeselect_linspace_min"))
        self.gridLayout_2.addWidget(self.timeselect_linspace_min, 3, 3, 1, 1)
        self.arange_max = QtGui.QDoubleSpinBox(self.syn_timeWidget)
        self.arange_max.setDecimals(6)
        self.arange_max.setMaximum(9999999.0)
        self.arange_max.setObjectName(_fromUtf8("arange_max"))
        self.gridLayout_2.addWidget(self.arange_max, 2, 5, 1, 1)
        self.timeselect_arange_min = QtGui.QPushButton(self.syn_timeWidget)
        self.timeselect_arange_min.setMaximumSize(QtCore.QSize(24, 16777215))
        self.timeselect_arange_min.setText(_fromUtf8(""))
        self.timeselect_arange_min.setIcon(icon1)
        self.timeselect_arange_min.setObjectName(_fromUtf8("timeselect_arange_min"))
        self.gridLayout_2.addWidget(self.timeselect_arange_min, 2, 3, 1, 1)
        self.datasetComboBox = QtGui.QComboBox(self.syn_timeWidget)
        self.datasetComboBox.setObjectName(_fromUtf8("datasetComboBox"))
        self.gridLayout_2.addWidget(self.datasetComboBox, 1, 1, 1, 8)
        self.label_7 = QtGui.QLabel(self.syn_timeWidget)
        self.label_7.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout_2.addWidget(self.label_7, 2, 4, 1, 1)
        self.time_custom = QtGui.QLineEdit(self.syn_timeWidget)
        self.time_custom.setObjectName(_fromUtf8("time_custom"))
        self.gridLayout_2.addWidget(self.time_custom, 4, 1, 1, 8)
        self.label_8 = QtGui.QLabel(self.syn_timeWidget)
        self.label_8.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_2.addWidget(self.label_8, 2, 7, 1, 1)
        self.timeselect_linspace_max = QtGui.QPushButton(self.syn_timeWidget)
        self.timeselect_linspace_max.setMaximumSize(QtCore.QSize(24, 16777215))
        self.timeselect_linspace_max.setText(_fromUtf8(""))
        self.timeselect_linspace_max.setIcon(icon1)
        self.timeselect_linspace_max.setObjectName(_fromUtf8("timeselect_linspace_max"))
        self.gridLayout_2.addWidget(self.timeselect_linspace_max, 3, 6, 1, 1)
        self.arange_min = QtGui.QDoubleSpinBox(self.syn_timeWidget)
        self.arange_min.setDecimals(6)
        self.arange_min.setMaximum(9999999.0)
        self.arange_min.setObjectName(_fromUtf8("arange_min"))
        self.gridLayout_2.addWidget(self.arange_min, 2, 2, 1, 1)
        self.timeselect_arange_max = QtGui.QPushButton(self.syn_timeWidget)
        self.timeselect_arange_max.setMaximumSize(QtCore.QSize(24, 16777215))
        self.timeselect_arange_max.setText(_fromUtf8(""))
        self.timeselect_arange_max.setIcon(icon1)
        self.timeselect_arange_max.setObjectName(_fromUtf8("timeselect_arange_max"))
        self.gridLayout_2.addWidget(self.timeselect_arange_max, 2, 6, 1, 1)
        self.label_12 = QtGui.QLabel(self.syn_timeWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setTextFormat(QtCore.Qt.PlainText)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.gridLayout_2.addWidget(self.label_12, 0, 0, 1, 1)
        self.arange_step = QtGui.QDoubleSpinBox(self.syn_timeWidget)
        self.arange_step.setDecimals(6)
        self.arange_step.setMinimum(1e-06)
        self.arange_step.setMaximum(10000.0)
        self.arange_step.setObjectName(_fromUtf8("arange_step"))
        self.gridLayout_2.addWidget(self.arange_step, 2, 8, 1, 1)
        self.syn_componentsWidget = QtGui.QWidget(self.syn_timeWidget)
        self.syn_componentsWidget.setObjectName(_fromUtf8("syn_componentsWidget"))
        self.gridLayout_2.addWidget(self.syn_componentsWidget, 6, 0, 1, 2)
        self.gridLayout_4.addWidget(self.syn_timeWidget, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.plot_Widget, 0, 0, 1, 1)

        self.retranslateUi(popFileEntry_Dialog)
        QtCore.QMetaObject.connectSlotsByName(popFileEntry_Dialog)

    def retranslateUi(self, popFileEntry_Dialog):
        popFileEntry_Dialog.setWindowTitle(QtGui.QApplication.translate("popFileEntry_Dialog", "PHOEBE - Import Data", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Data:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Name:", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_fileChooserButton.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Choose File", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_fileReloadButton.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Reload", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_synChooserButton.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Synthetic Only", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Sigma:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Passband:", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_filterComboBox.setItemText(0, QtGui.QApplication.translate("popFileEntry_Dialog", "--Passband--", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Category:", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_categoryComboBox.setItemText(0, QtGui.QApplication.translate("popFileEntry_Dialog", "lc", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_categoryComboBox.setItemText(1, QtGui.QApplication.translate("popFileEntry_Dialog", "rv", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_categoryComboBox.setItemText(2, QtGui.QApplication.translate("popFileEntry_Dialog", "sp", None, QtGui.QApplication.UnicodeUTF8))
        self.pfe_categoryComboBox.setItemText(3, QtGui.QApplication.translate("popFileEntry_Dialog", "etv", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "max:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "min:", None, QtGui.QApplication.UnicodeUTF8))
        self.times_match.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "match dataset", None, QtGui.QApplication.UnicodeUTF8))
        self.times_list.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "custom list", None, QtGui.QApplication.UnicodeUTF8))
        self.times_linspace.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "np.linspace", None, QtGui.QApplication.UnicodeUTF8))
        self.times_arange.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "np.arange", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Components:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "num:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "min:", None, QtGui.QApplication.UnicodeUTF8))
        self.timeselect_linspace_min.setToolTip(QtGui.QApplication.translate("popFileEntry_Dialog", "set from existing datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.timeselect_arange_min.setToolTip(QtGui.QApplication.translate("popFileEntry_Dialog", "set from existing datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "max:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "step:", None, QtGui.QApplication.UnicodeUTF8))
        self.timeselect_linspace_max.setToolTip(QtGui.QApplication.translate("popFileEntry_Dialog", "set from existing datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.timeselect_arange_max.setToolTip(QtGui.QApplication.translate("popFileEntry_Dialog", "set from existing datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("popFileEntry_Dialog", "Times:", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popHelp_Dialog(object):
    def setupUi(self, popHelp_Dialog):
        popHelp_Dialog.setObjectName(_fromUtf8("popHelp_Dialog"))
        popHelp_Dialog.resize(400, 300)
        self.buttonBox = QtGui.QDialogButtonBox(popHelp_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 260, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Close)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))

        self.retranslateUi(popHelp_Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popHelp_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popHelp_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(popHelp_Dialog)

    def retranslateUi(self, popHelp_Dialog):
        popHelp_Dialog.setWindowTitle(QtGui.QApplication.translate("popHelp_Dialog", "PHOEBE - Help", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popLock_Dialog(object):
    def setupUi(self, popLock_Dialog):
        popLock_Dialog.setObjectName(_fromUtf8("popLock_Dialog"))
        popLock_Dialog.resize(536, 300)
        font = QtGui.QFont()
        font.setStrikeOut(False)
        popLock_Dialog.setFont(font)
        self.verticalLayout = QtGui.QVBoxLayout(popLock_Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.save_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy)
        self.save_button.setMinimumSize(QtCore.QSize(0, 40))
        self.save_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon = QtGui.QIcon.fromTheme(_fromUtf8("gtk-save"))
        self.save_button.setIcon(icon)
        self.save_button.setObjectName(_fromUtf8("save_button"))
        self.horizontalLayout_2.addWidget(self.save_button)
        self.saveas_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveas_button.sizePolicy().hasHeightForWidth())
        self.saveas_button.setSizePolicy(sizePolicy)
        self.saveas_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon = QtGui.QIcon.fromTheme(_fromUtf8("gtk-save-as"))
        self.saveas_button.setIcon(icon)
        self.saveas_button.setObjectName(_fromUtf8("saveas_button"))
        self.horizontalLayout_2.addWidget(self.saveas_button)
        self.new_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.new_button.sizePolicy().hasHeightForWidth())
        self.new_button.setSizePolicy(sizePolicy)
        self.new_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon = QtGui.QIcon.fromTheme(_fromUtf8("gtk-home"))
        self.new_button.setIcon(icon)
        self.new_button.setObjectName(_fromUtf8("new_button"))
        self.horizontalLayout_2.addWidget(self.new_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.server_label = QtGui.QLabel(popLock_Dialog)
        self.server_label.setMinimumSize(QtCore.QSize(28, 0))
        self.server_label.setMaximumSize(QtCore.QSize(16777215, 28))
        self.server_label.setObjectName(_fromUtf8("server_label"))
        self.gridLayout_2.addWidget(self.server_label, 0, 1, 1, 1)
        self.job_refresh = StatusPushButton(popLock_Dialog)
        self.job_refresh.setMinimumSize(QtCore.QSize(28, 28))
        self.job_refresh.setMaximumSize(QtCore.QSize(28, 28))
        self.job_refresh.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/bullet.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.job_refresh.setIcon(icon)
        self.job_refresh.setIconSize(QtCore.QSize(32, 32))
        self.job_refresh.setFlat(True)
        self.job_refresh.setObjectName(_fromUtf8("job_refresh"))
        self.gridLayout_2.addWidget(self.job_refresh, 1, 0, 1, 1)
        self.job_label = QtGui.QLabel(popLock_Dialog)
        self.job_label.setMinimumSize(QtCore.QSize(0, 28))
        self.job_label.setMaximumSize(QtCore.QSize(16777215, 28))
        self.job_label.setObjectName(_fromUtf8("job_label"))
        self.gridLayout_2.addWidget(self.job_label, 1, 1, 1, 1)
        self.server_refresh = StatusPushButton(popLock_Dialog)
        self.server_refresh.setMinimumSize(QtCore.QSize(28, 28))
        self.server_refresh.setMaximumSize(QtCore.QSize(28, 28))
        self.server_refresh.setText(_fromUtf8(""))
        self.server_refresh.setIcon(icon)
        self.server_refresh.setIconSize(QtCore.QSize(32, 32))
        self.server_refresh.setFlat(True)
        self.server_refresh.setObjectName(_fromUtf8("server_refresh"))
        self.gridLayout_2.addWidget(self.server_refresh, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.loop_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loop_button.sizePolicy().hasHeightForWidth())
        self.loop_button.setSizePolicy(sizePolicy)
        self.loop_button.setMinimumSize(QtCore.QSize(0, 40))
        self.loop_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/repeat-2.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.loop_button.setIcon(icon1)
        self.loop_button.setIconSize(QtCore.QSize(30, 30))
        self.loop_button.setObjectName(_fromUtf8("loop_button"))
        self.horizontalLayout.addWidget(self.loop_button)
        self.getresults_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.getresults_button.sizePolicy().hasHeightForWidth())
        self.getresults_button.setSizePolicy(sizePolicy)
        self.getresults_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/pull.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.getresults_button.setIcon(icon2)
        self.getresults_button.setIconSize(QtCore.QSize(30, 30))
        self.getresults_button.setObjectName(_fromUtf8("getresults_button"))
        self.horizontalLayout.addWidget(self.getresults_button)
        self.unlock_button = QtGui.QPushButton(popLock_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.unlock_button.sizePolicy().hasHeightForWidth())
        self.unlock_button.setSizePolicy(sizePolicy)
        self.unlock_button.setMaximumSize(QtCore.QSize(16777215, 40))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/delete.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.unlock_button.setIcon(icon3)
        self.unlock_button.setIconSize(QtCore.QSize(30, 30))
        self.unlock_button.setObjectName(_fromUtf8("unlock_button"))
        self.horizontalLayout.addWidget(self.unlock_button)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(popLock_Dialog)
        QtCore.QMetaObject.connectSlotsByName(popLock_Dialog)

    def retranslateUi(self, popLock_Dialog):
        popLock_Dialog.setWindowTitle(QtGui.QApplication.translate("popLock_Dialog", "PHOEBE - waiting for results from server", None, QtGui.QApplication.UnicodeUTF8))
        self.save_button.setText(QtGui.QApplication.translate("popLock_Dialog", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.saveas_button.setText(QtGui.QApplication.translate("popLock_Dialog", "Save As...", None, QtGui.QApplication.UnicodeUTF8))
        self.new_button.setText(QtGui.QApplication.translate("popLock_Dialog", "New", None, QtGui.QApplication.UnicodeUTF8))
        self.server_label.setText(QtGui.QApplication.translate("popLock_Dialog", "Server: servername", None, QtGui.QApplication.UnicodeUTF8))
        self.job_label.setText(QtGui.QApplication.translate("popLock_Dialog", "Job: job info", None, QtGui.QApplication.UnicodeUTF8))
        self.loop_button.setText(QtGui.QApplication.translate("popLock_Dialog", "Wait for Results", None, QtGui.QApplication.UnicodeUTF8))
        self.getresults_button.setText(QtGui.QApplication.translate("popLock_Dialog", "Get Results", None, QtGui.QApplication.UnicodeUTF8))
        self.unlock_button.setText(QtGui.QApplication.translate("popLock_Dialog", "Forget Job", None, QtGui.QApplication.UnicodeUTF8))

from phoebe_widgets import StatusPushButton

class Ui_popObsOptions_Dialog(object):
    def setupUi(self, popObsOptions_Dialog):
        popObsOptions_Dialog.setObjectName(_fromUtf8("popObsOptions_Dialog"))
        popObsOptions_Dialog.resize(400, 300)
        self.verticalLayout = QtGui.QVBoxLayout(popObsOptions_Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(popObsOptions_Dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtGui.QLabel(popObsOptions_Dialog)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtGui.QLabel(popObsOptions_Dialog)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 5, 0, 1, 1)
        self.label_4 = QtGui.QLabel(popObsOptions_Dialog)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_2 = QtGui.QLabel(popObsOptions_Dialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.p_heating = QtGui.QCheckBox(popObsOptions_Dialog)
        self.p_heating.setEnabled(True)
        self.p_heating.setObjectName(_fromUtf8("p_heating"))
        self.gridLayout.addWidget(self.p_heating, 0, 1, 1, 1)
        self.p_eclipse_alg = QtGui.QComboBox(popObsOptions_Dialog)
        self.p_eclipse_alg.setEnabled(True)
        self.p_eclipse_alg.setObjectName(_fromUtf8("p_eclipse_alg"))
        self.p_eclipse_alg.addItem(_fromUtf8(""))
        self.gridLayout.addWidget(self.p_eclipse_alg, 3, 1, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.p_refl = QtGui.QCheckBox(popObsOptions_Dialog)
        self.p_refl.setEnabled(True)
        self.p_refl.setObjectName(_fromUtf8("p_refl"))
        self.horizontalLayout_2.addWidget(self.p_refl)
        self.p_refl_num = QtGui.QSpinBox(popObsOptions_Dialog)
        self.p_refl_num.setEnabled(False)
        self.p_refl_num.setObjectName(_fromUtf8("p_refl_num"))
        self.horizontalLayout_2.addWidget(self.p_refl_num)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.p_subdiv_alg = QtGui.QComboBox(popObsOptions_Dialog)
        self.p_subdiv_alg.setEnabled(True)
        self.p_subdiv_alg.setObjectName(_fromUtf8("p_subdiv_alg"))
        self.p_subdiv_alg.addItem(_fromUtf8(""))
        self.horizontalLayout_3.addWidget(self.p_subdiv_alg)
        self.p_subdiv_num = QtGui.QSpinBox(popObsOptions_Dialog)
        self.p_subdiv_num.setEnabled(True)
        self.p_subdiv_num.setObjectName(_fromUtf8("p_subdiv_num"))
        self.horizontalLayout_3.addWidget(self.p_subdiv_num)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.RENAMEp_mpi = QtGui.QCheckBox(popObsOptions_Dialog)
        self.RENAMEp_mpi.setEnabled(False)
        self.RENAMEp_mpi.setObjectName(_fromUtf8("RENAMEp_mpi"))
        self.horizontalLayout_4.addWidget(self.RENAMEp_mpi)
        self.pushButton_3 = QtGui.QPushButton(popObsOptions_Dialog)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.horizontalLayout_4.addWidget(self.pushButton_3)
        self.gridLayout.addLayout(self.horizontalLayout_4, 5, 1, 1, 1)
        self.label_6 = QtGui.QLabel(popObsOptions_Dialog)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.p_ltt = QtGui.QCheckBox(popObsOptions_Dialog)
        self.p_ltt.setObjectName(_fromUtf8("p_ltt"))
        self.gridLayout.addWidget(self.p_ltt, 4, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.resetPushButton = QtGui.QPushButton(popObsOptions_Dialog)
        self.resetPushButton.setEnabled(True)
        self.resetPushButton.setObjectName(_fromUtf8("resetPushButton"))
        self.horizontalLayout.addWidget(self.resetPushButton)
        self.setPushButton = QtGui.QPushButton(popObsOptions_Dialog)
        self.setPushButton.setEnabled(True)
        self.setPushButton.setObjectName(_fromUtf8("setPushButton"))
        self.horizontalLayout.addWidget(self.setPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtGui.QDialogButtonBox(popObsOptions_Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(popObsOptions_Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popObsOptions_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popObsOptions_Dialog.reject)
        QtCore.QObject.connect(self.p_refl, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.p_refl_num.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(popObsOptions_Dialog)

    def retranslateUi(self, popObsOptions_Dialog):
        popObsOptions_Dialog.setWindowTitle(QtGui.QApplication.translate("popObsOptions_Dialog", "PHOEBE - Observe Options", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Heating:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Subdivision:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "MPI:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Eclipse Detection:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Reflection:", None, QtGui.QApplication.UnicodeUTF8))
        self.p_heating.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.p_eclipse_alg.setItemText(0, QtGui.QApplication.translate("popObsOptions_Dialog", "auto", None, QtGui.QApplication.UnicodeUTF8))
        self.p_refl.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.p_subdiv_alg.setItemText(0, QtGui.QApplication.translate("popObsOptions_Dialog", "edge", None, QtGui.QApplication.UnicodeUTF8))
        self.RENAMEp_mpi.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_3.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Light Time Effect:", None, QtGui.QApplication.UnicodeUTF8))
        self.p_ltt.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.resetPushButton.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Reset to Defaults", None, QtGui.QApplication.UnicodeUTF8))
        self.setPushButton.setText(QtGui.QApplication.translate("popObsOptions_Dialog", "Set as Defaults", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popPlot_Dialog(object):
    def setupUi(self, popPlot_Dialog):
        popPlot_Dialog.setObjectName(_fromUtf8("popPlot_Dialog"))
        popPlot_Dialog.setWindowModality(QtCore.Qt.NonModal)
        popPlot_Dialog.resize(765, 466)
        self.gridLayout = QtGui.QGridLayout(popPlot_Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.plot_Widget = QtGui.QWidget(popPlot_Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_Widget.sizePolicy().hasHeightForWidth())
        self.plot_Widget.setSizePolicy(sizePolicy)
        self.plot_Widget.setObjectName(_fromUtf8("plot_Widget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.plot_Widget)
        self.gridLayout_4.setMargin(0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.plot_gridLayout = QtGui.QGridLayout()
        self.plot_gridLayout.setObjectName(_fromUtf8("plot_gridLayout"))
        self.plot_holder_widget = QtGui.QWidget(self.plot_Widget)
        self.plot_holder_widget.setObjectName(_fromUtf8("plot_holder_widget"))
        self.plot_gridLayout.addWidget(self.plot_holder_widget, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.zoom_in = QtGui.QCommandLinkButton(self.plot_Widget)
        self.zoom_in.setMaximumSize(QtCore.QSize(100, 32))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.zoom_in.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/zoom-in.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoom_in.setIcon(icon)
        self.zoom_in.setIconSize(QtCore.QSize(16, 16))
        self.zoom_in.setCheckable(True)
        self.zoom_in.setAutoDefault(True)
        self.zoom_in.setObjectName(_fromUtf8("zoom_in"))
        self.horizontalLayout_5.addWidget(self.zoom_in)
        self.zoom_out = QtGui.QCommandLinkButton(self.plot_Widget)
        self.zoom_out.setMaximumSize(QtCore.QSize(100, 32))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.zoom_out.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/zoom-out.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoom_out.setIcon(icon1)
        self.zoom_out.setIconSize(QtCore.QSize(16, 16))
        self.zoom_out.setObjectName(_fromUtf8("zoom_out"))
        self.horizontalLayout_5.addWidget(self.zoom_out)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.pop_mpl = QtGui.QCommandLinkButton(self.plot_Widget)
        self.pop_mpl.setMaximumSize(QtCore.QSize(100, 32))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.pop_mpl.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/pop.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pop_mpl.setIcon(icon2)
        self.pop_mpl.setIconSize(QtCore.QSize(16, 16))
        self.pop_mpl.setObjectName(_fromUtf8("pop_mpl"))
        self.horizontalLayout_5.addWidget(self.pop_mpl)
        self.save = QtGui.QCommandLinkButton(self.plot_Widget)
        self.save.setMaximumSize(QtCore.QSize(100, 32))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.save.setFont(font)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/list.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save.setIcon(icon3)
        self.save.setIconSize(QtCore.QSize(16, 16))
        self.save.setObjectName(_fromUtf8("save"))
        self.horizontalLayout_5.addWidget(self.save)
        self.plot_gridLayout.addLayout(self.horizontalLayout_5, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.plot_gridLayout, 3, 0, 1, 1)
        self.titleStackedWidget = QtGui.QStackedWidget(self.plot_Widget)
        self.titleStackedWidget.setMaximumSize(QtCore.QSize(16777215, 32))
        self.titleStackedWidget.setObjectName(_fromUtf8("titleStackedWidget"))
        self.page_3 = QtGui.QWidget()
        self.page_3.setObjectName(_fromUtf8("page_3"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.page_3)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.titleLinkButton = QtGui.QCommandLinkButton(self.page_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.titleLinkButton.sizePolicy().hasHeightForWidth())
        self.titleLinkButton.setSizePolicy(sizePolicy)
        self.titleLinkButton.setMaximumSize(QtCore.QSize(16777215, 32))
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.titleLinkButton.setFont(font)
        self.titleLinkButton.setIconSize(QtCore.QSize(0, 0))
        self.titleLinkButton.setObjectName(_fromUtf8("titleLinkButton"))
        self.horizontalLayout.addWidget(self.titleLinkButton)
        self.titleStackedWidget.addWidget(self.page_3)
        self.page_4 = QtGui.QWidget()
        self.page_4.setObjectName(_fromUtf8("page_4"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.page_4)
        self.horizontalLayout_3.setMargin(0)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.titleLineEdit = QtGui.QLineEdit(self.page_4)
        self.titleLineEdit.setObjectName(_fromUtf8("titleLineEdit"))
        self.horizontalLayout_3.addWidget(self.titleLineEdit)
        self.title_cancelButton = QtGui.QPushButton(self.page_4)
        self.title_cancelButton.setObjectName(_fromUtf8("title_cancelButton"))
        self.horizontalLayout_3.addWidget(self.title_cancelButton)
        self.title_saveButton = QtGui.QPushButton(self.page_4)
        self.title_saveButton.setObjectName(_fromUtf8("title_saveButton"))
        self.horizontalLayout_3.addWidget(self.title_saveButton)
        self.titleStackedWidget.addWidget(self.page_4)
        self.gridLayout_4.addWidget(self.titleStackedWidget, 1, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.yaxisCombo = QtGui.QComboBox(self.plot_Widget)
        self.yaxisCombo.setEnabled(False)
        self.yaxisCombo.setObjectName(_fromUtf8("yaxisCombo"))
        self.gridLayout_2.addWidget(self.yaxisCombo, 0, 5, 1, 1)
        self.label_2 = QtGui.QLabel(self.plot_Widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 4, 1, 1)
        self.label = QtGui.QLabel(self.plot_Widget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_2.addWidget(self.label, 0, 1, 1, 1)
        self.xaxisCombo = QtGui.QComboBox(self.plot_Widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xaxisCombo.sizePolicy().hasHeightForWidth())
        self.xaxisCombo.setSizePolicy(sizePolicy)
        self.xaxisCombo.setMinimumSize(QtCore.QSize(150, 0))
        self.xaxisCombo.setMaximumSize(QtCore.QSize(150, 16777215))
        self.xaxisCombo.setObjectName(_fromUtf8("xaxisCombo"))
        self.xaxisCombo.addItem(_fromUtf8(""))
        self.xaxisCombo.addItem(_fromUtf8(""))
        self.gridLayout_2.addWidget(self.xaxisCombo, 0, 2, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 6, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.popPlot_delPushButton = QtGui.QPushButton(self.plot_Widget)
        self.popPlot_delPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.popPlot_delPushButton.setText(_fromUtf8(""))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/delete.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.popPlot_delPushButton.setIcon(icon4)
        self.popPlot_delPushButton.setObjectName(_fromUtf8("popPlot_delPushButton"))
        self.horizontalLayout_2.addWidget(self.popPlot_delPushButton)
        self.popPlot_popPushButton = QtGui.QPushButton(self.plot_Widget)
        self.popPlot_popPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.popPlot_popPushButton.setText(_fromUtf8(""))
        self.popPlot_popPushButton.setIcon(icon2)
        self.popPlot_popPushButton.setObjectName(_fromUtf8("popPlot_popPushButton"))
        self.horizontalLayout_2.addWidget(self.popPlot_popPushButton)
        self.popPlot_gridPushButton = QtGui.QPushButton(self.plot_Widget)
        self.popPlot_gridPushButton.setMaximumSize(QtCore.QSize(24, 24))
        self.popPlot_gridPushButton.setText(_fromUtf8(""))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/grid.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.popPlot_gridPushButton.setIcon(icon5)
        self.popPlot_gridPushButton.setObjectName(_fromUtf8("popPlot_gridPushButton"))
        self.horizontalLayout_2.addWidget(self.popPlot_gridPushButton)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.gridLayout_4.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.treeviewLayout = QtGui.QVBoxLayout()
        self.treeviewLayout.setObjectName(_fromUtf8("treeviewLayout"))
        self.gridLayout_4.addLayout(self.treeviewLayout, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.plot_Widget, 1, 0, 1, 1)

        self.retranslateUi(popPlot_Dialog)
        self.titleStackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(popPlot_Dialog)

    def retranslateUi(self, popPlot_Dialog):
        popPlot_Dialog.setWindowTitle(QtGui.QApplication.translate("popPlot_Dialog", "PHOEBE - Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.zoom_in.setText(QtGui.QApplication.translate("popPlot_Dialog", "Zoom In", None, QtGui.QApplication.UnicodeUTF8))
        self.zoom_out.setText(QtGui.QApplication.translate("popPlot_Dialog", "Zoom Out", None, QtGui.QApplication.UnicodeUTF8))
        self.pop_mpl.setText(QtGui.QApplication.translate("popPlot_Dialog", "mpl", None, QtGui.QApplication.UnicodeUTF8))
        self.save.setText(QtGui.QApplication.translate("popPlot_Dialog", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.titleLinkButton.setText(QtGui.QApplication.translate("popPlot_Dialog", "Plot 1", None, QtGui.QApplication.UnicodeUTF8))
        self.title_cancelButton.setText(QtGui.QApplication.translate("popPlot_Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.title_saveButton.setText(QtGui.QApplication.translate("popPlot_Dialog", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popPlot_Dialog", "Y:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popPlot_Dialog", "X:", None, QtGui.QApplication.UnicodeUTF8))
        self.xaxisCombo.setItemText(0, QtGui.QApplication.translate("popPlot_Dialog", "time", None, QtGui.QApplication.UnicodeUTF8))
        self.xaxisCombo.setItemText(1, QtGui.QApplication.translate("popPlot_Dialog", "phase", None, QtGui.QApplication.UnicodeUTF8))
        self.popPlot_delPushButton.setToolTip(QtGui.QApplication.translate("popPlot_Dialog", "delete axes", None, QtGui.QApplication.UnicodeUTF8))
        self.popPlot_popPushButton.setToolTip(QtGui.QApplication.translate("popPlot_Dialog", "popout", None, QtGui.QApplication.UnicodeUTF8))
        self.popPlot_gridPushButton.setToolTip(QtGui.QApplication.translate("popPlot_Dialog", "thumbnail grid", None, QtGui.QApplication.UnicodeUTF8))


class Ui_popPrefs_Dialog(object):
    def setupUi(self, popPrefs_Dialog):
        popPrefs_Dialog.setObjectName(_fromUtf8("popPrefs_Dialog"))
        popPrefs_Dialog.resize(639, 608)
        self.verticalLayout_4 = QtGui.QVBoxLayout(popPrefs_Dialog)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.tabWidget = QtGui.QTabWidget(popPrefs_Dialog)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.gridLayout_5 = QtGui.QGridLayout(self.tab_4)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.p_panel_versions = QtGui.QCheckBox(self.tab_4)
        self.p_panel_versions.setChecked(False)
        self.p_panel_versions.setObjectName(_fromUtf8("p_panel_versions"))
        self.gridLayout_5.addWidget(self.p_panel_versions, 5, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem, 11, 0, 1, 1)
        self.p_panel_python = QtGui.QCheckBox(self.tab_4)
        self.p_panel_python.setChecked(True)
        self.p_panel_python.setObjectName(_fromUtf8("p_panel_python"))
        self.gridLayout_5.addWidget(self.p_panel_python, 7, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.tab_4)
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_5.addWidget(self.label_4, 0, 0, 1, 1)
        self.p_panel_system = QtGui.QCheckBox(self.tab_4)
        self.p_panel_system.setObjectName(_fromUtf8("p_panel_system"))
        self.gridLayout_5.addWidget(self.p_panel_system, 1, 0, 1, 1)
        self.p_panel_fitting = QtGui.QCheckBox(self.tab_4)
        self.p_panel_fitting.setChecked(True)
        self.p_panel_fitting.setObjectName(_fromUtf8("p_panel_fitting"))
        self.gridLayout_5.addWidget(self.p_panel_fitting, 4, 0, 1, 1)
        self.p_panel_datasets = QtGui.QCheckBox(self.tab_4)
        self.p_panel_datasets.setChecked(True)
        self.p_panel_datasets.setObjectName(_fromUtf8("p_panel_datasets"))
        self.gridLayout_5.addWidget(self.p_panel_datasets, 6, 0, 1, 1)
        self.p_panel_params = QtGui.QCheckBox(self.tab_4)
        self.p_panel_params.setChecked(True)
        self.p_panel_params.setObjectName(_fromUtf8("p_panel_params"))
        self.gridLayout_5.addWidget(self.p_panel_params, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.tab_3)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_12 = QtGui.QLabel(self.tab_3)
        self.label_12.setTextFormat(QtCore.Qt.RichText)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.verticalLayout_6.addWidget(self.label_12)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.co_edit_combo = QtGui.QComboBox(self.tab_3)
        self.co_edit_combo.setEnabled(True)
        self.co_edit_combo.setObjectName(_fromUtf8("co_edit_combo"))
        self.horizontalLayout_6.addWidget(self.co_edit_combo)
        self.co_delete = QtGui.QPushButton(self.tab_3)
        self.co_delete.setEnabled(True)
        self.co_delete.setMaximumSize(QtCore.QSize(28, 16777215))
        self.co_delete.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/bin-3.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.co_delete.setIcon(icon)
        self.co_delete.setObjectName(_fromUtf8("co_delete"))
        self.horizontalLayout_6.addWidget(self.co_delete)
        self.co_add = QtGui.QPushButton(self.tab_3)
        self.co_add.setEnabled(True)
        self.co_add.setMaximumSize(QtCore.QSize(28, 16777215))
        self.co_add.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/add.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.co_add.setIcon(icon1)
        self.co_add.setObjectName(_fromUtf8("co_add"))
        self.horizontalLayout_6.addWidget(self.co_add)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.co_psedit = ParameterTreeWidget(self.tab_3)
        self.co_psedit.setEnabled(True)
        self.co_psedit.setIndentation(2)
        self.co_psedit.setObjectName(_fromUtf8("co_psedit"))
        self.co_psedit.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout_6.addWidget(self.co_psedit)
        self.label_13 = QtGui.QLabel(self.tab_3)
        self.label_13.setTextFormat(QtCore.Qt.RichText)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.verticalLayout_6.addWidget(self.label_13)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.fo_edit_combo = QtGui.QComboBox(self.tab_3)
        self.fo_edit_combo.setEnabled(True)
        self.fo_edit_combo.setObjectName(_fromUtf8("fo_edit_combo"))
        self.horizontalLayout_7.addWidget(self.fo_edit_combo)
        self.fo_delete = QtGui.QPushButton(self.tab_3)
        self.fo_delete.setEnabled(True)
        self.fo_delete.setMaximumSize(QtCore.QSize(28, 16777215))
        self.fo_delete.setText(_fromUtf8(""))
        self.fo_delete.setIcon(icon)
        self.fo_delete.setObjectName(_fromUtf8("fo_delete"))
        self.horizontalLayout_7.addWidget(self.fo_delete)
        self.fo_add = QtGui.QPushButton(self.tab_3)
        self.fo_add.setEnabled(False)
        self.fo_add.setMaximumSize(QtCore.QSize(28, 16777215))
        self.fo_add.setText(_fromUtf8(""))
        self.fo_add.setIcon(icon1)
        self.fo_add.setObjectName(_fromUtf8("fo_add"))
        self.horizontalLayout_7.addWidget(self.fo_add)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.fo_psedit = ParameterTreeWidget(self.tab_3)
        self.fo_psedit.setEnabled(True)
        self.fo_psedit.setIndentation(2)
        self.fo_psedit.setObjectName(_fromUtf8("fo_psedit"))
        self.fo_psedit.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout_6.addWidget(self.fo_psedit)
        self.tabWidget.addTab(self.tab_3, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.tab_2)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.label_5 = QtGui.QLabel(self.tab_2)
        self.label_5.setTextFormat(QtCore.Qt.RichText)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_5.addWidget(self.label_5)
        self.servers_config_stackedWidget = QtGui.QStackedWidget(self.tab_2)
        self.servers_config_stackedWidget.setObjectName(_fromUtf8("servers_config_stackedWidget"))
        self.page = QtGui.QWidget()
        self.page.setObjectName(_fromUtf8("page"))
        self.gridLayout_3 = QtGui.QGridLayout(self.page)
        self.gridLayout_3.setMargin(0)
        self.gridLayout_3.setSpacing(2)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.serverlist_treeWidget = ServerListTreeWidget(self.page)
        self.serverlist_treeWidget.setEnabled(True)
        self.serverlist_treeWidget.setIndentation(2)
        self.serverlist_treeWidget.setObjectName(_fromUtf8("serverlist_treeWidget"))
        self.gridLayout_3.addWidget(self.serverlist_treeWidget, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.serverlist_recheck = QtGui.QPushButton(self.page)
        self.serverlist_recheck.setEnabled(True)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/refresh.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.serverlist_recheck.setIcon(icon2)
        self.serverlist_recheck.setObjectName(_fromUtf8("serverlist_recheck"))
        self.horizontalLayout_5.addWidget(self.serverlist_recheck)
        self.serverlist_edit = QtGui.QPushButton(self.page)
        self.serverlist_edit.setEnabled(True)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/pen.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.serverlist_edit.setIcon(icon3)
        self.serverlist_edit.setObjectName(_fromUtf8("serverlist_edit"))
        self.horizontalLayout_5.addWidget(self.serverlist_edit)
        self.serverlist_add = QtGui.QPushButton(self.page)
        self.serverlist_add.setEnabled(True)
        self.serverlist_add.setIcon(icon1)
        self.serverlist_add.setObjectName(_fromUtf8("serverlist_add"))
        self.horizontalLayout_5.addWidget(self.serverlist_add)
        self.gridLayout_3.addLayout(self.horizontalLayout_5, 4, 0, 1, 1)
        self.servers_config_stackedWidget.addWidget(self.page)
        self.page_2 = QtGui.QWidget()
        self.page_2.setObjectName(_fromUtf8("page_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.page_2)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.servers_returntolist = QtGui.QPushButton(self.page_2)
        self.servers_returntolist.setMaximumSize(QtCore.QSize(28, 28))
        self.servers_returntolist.setText(_fromUtf8(""))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/menu-2.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.servers_returntolist.setIcon(icon4)
        self.servers_returntolist.setObjectName(_fromUtf8("servers_returntolist"))
        self.horizontalLayout_3.addWidget(self.servers_returntolist)
        self.sx_serveredit_combo = QtGui.QComboBox(self.page_2)
        self.sx_serveredit_combo.setObjectName(_fromUtf8("sx_serveredit_combo"))
        self.sx_serveredit_combo.addItem(_fromUtf8(""))
        self.horizontalLayout_3.addWidget(self.sx_serveredit_combo)
        self.sx_serveredit_delete = QtGui.QPushButton(self.page_2)
        self.sx_serveredit_delete.setEnabled(True)
        self.sx_serveredit_delete.setMaximumSize(QtCore.QSize(28, 16777215))
        self.sx_serveredit_delete.setText(_fromUtf8(""))
        self.sx_serveredit_delete.setIcon(icon)
        self.sx_serveredit_delete.setObjectName(_fromUtf8("sx_serveredit_delete"))
        self.horizontalLayout_3.addWidget(self.sx_serveredit_delete)
        self.sx_serveredit_add = QtGui.QPushButton(self.page_2)
        self.sx_serveredit_add.setEnabled(True)
        self.sx_serveredit_add.setMaximumSize(QtCore.QSize(28, 16777215))
        self.sx_serveredit_add.setText(_fromUtf8(""))
        self.sx_serveredit_add.setIcon(icon1)
        self.sx_serveredit_add.setObjectName(_fromUtf8("sx_serveredit_add"))
        self.horizontalLayout_3.addWidget(self.sx_serveredit_add)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.sx_serveredit_psedit = ParameterTreeWidget(self.page_2)
        self.sx_serveredit_psedit.setIndentation(2)
        self.sx_serveredit_psedit.setObjectName(_fromUtf8("sx_serveredit_psedit"))
        self.verticalLayout_3.addWidget(self.sx_serveredit_psedit)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_9 = QtGui.QLabel(self.page_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_4.addWidget(self.label_9)
        self.sx_serveredit_mpitype_combo = QtGui.QComboBox(self.page_2)
        self.sx_serveredit_mpitype_combo.setEnabled(False)
        self.sx_serveredit_mpitype_combo.setObjectName(_fromUtf8("sx_serveredit_mpitype_combo"))
        self.sx_serveredit_mpitype_combo.addItem(_fromUtf8(""))
        self.sx_serveredit_mpitype_combo.addItem(_fromUtf8(""))
        self.sx_serveredit_mpitype_combo.addItem(_fromUtf8(""))
        self.horizontalLayout_4.addWidget(self.sx_serveredit_mpitype_combo)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.sx_serveredit_mpipsedit = ParameterTreeWidget(self.page_2)
        self.sx_serveredit_mpipsedit.setIndentation(2)
        self.sx_serveredit_mpipsedit.setObjectName(_fromUtf8("sx_serveredit_mpipsedit"))
        self.verticalLayout_3.addWidget(self.sx_serveredit_mpipsedit)
        self.servers_config_stackedWidget.addWidget(self.page_2)
        self.verticalLayout_5.addWidget(self.servers_config_stackedWidget)
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.gridLayout_2 = QtGui.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.PyInterp_restWidget = QtGui.QWidget(self.tab)
        self.PyInterp_restWidget.setObjectName(_fromUtf8("PyInterp_restWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.PyInterp_restWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_10 = QtGui.QLabel(self.PyInterp_restWidget)
        self.label_10.setTextFormat(QtCore.Qt.RichText)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_2.addWidget(self.label_10)
        self.p_pyinterp_tutsys = QtGui.QCheckBox(self.PyInterp_restWidget)
        self.p_pyinterp_tutsys.setChecked(True)
        self.p_pyinterp_tutsys.setObjectName(_fromUtf8("p_pyinterp_tutsys"))
        self.horizontalLayout_2.addWidget(self.p_pyinterp_tutsys)
        self.p_pyinterp_tutplots = QtGui.QCheckBox(self.PyInterp_restWidget)
        self.p_pyinterp_tutplots.setChecked(False)
        self.p_pyinterp_tutplots.setObjectName(_fromUtf8("p_pyinterp_tutplots"))
        self.horizontalLayout_2.addWidget(self.p_pyinterp_tutplots)
        self.p_pyinterp_tutsettings = QtGui.QCheckBox(self.PyInterp_restWidget)
        self.p_pyinterp_tutsettings.setEnabled(True)
        self.p_pyinterp_tutsettings.setObjectName(_fromUtf8("p_pyinterp_tutsettings"))
        self.horizontalLayout_2.addWidget(self.p_pyinterp_tutsettings)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.label = QtGui.QLabel(self.PyInterp_restWidget)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.px_pyinterp_startup_default = QtGui.QPlainTextEdit(self.PyInterp_restWidget)
        self.px_pyinterp_startup_default.setEnabled(False)
        self.px_pyinterp_startup_default.setFrameShape(QtGui.QFrame.StyledPanel)
        self.px_pyinterp_startup_default.setUndoRedoEnabled(False)
        self.px_pyinterp_startup_default.setPlainText(_fromUtf8(""))
        self.px_pyinterp_startup_default.setObjectName(_fromUtf8("px_pyinterp_startup_default"))
        self.verticalLayout.addWidget(self.px_pyinterp_startup_default)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.p_pyinterp_startup_custom = QtGui.QPlainTextEdit(self.PyInterp_restWidget)
        self.p_pyinterp_startup_custom.setEnabled(True)
        self.p_pyinterp_startup_custom.setObjectName(_fromUtf8("p_pyinterp_startup_custom"))
        self.horizontalLayout.addWidget(self.p_pyinterp_startup_custom)
        self.save_pyinterp_startup_custom = QtGui.QPushButton(self.PyInterp_restWidget)
        self.save_pyinterp_startup_custom.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_pyinterp_startup_custom.sizePolicy().hasHeightForWidth())
        self.save_pyinterp_startup_custom.setSizePolicy(sizePolicy)
        self.save_pyinterp_startup_custom.setMaximumSize(QtCore.QSize(40, 16777215))
        self.save_pyinterp_startup_custom.setObjectName(_fromUtf8("save_pyinterp_startup_custom"))
        self.horizontalLayout.addWidget(self.save_pyinterp_startup_custom)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout_2.addWidget(self.PyInterp_restWidget, 1, 0, 1, 1)
        self.gridLayout_4 = QtGui.QGridLayout()
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.radioButton_5 = QtGui.QRadioButton(self.tab)
        self.radioButton_5.setEnabled(False)
        self.radioButton_5.setChecked(False)
        self.radioButton_5.setObjectName(_fromUtf8("radioButton_5"))
        self.gridLayout_4.addWidget(self.radioButton_5, 2, 2, 1, 1)
        self.radioButton_6 = QtGui.QRadioButton(self.tab)
        self.radioButton_6.setEnabled(False)
        self.radioButton_6.setObjectName(_fromUtf8("radioButton_6"))
        self.gridLayout_4.addWidget(self.radioButton_6, 2, 1, 1, 1)
        self.p_pyinterp_thread_off = QtGui.QRadioButton(self.tab)
        self.p_pyinterp_thread_off.setEnabled(True)
        self.p_pyinterp_thread_off.setObjectName(_fromUtf8("p_pyinterp_thread_off"))
        self.gridLayout_4.addWidget(self.p_pyinterp_thread_off, 1, 2, 1, 1)
        self.label_11 = QtGui.QLabel(self.tab)
        self.label_11.setTextFormat(QtCore.Qt.RichText)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.tab)
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.radioButton_4 = QtGui.QRadioButton(self.tab)
        self.radioButton_4.setEnabled(False)
        self.radioButton_4.setObjectName(_fromUtf8("radioButton_4"))
        self.gridLayout_4.addWidget(self.radioButton_4, 2, 3, 1, 1)
        self.p_pyinterp_thread_on = QtGui.QRadioButton(self.tab)
        self.p_pyinterp_thread_on.setEnabled(True)
        self.p_pyinterp_thread_on.setChecked(True)
        self.p_pyinterp_thread_on.setObjectName(_fromUtf8("p_pyinterp_thread_on"))
        self.gridLayout_4.addWidget(self.p_pyinterp_thread_on, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_4, 10, 0, 1, 1)
        self.copy_panel_python = QtGui.QCheckBox(self.tab)
        self.copy_panel_python.setEnabled(True)
        self.copy_panel_python.setChecked(True)
        self.copy_panel_python.setObjectName(_fromUtf8("copy_panel_python"))
        self.gridLayout_2.addWidget(self.copy_panel_python, 0, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem1, 11, 0, 1, 1)
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_6 = QtGui.QWidget()
        self.tab_6.setObjectName(_fromUtf8("tab_6"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.tab_6)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.label_2 = QtGui.QLabel(self.tab_6)
        self.label_2.setTextFormat(QtCore.Qt.RichText)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_7.addWidget(self.label_2)
        self.lo_psedit = ParameterTreeWidget(self.tab_6)
        self.lo_psedit.setIndentation(2)
        self.lo_psedit.setObjectName(_fromUtf8("lo_psedit"))
        self.lo_psedit.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout_7.addWidget(self.lo_psedit)
        self.tabWidget.addTab(self.tab_6, _fromUtf8(""))
        self.verticalLayout_4.addWidget(self.tabWidget)
        self.buttonBox = QtGui.QDialogButtonBox(popPrefs_Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Close|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout_4.addWidget(self.buttonBox)

        self.retranslateUi(popPrefs_Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.servers_config_stackedWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popPrefs_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popPrefs_Dialog.reject)
        QtCore.QObject.connect(self.copy_panel_python, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.PyInterp_restWidget.setEnabled)
        QtCore.QObject.connect(self.copy_panel_python, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.p_panel_python.setChecked)
        QtCore.QObject.connect(self.p_panel_python, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.copy_panel_python.setChecked)
        QtCore.QMetaObject.connectSlotsByName(popPrefs_Dialog)

    def retranslateUi(self, popPrefs_Dialog):
        popPrefs_Dialog.setWindowTitle(QtGui.QApplication.translate("popPrefs_Dialog", "PHOEBE - Preferences", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_versions.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Versions", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_python.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Console", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Default Panels:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_system.setText(QtGui.QApplication.translate("popPrefs_Dialog", "System", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_fitting.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_datasets.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Data and Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.p_panel_params.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QtGui.QApplication.translate("popPrefs_Dialog", "Interface", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Compute Options:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Fitting Options:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QtGui.QApplication.translate("popPrefs_Dialog", "Defaults", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Configure Servers:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.serverlist_treeWidget.headerItem().setText(0, QtGui.QApplication.translate("popPrefs_Dialog", "Server", None, QtGui.QApplication.UnicodeUTF8))
        self.serverlist_treeWidget.headerItem().setText(1, QtGui.QApplication.translate("popPrefs_Dialog", "MPI", None, QtGui.QApplication.UnicodeUTF8))
        self.serverlist_recheck.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Recheck Connections", None, QtGui.QApplication.UnicodeUTF8))
        self.serverlist_edit.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Edit Servers", None, QtGui.QApplication.UnicodeUTF8))
        self.serverlist_add.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Add Server", None, QtGui.QApplication.UnicodeUTF8))
        self.servers_returntolist.setToolTip(QtGui.QApplication.translate("popPrefs_Dialog", "return to server list", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_combo.setItemText(0, QtGui.QApplication.translate("popPrefs_Dialog", "servername", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_delete.setToolTip(QtGui.QApplication.translate("popPrefs_Dialog", "delete server", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_add.setToolTip(QtGui.QApplication.translate("popPrefs_Dialog", "add new server", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_psedit.headerItem().setText(0, QtGui.QApplication.translate("popPrefs_Dialog", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_psedit.headerItem().setText(1, QtGui.QApplication.translate("popPrefs_Dialog", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("popPrefs_Dialog", "MPI type:", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_mpitype_combo.setItemText(0, QtGui.QApplication.translate("popPrefs_Dialog", "mpi", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_mpitype_combo.setItemText(1, QtGui.QApplication.translate("popPrefs_Dialog", "mpi:slurm", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_mpitype_combo.setItemText(2, QtGui.QApplication.translate("popPrefs_Dialog", "mpi:torque", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_mpipsedit.headerItem().setText(0, QtGui.QApplication.translate("popPrefs_Dialog", "Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.sx_serveredit_mpipsedit.headerItem().setText(1, QtGui.QApplication.translate("popPrefs_Dialog", "Value", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("popPrefs_Dialog", "Servers", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Tutorial Mode:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_tutsys.setText(QtGui.QApplication.translate("popPrefs_Dialog", "System", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_tutplots.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_tutsettings.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Startup Script:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_startup_custom.setPlainText(QtGui.QApplication.translate("popPrefs_Dialog", "import numpy as np", None, QtGui.QApplication.UnicodeUTF8))
        self.save_pyinterp_startup_custom.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_5.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Python Tab", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_6.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Terminal", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_thread_off.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Disabled", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Threading:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Redirect stderr:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_4.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Log Tab", None, QtGui.QApplication.UnicodeUTF8))
        self.p_pyinterp_thread_on.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Enabled", None, QtGui.QApplication.UnicodeUTF8))
        self.copy_panel_python.setText(QtGui.QApplication.translate("popPrefs_Dialog", "Enable Python Interface", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("popPrefs_Dialog", "Python Interface", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popPrefs_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Logger:</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), QtGui.QApplication.translate("popPrefs_Dialog", "Logger", None, QtGui.QApplication.UnicodeUTF8))

from phoebe_widgets import ServerListTreeWidget, ParameterTreeWidget

class Ui_popTimeSelect_Dialog(object):
    def setupUi(self, popTimeSelect_Dialog):
        popTimeSelect_Dialog.setObjectName(_fromUtf8("popTimeSelect_Dialog"))
        popTimeSelect_Dialog.resize(400, 300)
        self.gridLayout = QtGui.QGridLayout(popTimeSelect_Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.compute_max = QtGui.QPushButton(popTimeSelect_Dialog)
        self.compute_max.setObjectName(_fromUtf8("compute_max"))
        self.gridLayout.addWidget(self.compute_max, 1, 2, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(popTimeSelect_Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 4)
        self.compute_mean = QtGui.QPushButton(popTimeSelect_Dialog)
        self.compute_mean.setObjectName(_fromUtf8("compute_mean"))
        self.gridLayout.addWidget(self.compute_mean, 1, 4, 1, 1)
        self.label_2 = QtGui.QLabel(popTimeSelect_Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.compute_median = QtGui.QPushButton(popTimeSelect_Dialog)
        self.compute_median.setObjectName(_fromUtf8("compute_median"))
        self.gridLayout.addWidget(self.compute_median, 1, 3, 1, 1)
        self.time = QtGui.QLineEdit(popTimeSelect_Dialog)
        self.time.setReadOnly(True)
        self.time.setObjectName(_fromUtf8("time"))
        self.gridLayout.addWidget(self.time, 2, 1, 1, 4)
        self.datasetComboBox = QtGui.QComboBox(popTimeSelect_Dialog)
        self.datasetComboBox.setObjectName(_fromUtf8("datasetComboBox"))
        self.gridLayout.addWidget(self.datasetComboBox, 0, 1, 1, 4)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 1, 1, 1)
        self.label = QtGui.QLabel(popTimeSelect_Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtGui.QLabel(popTimeSelect_Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.compute_min = QtGui.QPushButton(popTimeSelect_Dialog)
        self.compute_min.setObjectName(_fromUtf8("compute_min"))
        self.gridLayout.addWidget(self.compute_min, 1, 1, 1, 1)

        self.retranslateUi(popTimeSelect_Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), popTimeSelect_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), popTimeSelect_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(popTimeSelect_Dialog)

    def retranslateUi(self, popTimeSelect_Dialog):
        popTimeSelect_Dialog.setWindowTitle(QtGui.QApplication.translate("popTimeSelect_Dialog", "PHOEBE - Select Time", None, QtGui.QApplication.UnicodeUTF8))
        self.compute_max.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "max", None, QtGui.QApplication.UnicodeUTF8))
        self.compute_mean.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "mean", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "Compute:", None, QtGui.QApplication.UnicodeUTF8))
        self.compute_median.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "median", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "Dataset:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "Time:", None, QtGui.QApplication.UnicodeUTF8))
        self.compute_min.setText(QtGui.QApplication.translate("popTimeSelect_Dialog", "min", None, QtGui.QApplication.UnicodeUTF8))


class Ui_datasetWidget(object):
    def setupUi(self, datasetWidget):
        datasetWidget.setObjectName(_fromUtf8("datasetWidget"))
        datasetWidget.resize(956, 378)
        self.verticalLayout = QtGui.QVBoxLayout(datasetWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.selectorWidget = QtGui.QWidget(datasetWidget)
        self.selectorWidget.setObjectName(_fromUtf8("selectorWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.selectorWidget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.ds_typeComboBox = QtGui.QComboBox(self.selectorWidget)
        self.ds_typeComboBox.setObjectName(_fromUtf8("ds_typeComboBox"))
        self.ds_typeComboBox.addItem(_fromUtf8(""))
        self.ds_typeComboBox.addItem(_fromUtf8(""))
        self.ds_typeComboBox.addItem(_fromUtf8(""))
        self.ds_typeComboBox.addItem(_fromUtf8(""))
        self.ds_typeComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.ds_typeComboBox)
        self.ds_plotComboBox = QtGui.QComboBox(self.selectorWidget)
        self.ds_plotComboBox.setObjectName(_fromUtf8("ds_plotComboBox"))
        self.ds_plotComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.ds_plotComboBox)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addWidget(self.selectorWidget)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.datasetTreeView = DatasetTreeWidget(datasetWidget)
        self.datasetTreeView.setIndentation(2)
        self.datasetTreeView.setObjectName(_fromUtf8("datasetTreeView"))
        self.datasetTreeView.headerItem().setText(0, _fromUtf8("1"))
        self.horizontalLayout_2.addWidget(self.datasetTreeView)
        self.addDataWidget = QtGui.QWidget(datasetWidget)
        self.addDataWidget.setObjectName(_fromUtf8("addDataWidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.addDataWidget)
        self.verticalLayout_4.setSpacing(2)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.addDataButton = QtGui.QCommandLinkButton(self.addDataWidget)
        self.addDataButton.setMaximumSize(QtCore.QSize(36, 36))
        self.addDataButton.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/add.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addDataButton.setIcon(icon)
        self.addDataButton.setObjectName(_fromUtf8("addDataButton"))
        self.verticalLayout_4.addWidget(self.addDataButton)
        self.horizontalLayout_2.addWidget(self.addDataWidget)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(datasetWidget)
        QtCore.QMetaObject.connectSlotsByName(datasetWidget)

    def retranslateUi(self, datasetWidget):
        datasetWidget.setWindowTitle(QtGui.QApplication.translate("datasetWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_typeComboBox.setItemText(0, QtGui.QApplication.translate("datasetWidget", "all types", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_typeComboBox.setItemText(1, QtGui.QApplication.translate("datasetWidget", "lc", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_typeComboBox.setItemText(2, QtGui.QApplication.translate("datasetWidget", "rv", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_typeComboBox.setItemText(3, QtGui.QApplication.translate("datasetWidget", "sp", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_typeComboBox.setItemText(4, QtGui.QApplication.translate("datasetWidget", "etv", None, QtGui.QApplication.UnicodeUTF8))
        self.ds_plotComboBox.setItemText(0, QtGui.QApplication.translate("datasetWidget", "all plots", None, QtGui.QApplication.UnicodeUTF8))

from phoebe_widgets import DatasetTreeWidget

qt_resource_data = "\
\x00\x00\x0e\xdb\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x30\x00\x00\x00\x30\x08\x06\x00\x00\x00\x57\x02\xf9\x87\
\x00\x00\x00\x06\x62\x4b\x47\x44\x00\xff\x00\xff\x00\xff\xa0\xbd\
\xa7\x93\x00\x00\x00\x09\x70\x48\x59\x73\x00\x00\x0b\x13\x00\x00\
\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07\x74\x49\x4d\x45\x07\
\xd4\x0c\x02\x11\x0a\x1f\x40\x81\xfd\x2b\x00\x00\x00\x1d\x74\x45\
\x58\x74\x43\x6f\x6d\x6d\x65\x6e\x74\x00\x43\x72\x65\x61\x74\x65\
\x64\x20\x77\x69\x74\x68\x20\x54\x68\x65\x20\x47\x49\x4d\x50\xef\
\x64\x25\x6e\x00\x00\x0e\x3f\x49\x44\x41\x54\x68\xde\xed\x99\x79\
\x90\x5c\xd5\x75\xc6\x7f\xe7\xde\xd7\xdd\xb3\x48\x23\x8d\x85\xd0\
\x2e\xc1\x68\x21\x58\x02\x04\x5a\x30\x91\x45\x8c\x31\x46\x94\x21\
\x76\x58\x5c\xc4\x89\xd9\x4d\x9c\x54\x61\x8c\x97\x54\xb9\xc0\xce\
\x62\xcb\x0e\xae\x50\x76\x25\x15\x83\xcb\x89\x21\xc6\xc1\x80\x08\
\x54\x1c\xe2\x18\x0a\xc2\x6e\x24\x5b\x2e\x40\x0b\x58\xa3\x91\x04\
\x68\x1b\xcd\x22\xcd\xf4\xcc\x74\xf7\x7b\xef\x2e\xf9\xe3\xde\x6e\
\xb5\x06\x89\x90\x00\x29\xfe\x70\x57\xbd\xea\xd7\xdd\xaf\x5f\x9f\
\xef\xdc\xef\x7c\xe7\x3b\xb7\xe1\xb7\x8f\xdf\x3e\xde\xd6\x43\xde\
\xab\x81\xfd\xe8\x5f\x1e\xb8\xea\x37\xdb\x76\xce\xdc\xbc\x65\xeb\
\xae\xbe\xde\xde\x9d\xc7\x4f\xed\xdc\x7b\xc9\x25\x1f\x3f\x70\xd5\
\x15\x7f\x98\xbf\x67\x01\x6c\xeb\xde\x39\x6f\xc7\xae\x3d\x77\xfe\
\xf4\x3f\x9e\x48\x72\x63\x16\x56\xc6\x46\x27\x0d\x0c\xf4\x33\x52\
\x1e\xa2\x73\x72\xbb\x5e\xd8\x35\xaf\xdc\xda\x56\xfa\xe2\xaa\x55\
\x2b\x7f\xf6\xfb\x17\x5e\x34\xf0\x9e\x02\xf0\xec\x73\x2f\xae\xd9\
\x77\x60\xe0\x8e\xe7\x7f\xb5\x79\xf6\xbe\x7d\x03\x92\xe5\x99\xb2\
\x36\xc7\x9a\x8c\x2c\xad\x32\x3a\x32\xc4\xe0\x40\x2f\x93\x27\xb7\
\x72\xee\x39\x1f\x7c\xed\xf7\xce\x5e\x75\xed\x9a\x8f\x9e\xff\xb8\
\x7e\x2f\x04\xef\xbd\xd7\xeb\xfe\xed\xd9\x4b\xb7\xf5\xec\xbf\x64\
\xa8\x9c\xa9\x52\x4b\xbb\x94\x5a\x27\x50\x6a\x69\xa3\x90\x94\x50\
\x3a\x21\x49\x0a\xb4\xb6\xb6\x50\x1e\x2e\xf3\xfa\xee\xd7\x26\x0f\
\x0e\xf4\x4f\xf8\xda\xd7\x6e\x7e\xec\xff\x1d\xc0\x03\x0f\xfd\xfc\
\xc9\x53\x4f\xff\xc0\xa7\xe6\xcd\x5f\x72\xe1\xcc\xb9\x0b\xcf\x9f\
\x31\x7b\xd1\xea\xbe\xc1\xb1\xcf\x3b\x29\x5d\x3e\x70\xa8\x56\x6a\
\x69\xeb\x90\xd6\xf6\x89\x14\x63\xf0\xba\x50\x40\x74\x08\xd3\x39\
\x87\x08\x94\xcb\xc3\x54\x46\x47\x3a\x8e\x9f\x32\xa5\x27\x79\xb7\
\x03\x7e\x6e\xfd\x4b\x53\xb7\x75\xef\xec\x7b\xfc\xc9\x0d\x03\x87\
\x86\x86\xb3\x75\x0f\x3d\x3a\x33\xcf\x72\xd2\xac\x46\x9e\x39\x44\
\x29\x86\x47\x32\x26\x4c\x2a\x31\xa9\x73\x1a\x3a\x29\xa1\x13\x85\
\x73\x96\x3c\xab\x51\xab\x96\x10\x51\x78\x6b\xb1\x79\x4e\x96\x55\
\x29\x95\xda\xc9\xb2\x7c\xd6\x53\x4f\x3d\x73\xd6\x3b\x0e\xe0\x40\
\xff\x60\x47\x2d\x4f\xd3\x47\x1e\xf9\xe5\xa7\x06\x0f\x0e\xfd\xf0\
\xd6\xef\xdc\x65\x95\x56\xe0\xfc\x71\x49\x52\x64\xac\x92\x61\x9d\
\xc5\x39\x10\x5d\x20\x41\xe8\x1b\xac\xa0\x5b\x15\x6d\x13\x27\x52\
\x2a\xb5\x91\x24\x05\xac\x33\xd4\xaa\xa3\x88\x12\xac\x35\xe4\x59\
\x8d\x42\xb1\x42\x31\x29\x51\x2a\xb5\x50\x1e\x1a\xa6\xaf\xbf\xb7\
\xf0\x8e\x02\xb8\xf7\xc1\xc7\x26\x6d\xda\xba\xf3\xde\x8d\x2f\xbc\
\xf2\x91\xcd\x5b\x76\x90\xe7\x96\xd6\xd6\x76\xed\xbd\xc7\x79\x8b\
\xb3\x16\x8f\x47\x5b\x8b\xb7\x06\xeb\x2c\x4a\x69\x0a\xa5\x16\x4a\
\x2d\x6d\x94\x5a\xda\x69\x6b\xef\xa0\x58\x6a\x23\xcf\x6b\x78\xe7\
\xc8\xd2\x1a\x49\x52\x22\xd1\x09\x4a\x6b\x44\x69\x94\x68\x0a\x85\
\xa2\x9b\xdf\x75\xe2\xe8\x3b\x06\xe0\x9f\xef\x79\xe4\xf4\x43\xc3\
\x63\xf7\x3f\xf3\xfc\xd6\x05\x43\xc3\x23\x88\x2e\x50\xd4\x45\xf0\
\x0e\xef\x1d\xce\x79\xd0\x0e\xe7\x2c\x56\x59\x9c\xd6\x88\x31\x88\
\x28\x4a\xc5\x89\x78\xaf\x51\x4a\xc2\x75\x78\x70\xe0\x43\x89\xe3\
\x71\x38\xef\xe3\xbd\x3c\x0e\x47\x4b\x4b\x2b\x17\x5d\xf8\xb1\x17\
\x93\xb7\xd8\xe0\xde\x20\xb7\x97\x5d\x76\x99\xac\x5b\xb7\xce\xdf\
\x79\xd7\xba\x62\xea\x5a\x2f\x7b\x7d\xef\xc0\x6d\x7b\xf6\xf7\x4d\
\xad\xd5\x32\x0a\x85\x52\x08\x02\x8f\xf3\x1e\xef\x40\x7b\x8f\x73\
\x06\x71\x1a\xe5\x2c\xd6\x29\x1c\x21\xa3\xfb\x77\x6f\xa6\xb5\x7d\
\x12\xce\x39\x5a\xdb\x1d\x79\x56\xc5\x98\x9c\x5a\x75\x94\x2c\xad\
\x92\x65\x35\x72\x93\x62\x4c\x8e\xb5\x19\x23\xe5\x32\xa7\x2d\x59\
\xf0\xd0\xe7\x6f\xbc\xe1\x27\xc7\x0a\xf4\x58\xcf\x47\x9c\x3f\xfc\
\xf0\x63\x33\xf6\xf6\xd7\xfe\x6c\x6f\xef\xc1\xcf\xf6\x0d\x0c\x4f\
\x4a\x53\x83\xd6\x2a\x66\x2e\x66\xcf\x7b\xc0\xe1\xac\xc7\x39\x87\
\xf7\x06\xe7\x2c\xce\x3a\xac\xcd\x71\x36\xa7\xa5\x54\x20\xcf\x2d\
\x27\x2f\x3d\x1f\x4f\x46\x9e\x39\xc0\x51\xab\x8e\x91\xe7\x63\x54\
\x46\x86\x18\x1d\x19\xe4\xe0\x40\x2f\xb5\xb1\x21\x4e\x3b\x75\x81\
\xfb\x83\x8b\xce\xbf\xf2\xdc\x73\x3f\xf4\xe3\xe4\x28\x81\x8e\x3f\
\x1f\x7f\x00\xb0\x79\xf3\xe6\x33\x7f\xbd\x79\xdf\x0d\x7b\xf6\x1f\
\x5c\xd3\x3f\x58\x6e\x33\xc6\x53\x2c\x16\x9b\x82\x97\x00\xc0\x79\
\x04\x8b\xc3\xe3\x94\xc3\x3b\x85\x13\x8b\xc5\x20\x02\x06\xa1\x9a\
\x66\x28\x60\xdb\xa6\xc7\x28\x14\x0a\xcc\x39\xf1\x64\x86\x86\x47\
\xd1\xca\x31\x3a\x52\xa6\x3a\x56\xc6\x66\x15\x66\xcd\x9a\xcb\xf6\
\xad\x7b\x38\x69\xe1\xdc\xed\xab\xcf\xfe\xe0\x4f\x00\xa4\x6b\xd9\
\x9f\xf8\xa6\x96\x02\xfe\x30\xf3\x62\x97\xa1\xc1\x46\xef\xdf\x72\
\x4d\xbc\xaf\xb3\x83\x15\xcb\x4e\x6e\x80\xf0\xde\xe3\xbd\xc3\x3b\
\x8b\x73\x16\xef\x2c\xd6\x1a\xac\xcd\x31\xd6\x60\xf3\x0c\xef\x6d\
\xc8\x90\x77\x14\x0a\x45\x4a\x2d\xad\xe4\x59\xca\xc1\x83\x7d\x58\
\x53\xc1\x3b\xa1\xa7\xfb\x05\xa6\x4e\xe9\xe4\xc1\x07\xef\x9d\x37\
\x67\xf6\x9c\xd7\x13\x80\x1d\x1b\xef\x78\x57\x7a\xc0\xb2\x73\x3e\
\xc7\xca\x15\x4b\x40\x85\x02\x14\xa7\xf0\xb1\x75\x3a\x40\x21\xe0\
\x03\x40\x74\x82\x31\x8e\xdc\x64\x28\xf1\x54\x87\xc7\xb0\x07\xfb\
\xa8\x55\x47\xa9\x56\xc7\xc8\xb3\x0a\x59\x56\xa3\x58\x2c\xd1\x3f\
\x30\xc0\x57\xbe\x72\xcb\xb7\x81\xcb\x1b\x45\xfc\xc3\x1f\xfd\x3b\
\x88\xa0\x94\x20\x02\x22\x0a\x11\x41\x44\x50\xf1\xb5\x12\x10\x25\
\xf1\x33\x40\x09\x8a\x70\x8d\xd4\xcf\x95\xa0\x44\xd0\x0a\x16\xcd\
\x9f\xc6\xcf\x7e\xfe\x14\x17\x7c\x74\x75\x58\x09\x01\xef\xc3\xf7\
\x95\x0a\x2b\xeb\x75\x82\x8a\xb5\xa2\xb4\x43\x9c\x21\xcf\x33\xac\
\x73\xe4\x26\xc7\x79\x17\x62\x52\x1a\x15\x63\xb2\xd6\x71\xe8\xd0\
\x50\xd7\xad\xb7\xde\x56\x8c\x14\xf2\x9c\xb5\x7c\x51\xa4\x88\x67\
\xcd\x87\x96\xc4\x40\x43\x30\xd2\x74\xf4\xf5\xed\x07\xef\x79\xb9\
\xa7\x9f\xc1\x83\xc3\x80\xf0\x47\x97\x9e\x03\x4a\xc5\x6b\x03\x38\
\x25\x02\x4a\xf1\x99\xcf\xdd\x06\x22\xac\x39\x6f\xf5\x61\x4a\x7a\
\x87\x27\x50\xca\x5a\x87\xb3\x06\x6b\xb3\x10\xb8\xc9\xc8\xb3\x14\
\x63\x52\xf2\xac\x46\x96\xa5\xe4\x59\x85\x3c\xad\x91\x65\x55\xb2\
\xb4\x42\x65\x6c\xd4\xcf\x9c\x7e\x9c\xac\x5f\xff\x9c\x08\xa0\x01\
\x15\x9f\x0b\x27\x2c\xbd\xaa\xbc\xeb\x85\x3b\xb9\xe7\xfe\x47\x63\
\x36\x41\x50\x21\xdb\x02\xaf\xbd\xba\x83\xfb\x7f\xba\x9e\xb9\x73\
\x67\x51\x28\x14\x79\xf2\xe9\x0d\x08\xc2\x1d\xdf\xf9\x02\x3a\x29\
\x1e\x01\xb6\xbe\x72\x57\xfe\xe9\x5a\x44\x14\xe7\x7d\x64\xf5\x11\
\x72\x56\x4f\x98\xb5\x06\x67\x83\x44\x9a\x2c\x23\xcf\x53\x4c\x9e\
\x06\xf9\xcc\x42\xe0\x87\x01\x54\xc9\xb3\x0a\xb5\xea\x58\x7a\xdd\
\x35\x57\x5d\xaf\x9a\xd4\x45\x01\xaa\x5e\xc4\x4a\x09\x5a\x04\x51\
\x2a\x9c\x2b\x41\x29\xc5\xf3\x1b\xbb\x99\x35\x73\x0a\x26\x0f\x36\
\x77\xf5\xaa\x65\x4c\x9f\x36\x85\xeb\x6f\xbc\x15\x25\xa0\x24\x5c\
\xaf\x9a\xbe\x7b\xf7\xf7\x6f\xc1\xd9\x94\x3c\xcf\xc8\x4d\x86\x31\
\x39\xce\xd9\xb0\x12\xce\x45\x38\x31\x87\x4a\x85\x43\x14\x4a\x74\
\x23\x19\x44\x46\x84\x68\x85\xce\xce\xce\x52\xff\x40\xff\x62\x35\
\x4e\x32\x55\x5d\x7d\x94\x52\x21\x00\x11\x44\x4b\xe0\x9f\x12\xac\
\xcd\xa9\x54\x2a\x21\x2b\x31\x43\xf3\xbb\xe6\xe2\x6c\xce\xca\xe5\
\x8b\x23\x7d\x14\x4a\x2b\x94\x0a\x87\x56\x9a\xb3\x56\xbe\x9f\xad\
\x5b\xb6\x90\xa7\x29\x26\x82\x30\xd6\xe2\xbc\x0d\xea\x84\x8b\x6a\
\x27\x08\x02\x22\x38\x3c\x88\x8a\xef\x1c\x56\x77\xef\x3c\x69\x96\
\xb2\x69\xd3\x16\xab\xc6\x35\x28\x69\xac\x40\x2c\x68\xa5\x54\x08\
\x5e\x83\x56\x0a\xef\x72\xb2\x5a\x58\xca\x34\xab\x91\xe7\x19\xde\
\x19\x66\xce\x9c\xc6\x2d\xdf\xf8\x01\x6d\xed\xad\x21\x78\x11\x74\
\x7d\x35\xb4\x70\xed\xa7\x2f\x64\xef\xde\x3d\x64\x69\x25\xd0\x20\
\x0f\x40\xac\xb5\x38\xe7\x1a\xf2\x2d\xf8\xc3\xd9\x14\x41\xc4\x83\
\x92\x86\x94\x2b\x3c\x22\x50\xab\x65\xec\xd9\xb3\xdb\x1e\xc5\x4a\
\xc4\x1b\x69\x15\x55\x85\x23\x78\xed\x9c\xc5\xe4\x35\x9c\xd5\x78\
\xe7\x10\x84\x5c\x69\x16\x2d\x5a\x40\xa9\x50\x60\xfe\x89\xb3\xe8\
\xd9\xb1\x1b\xe7\x7c\x54\xb3\x50\xd8\xed\x6d\x13\x58\xb5\x72\x09\
\x69\x56\x25\xd1\x09\xce\x59\x74\x52\x40\x2b\x8d\x88\x6a\xf8\x9c\
\xc3\xad\x26\x9e\x7b\xa1\xd1\x1d\x3d\x87\xe1\x79\x6b\x73\x6b\x77\
\x24\x6f\x88\xbc\xce\xc8\x63\xa8\x10\xde\x62\xf2\x2c\x38\x43\x11\
\xb4\x4e\x70\xde\xe3\x9c\xa7\x7f\xf0\x60\x90\x57\x1f\xbe\x8f\x28\
\x74\xe4\xac\x13\x70\xde\x90\xa5\x16\x9b\x14\x48\x9c\xa3\xe0\x1d\
\x5e\x17\xd0\x4a\x81\x08\xde\x5b\x20\x98\x3f\xef\x01\x09\xc6\xad\
\xd1\x5c\x85\x46\x43\x2c\x14\x0a\x7a\xe2\xc4\xc9\xcf\x26\x0d\xd7\
\xd5\x00\x21\x91\x42\x0a\x51\x71\x19\x9b\xf4\x3d\x74\xd0\x1c\x8f\
\x43\x94\x26\x71\xe1\x07\xf1\x30\x7d\xda\x71\x14\x0b\x05\xa8\xf7\
\x85\x46\xff\x10\x86\x47\x47\xf0\x71\x48\xf1\xd6\x84\xd5\xf3\x1e\
\x5f\x88\xbd\x40\xc2\xef\x5a\x57\xef\xdc\x2e\x3c\xbb\x68\x08\xbd\
\x07\x67\xf1\xde\x63\xad\x63\xce\xec\x99\x3c\xf3\xf4\x13\x3d\xaa\
\x29\xfb\xcd\x40\x50\x0a\xb4\xc4\x42\x8e\x45\xac\x94\x8a\x37\x30\
\x58\x5b\x2f\xbe\x48\x39\xa5\xf8\xdd\x33\x97\x92\xe7\x79\xbc\x5e\
\x50\x4a\xc7\x3a\x12\x6e\xff\xc1\x8f\x99\xd4\x31\x01\x93\xa7\x51\
\xef\x63\x0d\x18\x83\x73\x26\x0e\x39\x16\x9c\x8d\x59\xaf\x1f\x1e\
\x7c\x30\x7f\x3e\x5a\xf3\x6a\xb5\xe2\x97\x2f\x5f\x66\xbd\xf7\x05\
\x75\x64\xf6\x63\x85\x40\x90\xb0\x28\xa5\x2a\x82\x10\x91\xe0\x2c\
\x5d\x90\xbf\xfa\xb3\xf7\x9e\xff\x7a\xec\x51\x3e\x7c\xf6\xe9\x94\
\x47\x2a\x28\x51\x68\x5d\x97\x53\x15\x33\xe8\xe8\xee\xd9\x15\xd4\
\xc7\xe4\x98\xdc\x04\x39\xb5\x39\xce\x18\x9c\x0d\x00\x02\x90\xe0\
\xff\x1d\xf5\xdf\xf1\x78\x6c\x78\xcf\x5b\x8a\xc5\x44\x9e\x7c\xfa\
\xe9\x8f\x89\x48\x7e\x94\x15\x08\x00\xb4\x26\x06\xae\x1a\xd9\x57\
\x5a\x35\xba\xa9\x8f\x19\xf2\xde\xd3\x77\xe0\x00\xdd\x9b\x1f\xa1\
\x63\x62\x1b\xd5\x6a\xad\x11\x78\x5d\xc9\xd6\xfe\xcd\x77\x29\x95\
\x4a\x38\xeb\x1a\x81\x3a\x1b\x6c\xb5\xb5\xb6\x91\x7d\xdb\xfc\x59\
\x04\x65\xa3\xf1\xab\x7f\x37\xcf\x52\xbf\x70\xc1\x7c\x2e\xfe\xc4\
\xc7\x15\xc0\xf8\x22\xf6\x0d\xb5\x15\x8d\x52\x41\xd8\x94\x56\xa1\
\xf6\xc3\x1b\x6c\xef\xd9\x11\x69\xa3\x51\xba\xc8\x50\xef\x4b\xc1\
\x62\xbf\xbc\x93\x34\xcd\xd1\x81\xfc\x88\xc0\x5f\xaf\xfd\x2e\x93\
\x27\x4f\x62\x5b\xf7\x8e\x40\xb7\xba\x2b\xc5\xe3\x5d\xa0\xa1\xb3\
\xae\xa1\xf2\xce\x59\x9c\x0f\x93\x9b\xb7\x16\xef\x4c\x18\x47\x23\
\x10\x93\x1b\xe9\x1f\xe8\xbf\xfd\xab\xb7\xdc\xfc\x9f\xf5\xf6\x37\
\xae\x88\x9b\x55\xa8\xbe\x02\x71\x35\x10\xfe\xfc\x4b\x37\x31\x32\
\xd8\xcd\xc8\x60\x37\xe5\xfe\x57\x1a\xc1\x6f\xd8\xb8\x95\x5a\x35\
\x0d\x1e\x4a\x29\x0e\x1d\x1c\xe0\xeb\xdf\xfa\x7b\x3a\x3a\x3a\xd8\
\xb9\x6b\x4f\x43\xe3\xeb\x14\xf5\xde\xe3\x91\xd8\x89\x1d\xbe\x1e\
\xa4\x0f\x81\x3b\x6f\xb1\xb1\x36\x82\xd5\x30\xe4\x79\xca\xf4\xe9\
\xc7\xe7\x57\x7c\xfa\x8f\xff\xb5\x1e\xe7\x9b\xc8\x68\x30\x65\x12\
\xe9\x10\x56\x20\xd4\xc1\x2f\xd6\x6f\x6a\x38\x50\x11\xc1\x59\x87\
\xc9\x53\xca\xe5\x21\xfe\xf1\xee\x87\x1b\x73\xc5\xf4\xe9\xd3\xd8\
\xd6\xbd\x33\x96\x56\x34\x7a\xf5\xc4\x28\x15\xb2\x27\xd1\xd4\x39\
\x15\xeb\xcb\x34\x02\x76\xd6\xe2\x4c\x78\x6d\x5d\x10\x8e\x95\xcb\
\x97\x3d\xf2\xe5\x2f\x7d\xf1\xf1\xa3\x01\xa0\xb9\x88\xa5\x61\xad\
\x25\x50\x29\x5a\xd9\x6f\xdd\x7a\x1b\x38\x1b\x51\xea\x58\xec\x09\
\xa2\x12\x94\x68\xa6\x1e\x3f\x95\xfd\x7d\x43\x58\x9b\xb3\x73\xd7\
\xde\x98\x16\x15\x57\xb3\x6e\x33\x62\xf3\x12\x41\xbc\x04\x89\xf4\
\x75\x3a\x59\x9c\x33\x18\x63\x30\x36\x0f\x63\xa7\x31\xa4\xb5\x2a\
\x67\x9e\xb9\xdc\x7e\xe1\xa6\x1b\x3f\xf3\xbd\xef\xfd\x1d\xc7\x02\
\xd0\x78\x68\x7d\x64\x03\xab\x67\x1b\xef\xd9\xf5\xea\xeb\xa1\x89\
\x25\x05\x4a\xa5\x56\x8a\xa5\xb6\xb0\x0d\x58\x6c\x61\xcf\xbe\xfe\
\xe8\x57\x82\x59\x93\x48\x47\x48\xc0\xc7\xee\xac\x13\x54\xa2\x51\
\x3a\x9a\xb3\xa8\xf3\x41\x71\x2c\xb6\x49\x9d\x8c\x31\xd4\xaa\x55\
\xbf\x60\x41\x97\x9c\xb5\x62\xc5\x92\x05\x0b\xe7\xf7\x36\xc7\x79\
\x4c\x00\x2a\xa4\xbd\x69\xc0\x91\xf8\x1e\x8d\x26\x13\x86\x8b\x38\
\x16\x9a\x0c\x89\x7d\x42\xa9\x60\xb8\x42\xf2\x05\x45\x82\x88\xc5\
\x37\x3c\x56\x58\xb9\xba\xe2\x79\xef\x23\xff\x0d\xd6\xe4\x58\x1b\
\xb2\x6f\x4c\x46\x5a\xab\xf8\x79\xf3\x66\xc9\xe2\xf7\x9f\xfc\xe1\
\x1b\x6f\xba\xe1\x37\xe3\xe3\x3c\x36\x00\x51\xd0\x14\x78\x63\x48\
\x89\x33\xab\xf3\x1e\x71\x2e\x72\xf5\xf0\xa1\x44\xe1\xd1\xf8\xc8\
\x7b\xa5\x34\x82\xe0\xeb\x53\x9c\x44\x4a\xc5\x5a\xa8\xdf\xcb\x39\
\x17\x24\xd5\x18\x9c\xc9\x71\x26\x67\xac\x32\xea\x4f\x98\x37\x4b\
\x96\x9f\x71\xfa\x35\xdf\x5c\xfb\x17\x4f\x1c\x2d\xce\x63\x02\x08\
\x14\x0d\xcd\xa8\x2e\xa1\xf5\xf8\x8d\x49\x47\x41\xb7\x29\x51\xca\
\xc5\x86\x76\xb8\xa9\x39\xbc\x57\x51\x72\x5d\x58\x35\x05\xde\x1f\
\x36\xbe\x22\x87\x27\xb3\xd0\xe4\x6c\xa0\x8c\xcd\x1b\xbc\x1f\x29\
\x97\x39\x65\xf1\x22\xb9\xe0\x82\xf3\xfe\xea\xba\x6b\xae\xbc\xf3\
\x98\x89\x7e\xc3\xbe\x50\x63\xa0\x39\x6c\x03\x1a\xf6\x38\x52\x68\
\xe0\xc0\x9e\x97\xbc\x33\x63\x00\xce\xd7\xbd\x90\x6f\xb2\x15\x21\
\x48\xa5\x34\x4a\x6b\x94\x4a\xd0\xba\x7e\xae\xa2\xb7\x97\x68\x15\
\xc2\x96\xa3\xb5\xa1\x33\xa7\xb5\x31\xd2\xb4\xca\xe9\x4b\x97\xbc\
\xbc\x74\xe9\x69\x67\x5d\x7b\xf5\x15\x6b\xdf\x6c\xe3\x40\x1d\x6b\
\xf7\x2d\x4c\x61\x2a\x3a\xd2\xd0\x0f\x24\x2e\x41\xb5\x52\x1e\x6c\
\x2d\xa9\x91\x52\x29\x71\x3a\x6e\x07\xba\x86\x6f\xa9\xcf\x25\x12\
\x37\x09\x54\x4c\x46\x94\x62\x24\x34\x31\x1c\xd6\xb9\x20\x99\x26\
\xc7\x98\x14\x6b\x33\xba\x4e\x9c\xc3\xd2\x53\x17\x7f\xfb\x93\x97\
\x7e\x62\xcd\x57\x6f\xfe\xf2\x7a\x11\xc9\xdf\x0c\x40\x72\x94\xcd\
\xac\x06\x80\xba\x95\xa6\x49\x52\x01\xb2\x2c\x4d\xb7\x77\x6f\xfd\
\xe5\xac\xd9\x27\xcc\xe9\x9c\x32\x7d\x5e\xa9\xb5\x78\x9c\xb1\x91\
\x46\x84\x11\x51\x79\x1a\xd7\x7b\x71\x78\x17\xe7\x25\x5c\x74\x98\
\x39\xd6\x18\xf2\x3c\x07\x6f\x98\xd4\x31\x71\xa4\xb7\xb7\xf7\x6f\
\xcf\x58\x7a\xca\xf6\x85\xf3\x67\xdf\x7f\xf1\xc5\x17\xd9\xb7\xb2\
\x75\x93\x8c\x9f\xc8\xfc\x11\x8d\x4c\x22\x15\xc2\xb2\x2b\x69\x0c\
\xe3\x3e\x4d\x6b\xe9\xae\x9d\xdb\xb6\x8c\x8e\x96\x7b\xba\xe6\x9f\
\x34\x77\xc6\x8c\x13\x56\x3a\xaf\x75\x79\x34\x05\x31\x5e\x04\x31\
\x26\xf7\xc5\x44\x8b\x87\xc6\x1c\x2c\x10\x8a\xd6\x1a\xb4\x86\xce\
\xc9\x9d\x87\x4e\x5a\x74\xe2\xf6\xdf\x39\x69\xfe\x3f\x5c\x7d\xc5\
\xe5\x77\xbf\xf8\xab\xc7\xdf\xd0\x54\xdf\x14\x40\xd7\xb2\xeb\x6d\
\x9d\xbf\x3e\xb6\xf2\xfa\x48\xd9\xe8\x9a\xf5\xc9\x2a\x3a\x8f\xf6\
\xce\xae\x4f\x06\x89\x4d\x48\x7d\x81\x97\x5e\x58\xff\x8d\x15\x05\
\xd9\x32\x61\xe2\xfb\x4a\x1d\xed\xad\x4c\x9b\x31\xfd\xb2\x7d\xfb\
\x07\x77\x9e\x30\x77\xe6\xe2\xfd\xbd\x03\x0e\x91\xe1\x39\x33\xa7\
\x76\xd6\x6a\x29\xc3\xe5\xf2\xfe\xd1\xd1\xd1\x17\xbd\xb5\xbf\x4e\
\x8a\x85\x57\x26\x77\x4e\x78\xa5\xb5\xb5\xb0\xe3\xea\x2b\x2e\x1f\
\x39\x9a\x23\xf8\x1f\xff\x66\xed\x5a\x76\xbd\xdf\xb1\xf1\xfb\x6f\
\xf8\x60\xd3\x96\x9e\x46\x17\x16\x22\xff\xe3\x80\x73\xd2\xc2\x79\
\x8d\xeb\x26\x4d\x3b\x85\x72\xdf\x96\xbf\x04\x06\x80\xfe\x69\xd3\
\x66\xd4\x06\x06\x07\x0e\x5c\x75\xcd\x67\x4f\xce\x0c\xee\xbe\xfb\
\x1e\xdc\xb8\xf6\xeb\x5f\xbb\xd8\x58\x6b\x76\xed\x7a\x7d\xdb\x9e\
\x3d\xfb\x7b\x5e\x7b\x75\xfb\xee\xad\x9b\x7e\x31\x3a\x6e\x0e\xf1\
\xff\x97\xdd\xbf\x06\x85\x5e\xd8\xd4\x1d\xbb\xa6\x3a\xc2\x46\x1c\
\xb1\xcf\x13\x5d\x66\xcf\xce\xdd\x8d\x49\xeb\xec\x55\x2b\x78\xf8\
\xa1\x2d\x3a\xee\x2b\xe9\x03\x07\xf6\xd7\x00\x77\xcf\xdd\xff\xb4\
\xa1\x5a\xad\x8c\x00\xb5\x07\xee\xbb\xeb\xf6\x0d\x1b\x9e\xaf\x02\
\x36\xee\x2a\xba\x63\x99\xc8\xff\x3d\x00\xef\xe9\x3a\xe3\x3a\x3e\
\xb0\x7c\x51\x70\x82\xce\xe2\x5d\x6c\xe5\x36\xc3\xd9\x14\x6b\x32\
\xbc\xcd\x70\x2e\x0f\xd7\xf8\xba\x39\x0b\x3e\x08\x28\x02\x85\x78\
\x24\x80\xae\x56\x2b\x0d\x51\xd8\xb0\xe1\xf9\x2c\x06\x39\x3e\x78\
\xff\x76\xf7\x5f\x05\x28\xc5\x00\x4a\x40\x1b\x30\x11\xe8\x04\xa6\
\x00\xc7\x01\xc7\x03\x53\xe3\xeb\x49\xf1\x1a\x1d\xb3\x59\x05\x86\
\x81\xc1\x48\xa1\x3e\xa0\x3f\xbe\x1e\x02\x46\x80\x0a\x50\x03\x72\
\xc0\x8c\x03\xf1\xb6\x1f\x49\x63\x47\x2e\x9c\xd7\x81\xb4\x00\xad\
\x31\xd8\xfa\xd1\x1a\x8f\x96\x26\x00\x1e\xc8\xc6\x7d\xde\xda\x94\
\x94\xa4\x69\xeb\xf2\x5d\xf9\x53\x3d\x69\xda\x17\x4d\x22\x05\x9a\
\x01\xb4\xc7\xa3\x19\x44\x33\x00\x62\x66\x8f\x06\xb4\x12\xef\xd7\
\x0c\xe0\x1d\x07\xf1\xdf\x91\x59\x83\x74\xeb\x5b\x8e\x51\x00\x00\
\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x00\xe1\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x12\x00\x00\x00\x12\x08\x06\x00\x00\x00\x56\xce\x8e\x57\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x00\x83\x49\x44\x41\x54\x78\xda\x62\
\x60\x18\x2a\x80\x1f\x88\xfb\x81\xf8\x3d\x10\xff\x47\xc2\xeb\x81\
\x58\x9e\x14\x83\xce\xa3\x19\x80\x8c\xdf\x13\x6b\x58\x3c\x9a\x46\
\x90\xcb\xf2\xd1\x5c\x37\x9f\x18\x83\xd6\x23\x69\xa8\x47\x12\xaf\
\x47\x12\xdf\x8f\xae\x89\x05\x8b\x41\x0f\x80\xf8\x00\x94\xbd\x01\
\x49\x5c\x80\x1a\x81\x8f\xee\xdd\x7a\x6a\x18\x72\x1e\x1a\xab\x24\
\x01\x79\x34\x43\xe6\x93\xeb\xa5\x7c\xb4\x34\x84\x13\x30\x11\x30\
\x08\x39\x80\x27\x0c\x8a\xac\x12\x8f\x94\x9a\xf5\x29\xf1\x5a\x02\
\x92\x17\x03\xe8\xe2\xa2\xc1\x07\x00\x02\x0c\x00\x33\xab\x33\x2a\
\x5d\xff\x92\x6a\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\
\x00\x00\x04\xc0\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x43\x42\x36\x32\x46\x45\x34\
\x42\x32\x46\x45\x43\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x43\x42\x36\x32\x46\x45\x34\x41\x32\x46\x45\
\x43\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x31\x32\x39\x37\x31\x33\x42\x41\x32\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\xfb\x35\x86\xcf\x00\x00\x00\xee\x49\x44\x41\
\x54\x78\xda\x62\x62\xa0\x0f\x50\x04\xe2\xd5\x40\xfc\x0a\x88\x9f\
\x43\xd9\x72\x74\xb2\x9b\xc1\x06\x88\x3f\x03\xf1\x7f\x34\xfc\x81\
\x1e\x8e\xc0\x65\x39\x0c\xaf\x1e\x48\xcb\xc1\xa1\xc0\x44\x43\x07\
\x04\x02\x31\x0f\xc3\x00\x83\xc9\x04\x42\x80\xea\x51\x60\x09\xc4\
\x67\x80\x58\x1a\x49\xac\x0b\x57\xf0\xa3\xa9\xa3\x8a\xe5\xb0\x38\
\xbf\x81\x66\x78\x0b\x16\xcb\xcd\x68\x65\x39\x0c\xdf\x43\xcb\x66\
\xd5\x50\xf1\xb7\x40\x6c\x4c\x6b\xcb\x61\xf8\x21\xb4\x10\x82\x81\
\x7c\x7a\x5a\x0e\xc3\x4f\x80\x58\x9d\x16\x29\x9c\x18\xcb\x61\xf8\
\x2a\x10\x33\x0f\x94\xe5\x6f\x07\x22\xd8\x47\x2d\x1f\xb5\x7c\xd4\
\xf2\x51\xcb\x47\x2d\x27\x19\x30\x43\xdb\xec\x03\x62\x39\xac\x05\
\x3b\x20\x96\xb3\x40\x69\x57\x22\xd4\xbe\x03\x62\x67\x20\xbe\x40\
\x8b\xba\xfd\x30\x11\x3e\x37\xa0\x55\xd3\x59\x08\x88\xff\x0c\x94\
\xe5\x20\xe0\x87\xc7\xf2\xe7\xb4\xb6\x1c\x94\x06\x3c\xd0\xc4\xf6\
\x41\xf1\x0e\x20\x3e\x4b\xeb\x9e\x0b\x23\x10\x1f\x00\xe2\x6b\x50\
\x0b\x41\x16\x7f\xa1\x67\xd7\x09\x20\xc0\x00\xf6\x44\x14\xcd\xbc\
\xc4\xf2\x59\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x04\x5c\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x32\x37\x32\x34\x35\x42\x42\
\x45\x32\x46\x45\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x32\x37\x32\x34\x35\x42\x42\x44\x32\x46\x45\
\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x31\x33\x39\x37\x31\x33\x42\x41\x32\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\xc6\xe2\x9b\x1d\x00\x00\x00\x8a\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\xa3\
\x0e\x18\x75\xc0\xa8\x03\x46\x1d\x30\xe0\x0e\x60\x18\x2c\x21\x20\
\x0d\xc4\x2b\x80\xf8\x09\x14\xaf\x80\x8a\xe1\x02\x54\x55\xaf\x0e\
\xc4\x6f\x81\xf8\x3f\x1a\x06\x89\x69\x61\x31\x8c\xea\xea\x57\x63\
\x91\x84\xe1\x75\x58\x0c\xa4\xba\xfa\x0f\x78\x14\x7c\xc6\x62\x20\
\x55\xd5\x0f\x8a\x6c\xb8\x0f\x8f\xfc\x6e\x2c\x62\x54\x57\xaf\x45\
\x62\xa2\xa2\x89\x7a\x69\x68\x62\x79\x0e\xc5\xab\x89\xc8\x56\xb4\
\x54\x3f\x82\x8a\x62\xc6\xd1\x56\xf1\xa8\x03\x46\x1d\x30\xea\x80\
\x11\xef\x00\x80\x00\x03\x00\x2e\xb6\x71\x11\xea\x94\x18\x46\x00\
\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x02\x78\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x18\x00\x00\x00\x18\x08\x06\x00\x00\x00\xe0\x77\x3d\xf8\
\x00\x00\x02\x3f\x49\x44\x41\x54\x78\xda\xad\x95\xdd\x4e\x13\x41\
\x14\xc7\xcf\xec\xde\x78\xe1\x85\x17\x35\xd4\x8f\x08\x85\xd1\x1a\
\x42\x53\xb4\x20\x88\xa9\x2d\x16\xa2\x8d\xbe\x88\x2f\x40\xe2\x13\
\x98\xf0\x02\xbe\x88\xa6\x1a\x28\x82\x0d\x44\xa2\xbd\x30\x1a\x92\
\xea\xc2\xa2\xf1\xa3\xc4\x5e\x78\xe1\x85\x37\x3b\xe3\xff\xcc\x96\
\x86\x2e\x1f\xbb\xc9\x70\xd2\x99\xdd\x3d\x7b\xe6\xfc\xf6\x7c\xcc\
\x54\x50\x02\x79\x58\xad\x16\x5c\xd7\x7d\x86\xdb\xf3\x5d\xd5\xef\
\x20\x08\x1e\xbf\xa8\xd5\x9a\x71\x6b\x45\x12\xc0\xa3\x6a\x75\x93\
\x84\x98\xdc\x37\xd6\x66\xd2\xef\x9e\xd7\x6a\x53\xd6\x80\x07\xf3\
\x73\xc2\x71\x5c\x0f\x86\x19\x40\x42\x7b\x0d\x21\xf2\x95\x0a\xe4\
\xcb\xa5\x65\x6d\x05\xb8\x5f\xa9\xc0\xad\xf0\x30\x32\x07\xec\x19\
\xe1\x63\xc8\x57\xf5\xba\x1d\x60\x7e\xb6\x6c\x00\x30\xed\x03\xe0\
\x67\x00\x4b\xaf\x57\xed\x00\xf7\xee\x16\xe1\xdf\x41\x04\x14\x89\
\x80\x00\x50\x72\xe5\x4d\xc3\x0e\x50\x9a\xb9\x2d\x1c\x97\x6b\x10\
\x49\x11\x47\xa0\x94\x5c\x5d\xdf\xb0\x03\x14\xa7\x6f\x71\x91\xb7\
\x11\xc6\x50\xa4\x06\xbb\x4a\xa9\x91\xc6\xdb\xcd\x78\xc0\xcc\x44\
\x41\x22\x05\x77\xa2\x5c\x6e\x1a\x11\x4e\x8b\xd0\xa4\xfa\x23\xa0\
\x0e\xe6\x85\xb0\xa1\xc2\x9c\xf5\xca\x13\x1a\xac\x6f\xbc\x6f\x7a\
\x62\xfa\xe6\x8d\xb4\xeb\x3a\x2d\xac\x3d\x7b\x52\x44\xdd\x0e\x3d\
\x18\xc1\x49\x1f\xce\x4d\xf0\x37\x08\x54\x16\x80\xf1\x32\x3e\xb1\
\x8e\xa5\x82\xdd\xc4\xa5\x2c\x99\x70\x89\xcc\x54\x11\x53\xe3\xf9\
\x59\x68\x96\x31\x9c\xd3\x71\xde\x13\x85\x31\x27\x26\xf3\xb9\x32\
\x68\x75\x50\x05\x25\x3c\x3a\x92\x84\x00\x57\x1a\xde\x2a\x62\x22\
\x37\x96\x46\x3f\xf3\xa1\x95\x3e\x12\xd0\x9f\x6a\x71\x48\x2b\x8e\
\x03\x50\x1b\xfb\xa7\x60\x5e\x17\xc6\x46\xd9\x79\x09\xe3\x8c\x29\
\x4f\xaf\x13\xf8\xc1\x44\xb6\x48\x91\x2e\x22\xee\x22\xa2\x05\x94\
\x4d\x87\xe5\x17\xdd\x32\x9a\xf7\xff\x30\xd6\x9a\x9f\xb6\xda\xb1\
\x29\xc9\x67\xaf\xb2\xcd\x36\x40\x43\x11\xc0\x2e\x9c\x8e\x7c\x68\
\x7d\xb1\xdb\x68\x39\x39\x8c\x64\x12\xce\x22\x8a\x9c\x45\xe4\xe3\
\x41\x7e\xf4\x76\xec\x00\xa3\x99\x41\xb6\x39\x02\xa0\x7d\x3c\xca\
\x2d\xff\xab\x1d\xe0\xfa\x95\x4b\x9c\xd8\x43\x00\xfe\x3f\xc0\x45\
\xb6\xbe\xfd\xb0\x03\x5c\xbb\x7c\xe1\x98\x08\x18\x40\xf2\xf3\xf7\
\x5f\x76\x00\x79\x31\x0d\x1b\xed\xc1\x65\x3f\x40\x90\x49\x91\xf7\
\xb3\x6d\x07\x60\x19\x1e\x48\xad\xe1\x52\x8c\xa8\x1b\x3b\x7b\x9d\
\x52\xdc\xda\xa4\x80\x2c\xf6\xc4\x53\xdc\x9e\xeb\xaa\xfe\xe0\xf0\
\x7b\x02\x40\xeb\x54\x00\xfb\x92\x19\x48\x19\x7b\x7f\xaf\xa3\x93\
\xae\xf9\x0f\x4d\xe4\xea\xf3\x31\xcc\xe8\x82\x00\x00\x00\x00\x49\
\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\x60\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x43\x36\x35\x33\x35\x45\x36\
\x32\x32\x46\x45\x38\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x43\x36\x35\x33\x35\x45\x36\x31\x32\x46\x45\
\x38\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x41\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\xbf\x31\x69\xb8\x00\x00\x01\x8e\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\xc6\x51\x07\x8c\x78\
\x07\x0c\x38\x60\x24\x41\xad\x34\x10\x47\x01\xb1\x31\x10\xab\x00\
\xb1\x16\x10\x73\x42\xe5\xbe\x03\xf1\x35\x20\xbe\x03\xc4\x67\x81\
\x78\x19\x10\x3f\xa5\x86\x03\xd9\x80\x38\x07\x88\x4f\x02\xf1\x7f\
\x12\xf1\x49\xa8\x5e\x36\x72\x2d\x0f\x04\xe2\xdb\x64\x58\x8c\x8e\
\x6f\x43\xcd\x22\x1a\x80\x82\x75\x35\x15\x2c\x46\xc7\xab\x91\xa2\
\x0c\x27\x90\x20\x22\xb8\x7f\x02\xf1\x64\x20\xb6\x87\x06\x2f\x1b\
\x94\x3d\x19\x2a\x47\x28\x5a\x24\x70\x59\x2e\x4a\x44\x90\x3f\x01\
\x62\x03\x3c\x1e\x30\x80\xaa\x21\x14\x25\xa2\xd8\x12\x1b\x31\x3e\
\xd7\x85\xaa\x57\x84\x06\xe9\x13\x28\x5e\x01\xcd\x25\x0c\x50\x35\
\xc4\x84\x04\x4a\xe2\xec\x25\x22\x0e\x27\x23\xa9\xaf\xc6\x22\xff\
\x16\x9a\x35\x19\xa0\x6a\x09\x99\xd7\x0b\x33\xcc\x12\x88\xff\x10\
\xa1\xc1\x06\x2d\xd4\x1a\xb1\xa8\x59\x07\x95\xb3\x21\xc2\x3c\x90\
\x9d\x66\x20\xc5\x07\x88\x4c\xc5\xd8\xf2\x33\xba\x23\x3e\x23\x45\
\x29\x31\x66\xee\x65\x01\x12\x7f\x49\x2c\x1f\x3e\x00\x31\x3f\x11\
\x05\x18\xd1\x80\xd8\x28\xb0\x84\xaa\x5f\x87\x47\xcd\x3a\x24\x33\
\x89\x89\x02\x4b\x52\x12\xe1\x04\xa8\x5a\x2d\x68\x82\xc3\x97\x08\
\x27\x90\x92\x08\x61\xa5\x1f\x29\xd9\x50\x1a\x9a\x0d\x9f\x43\xf1\
\x6a\x4a\xb3\x21\x29\x05\x91\x2e\x9e\xe8\xd4\x25\xb7\x20\x82\x01\
\x39\x20\xbe\x44\x44\x48\x4c\x80\x16\xbf\x0c\x48\xd9\x6e\x02\x11\
\x3e\xbf\x04\xb5\x03\x2f\xe0\x01\xe2\x2d\x34\xa8\x8c\xb6\x40\xcd\
\x26\xa9\x3a\xbe\x47\x05\x8b\xef\x91\x5a\x1d\xa3\x27\xce\x22\x0a\
\x1a\x24\x45\x84\xaa\x60\x52\x9a\x64\xa0\xb8\x0b\x87\x16\x9f\x2a\
\x58\x6a\xc5\x0b\xd0\x26\xd9\x29\x20\x5e\x09\xc4\x8f\x18\x46\xc1\
\x68\xc7\x64\xd4\x01\x44\x00\x80\x00\x03\x00\x43\x6b\x83\x67\xc6\
\x30\xe4\xce\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\x31\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x41\x37\x36\x33\x32\x37\
\x37\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x44\x41\x37\x36\x33\x32\x37\x36\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x44\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x46\x57\xe8\x62\x00\x00\x01\x5f\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\x03\x0c\x30\x32\x32\x32\xe0\x00\
\x3c\x40\x6c\x0f\xc4\x4e\x40\xac\x0e\xc4\x12\x40\xac\x05\xc4\x9c\
\x50\xf9\xef\x40\x7c\x0d\x88\x5f\x00\xf1\x4d\x20\xde\x07\xc4\x07\
\x81\xf8\x0b\x36\xc3\x90\xed\x04\x73\x60\x18\x0b\x30\x03\xe2\xa5\
\x40\xfc\x13\xa4\x94\x44\xfc\x13\xaa\xd7\x0c\x9b\x03\xe0\x76\xe2\
\x70\x00\xc8\xc7\xf3\xc9\xb0\x14\x17\x5e\x04\x35\x93\x28\x07\x80\
\x14\x9e\xa1\xa2\xe5\x30\x7c\x06\xe6\x08\x42\x0e\x98\x4c\x03\xcb\
\x61\x78\x32\x31\x0e\x78\x45\x43\x07\xbc\x42\x77\x00\x23\x96\x5c\
\xf0\x9f\x81\xb6\x00\xc5\x4e\x26\x86\x01\x06\xd8\x1c\xf0\x8e\x86\
\xf6\x7d\x24\xc6\x01\xcb\x68\xe8\x80\xc5\x0c\x78\x0b\x05\x08\xe0\
\xa7\x61\x36\xe4\x27\x26\x17\xa8\xd3\xb0\x20\x52\x27\xc6\x01\x5b\
\xd0\x8a\xe2\x15\x14\x14\xc5\x2b\xd0\x8a\xe2\x2d\xc4\x38\x00\xc4\
\x58\x8d\x5c\x74\x42\x83\xce\x0f\x88\x7b\x81\x78\x3b\x10\x9f\xc7\
\x62\xe1\x79\xa8\x5c\x2f\x54\x2d\x3f\x5a\xd1\xbe\x1a\x96\xc5\x09\
\x39\xe0\x2d\x54\x21\x88\xae\x80\xd6\x7c\xe4\x02\x09\xa8\x19\x30\
\x33\x3f\x10\x53\x10\x81\x8a\xcb\x1c\x34\x83\x8e\x43\xf1\x4d\x68\
\xb5\xfb\x1d\x9a\xa5\xee\x40\xe5\x55\xa0\x3e\xe6\x84\x56\xd3\x20\
\xbe\x0d\x10\x5b\xa2\x99\x33\x05\x88\x73\x09\x55\xc7\xa3\xb9\x60\
\x34\x17\x8c\xe6\x82\xd1\x5c\x30\x9a\x0b\x46\x73\xc1\x80\xe7\x82\
\x01\xef\x9a\x31\xd0\x30\x17\x30\x10\x93\x06\xb0\x01\x7e\x68\xf7\
\xdc\x1e\x1a\xc7\xa0\xc4\x65\x80\xa6\xe6\x02\xb4\x7b\x7e\x0d\xda\
\x35\x3f\x88\xad\x19\x8e\xde\x3d\x07\x08\x30\x00\xf7\x1c\x74\xa4\
\x68\x77\xa0\x2a\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\
\x00\x00\x04\x24\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x32\x32\x46\x44\x31\x39\x38\
\x42\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x32\x32\x46\x44\x31\x39\x38\x41\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x41\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\xc6\x00\xa5\x28\x00\x00\x00\x52\x49\x44\x41\
\x54\x78\xda\xec\x97\xc1\x09\x00\x30\x08\xc4\xbc\xe2\xfe\x2b\xdb\
\x0d\x44\x68\x41\x5a\xe3\x57\xc4\xa0\xe6\xa1\x22\xc2\x3a\x63\x59\
\x73\x00\x00\x40\x3b\x80\x67\x49\x49\x57\x9a\x64\xaa\x7b\xa5\xfe\
\xb0\xbf\xb8\x81\x77\x8f\xb0\xb2\xc3\x11\x13\xc0\x02\x2c\xc0\x02\
\x2c\xc0\x82\xbf\x2d\x10\x9f\x11\x00\xe3\x01\xb6\x00\x03\x00\x36\
\xb1\x0c\x4a\xab\x95\xf0\x77\x00\x00\x00\x00\x49\x45\x4e\x44\xae\
\x42\x60\x82\
\x00\x00\x03\x5a\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x18\x00\x00\x00\x18\x08\x06\x00\x00\x00\xe0\x77\x3d\xf8\
\x00\x00\x03\x21\x49\x44\x41\x54\x78\xda\x95\x54\x4b\x4f\x13\x51\
\x14\x9e\x3b\xa5\x09\x0b\xb6\xc8\x94\x57\xc0\xf2\x28\x8a\xef\x07\
\x8a\x28\x64\x5a\xa0\x3c\xc2\xc2\x12\x17\x76\xe1\xd2\x8d\x09\x09\
\x24\x26\xba\x73\x6b\xa2\xa1\xac\x5c\xb1\xf3\x11\x83\x0b\x51\x1a\
\xf0\x51\x31\x28\x86\x28\x3e\x08\x4a\x2b\xaf\x84\x04\x3b\xca\x0f\
\xd0\xa4\xce\x5c\xbf\xb9\x33\xd3\x4e\xdb\xe9\x30\xde\x64\xee\xbd\
\x73\xcf\xf9\xbe\xef\xcc\xb9\x73\x0e\xe1\x72\x46\x7f\x6f\x2f\x51\
\xd7\xa7\xd1\x28\xe5\xfe\x73\xa8\xd8\x5c\x1c\x31\xbf\x0c\xf4\xf7\
\x5d\x26\x84\xbf\x89\x6d\x31\x55\x94\x5b\x93\x53\x53\xd7\x1d\x12\
\x0b\x2e\x97\xeb\x11\xb6\xa7\xf0\xac\x29\xb2\x3c\xf0\x24\x1a\x4d\
\x64\x09\xc0\x69\x88\xe7\xf9\xdb\xe6\x33\x45\x51\x46\x11\xd1\xb0\
\x1d\x79\x5f\x30\x28\xf0\x2e\x57\x8c\x10\xe2\x33\xce\x28\xa5\x49\
\x88\x88\x53\xd3\xd3\x09\x46\xd6\x1b\xec\xbe\x84\xc8\xc7\x89\x4a\
\x0e\x4f\xc3\x8b\xb2\x45\x19\x8d\x4e\xcf\x58\x8a\xf4\x74\x76\x0a\
\x7c\x91\x2b\x06\x98\x8f\x81\x54\xac\x8e\x03\x32\x89\x2c\x9c\x61\
\xe7\x3d\x5d\x5d\x5f\xb0\x1c\x20\x06\x79\x26\x12\xca\xe9\x22\xb2\
\x2c\x8f\x3c\x8f\xbd\x4a\xe7\xb7\xcb\x2f\x22\x2d\x45\x20\xe7\x7c\
\x44\x67\xb7\x40\x5e\x63\x87\xc1\x40\x60\x01\xcb\x09\xab\x7b\xa1\
\x70\xc6\xa4\x42\xee\x20\x65\x43\x2f\x66\x67\x53\x81\x8e\x0e\x0f\
\x72\xfe\x52\x23\x27\x79\x18\x06\xd3\x9e\x61\xa2\x47\xd3\x0d\x86\
\x49\x50\xb9\xad\x44\x54\x7a\x7d\x9e\x80\xd0\x08\x48\x67\xc0\xeb\
\xd3\xdd\xac\xc8\xd5\xef\x59\xc1\xdc\x96\x36\x06\x3a\xda\x43\x00\
\xdf\xc3\xd6\x5a\x84\x6a\x40\x4c\x7f\x60\x28\xd6\x3d\x2c\xc9\x31\
\xe2\xb8\x53\x11\x5f\x2b\x65\x39\x88\x67\xdb\xec\x45\x32\x04\x36\
\x36\x02\x72\x22\xc6\xe6\xe6\x24\x2b\x27\xae\xbd\xf5\x74\x08\xae\
\x76\x22\x56\x23\x1d\x39\xbc\xc5\xd7\xf3\xef\x24\xce\x0e\xdc\xd6\
\x72\x32\x84\xc5\xa9\x08\xd5\xf9\xe3\x6a\x12\xde\x2c\xbc\x97\xcc\
\xc6\x82\xc0\xd6\x63\x47\x43\xb0\xde\x87\x4b\x91\x03\x81\x38\x66\
\xff\xfc\xe2\xc7\x64\xae\xb1\x20\xb0\xe5\xf0\x21\x0f\xac\x6f\x71\
\x8f\x35\xbb\x09\xe0\xfe\x1f\x63\xbe\xb0\xf0\x79\x29\xe5\x48\xe0\
\x78\xf3\x7e\x81\xf0\xc4\x54\xa1\x36\x02\x94\x33\x2a\x77\x82\x2a\
\x34\xfc\x61\xf9\x6b\xca\x56\xe0\x48\x53\x23\xc8\x79\x73\x6f\x71\
\x7c\xc9\xf8\x03\x21\xa2\x84\x3f\xad\x24\x52\x96\x02\x07\x1b\xea\
\x04\x34\xbc\x18\xaa\x44\x8b\x9c\x23\x4e\xc8\xd3\x99\xd2\x55\x26\
\x50\xf1\xe1\xa5\xef\x6b\xa9\x2c\x81\x66\x6f\x2d\xeb\x8a\x9c\x5a\
\xfe\x1a\xb7\x5d\x0d\x58\x67\x20\xd3\xe8\x98\xc8\xf2\xda\x66\x8a\
\x39\xec\xab\xad\x46\xe4\x1a\x39\x47\x6c\xca\x5f\xfb\xcf\x23\x78\
\x1b\xa3\xdc\xae\x6d\xe5\x01\x55\xe4\x30\x33\x36\xd5\x54\x3f\x04\
\xef\xa0\x5d\x6f\xc1\x69\x9c\x27\xbc\x1f\x3b\x09\xf7\x7a\x9e\x55\
\x3c\x2d\x5c\xf1\x5a\x93\xa4\x17\x99\xa1\xb1\xba\x72\x1d\x9b\x5a\
\xbb\xde\x82\xbb\x11\xbf\x6d\x6e\xa5\x8b\xa8\xa9\xa6\xca\x49\xef\
\xba\xc1\x0e\x1b\x2a\xcb\xc7\x70\x72\x25\xc7\x31\xd3\x5b\x78\x22\
\x26\xb6\xb6\xa5\x1c\x71\xae\xb1\xaa\x42\x17\xa1\xee\x7c\x2c\xa1\
\x78\x3b\xc7\x0e\xea\x2b\x3c\x6e\xfc\x5e\x77\xb1\x1d\xa4\xd9\x9e\
\x20\xe7\xc5\xd5\xed\x64\x1e\xb9\x31\xea\xcb\x05\x26\x62\xba\x13\
\x15\x4b\xf1\x8f\x0c\xaf\xfe\x90\x22\xe9\xcf\xaa\xf3\x94\xb9\x65\
\x45\x56\x45\x42\xa6\xb4\xf8\xd7\x93\xbf\x0a\x92\x1b\xc3\x2b\x94\
\x86\x14\x4a\xc7\xb1\x2d\xc1\xf3\x97\x27\xe4\xea\xba\xb4\x13\xc9\
\xcb\x9b\x57\xd8\x43\x10\x8d\x88\xf4\x95\x20\x82\x67\x1b\xd2\xce\
\xef\xdd\xc8\x8d\xb1\xb7\xac\xb4\x0a\x38\xa4\x84\x2c\x6e\xfc\xdc\
\x89\x1b\xe7\xff\x00\x5c\xa8\x69\x24\xb6\x70\xae\x2d\x00\x00\x00\
\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\x62\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x37\x45\x38\x44\x45\x43\x32\
\x43\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x37\x45\x38\x44\x45\x43\x32\x42\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x42\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x5c\x2f\x62\xaf\x00\x00\x01\x90\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\x03\
\xee\x00\x86\x81\x8e\x02\x16\x22\xd5\x49\x03\x71\x20\x10\x7b\x03\
\xb1\x28\x10\x1b\x40\xc5\x2f\x00\xf1\x6b\x20\xde\x0a\xc4\xeb\x81\
\xf8\x29\xb5\x43\x40\x02\x88\x67\x02\xf1\x1f\x90\x52\x02\xf8\x0f\
\x54\xad\x04\xb5\x1c\xe0\x01\xc4\x9f\x89\xb0\x18\x1d\x7f\x86\xea\
\xa5\xc8\x01\xf9\x64\x58\x8c\x8e\xf3\xc9\x75\x80\x1f\x8e\x20\xbf\
\x01\xc4\x45\x40\x6c\x0c\xc4\xcc\x50\x6c\x0c\x15\xbb\x81\x23\x4a\
\xfc\x48\x75\x00\x28\xb1\x7d\xc0\x62\x58\x2f\x10\x73\xe2\x31\x86\
\x13\xaa\x06\x5d\xdf\x07\xa8\x99\x44\x3b\x60\x26\x9a\x01\x3f\x49\
\x8a\x4f\x88\xda\x9f\x68\x66\xcc\x24\xd6\x01\x72\x58\x82\x3e\x9f\
\x8c\xac\x5d\x84\x25\x2a\xe4\x88\x71\x40\x05\x9a\xc6\x4b\xd0\x78\
\x26\x15\x80\xf4\x5c\x45\x33\xab\x82\x98\xa2\xd8\x15\x4d\x6e\x2e\
\x10\xff\x25\xc3\x01\x7f\xa1\x7a\x91\x81\x13\x31\x21\xf0\x04\xcd\
\xd5\xea\x14\x94\xb0\x5a\x68\x66\x3d\x21\xc6\x01\xe8\x89\x87\x8d\
\x02\x07\xf0\x63\x49\xcc\x34\xaf\x0d\x3d\x90\xca\x83\x0f\x68\x72\
\x6c\x68\xe5\x89\x07\x31\x51\xa0\x42\xa2\x03\x9e\x93\x50\x4a\x3e\
\xc7\x16\x02\x37\xb1\xf8\x88\xd4\xc4\x47\xba\x5a\x2a\x66\x43\x3f\
\x12\x6a\x4d\x6f\x6c\x0e\x50\xc4\x62\x40\x0e\x89\xa1\x10\x42\xc0\
\x11\x7f\xa0\x6a\x68\x56\x14\xe3\x73\xc4\x4f\x0c\xcb\x71\x54\x46\
\x9f\xc9\xa8\x8c\x08\x39\x02\xbb\xe5\x78\xaa\xe3\xff\x78\xaa\x63\
\x03\x24\xb5\x06\xd0\xfa\x42\x0e\x47\x9a\x78\x0e\xc5\x7e\xb4\x6e\
\x90\x3c\x84\xa6\x21\xaa\xb6\x8a\xfd\x48\x6c\x92\x3d\x21\xab\xf8\
\x26\xd0\x28\x05\xa5\x89\x39\x44\x66\x2f\x90\x63\x1b\x69\xd5\x2f\
\x90\x86\x66\xc9\xed\x40\x7c\x06\xad\x44\xdb\x02\x95\xe3\x1f\x92\
\x1d\x93\x01\xef\x9a\x31\x8e\xf8\x10\x00\x08\x30\x00\xa5\x45\x3d\
\xd3\xc3\x27\x28\xe6\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\
\x82\
\x00\x00\x04\xb4\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x37\x35\x45\x46\x46\x44\x37\
\x31\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x37\x35\x45\x46\x46\x44\x37\x30\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x42\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x81\xbe\x72\xe3\x00\x00\x00\xe2\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\x03\
\xee\x80\x21\x01\xec\x81\xf8\x24\x10\xff\xa7\x10\xff\x04\xe2\x03\
\x40\x9c\x03\xc4\x6c\xa4\x38\xe0\x39\x15\x2c\x47\xc7\xe7\x81\x58\
\x9a\x58\x07\x3c\xa1\x81\x03\x40\xf8\x12\x28\x24\x98\x89\x70\xc0\
\x35\x20\x76\x01\x62\x1e\x2a\x47\xad\x38\x10\xbf\xa2\x67\x5a\x32\
\x83\xa6\x01\xe4\x50\x38\x4c\xef\x04\x2d\x81\x9e\x30\x19\x07\x20\
\x57\xfd\x1f\x2d\x09\x47\x1d\x30\xe8\x1d\xe0\x01\x2d\xfb\x1f\x42\
\xeb\x01\xba\x82\x10\x20\xfe\x83\x94\x4f\x9f\xd3\x28\x1b\x22\x63\
\x9c\x96\x63\x28\xa0\x02\x90\x46\x33\xfb\x33\x21\xcb\xa9\xe9\x00\
\x50\x29\xb8\x1d\xcd\xec\x63\x20\x09\x6f\x3c\x96\xd3\x1a\xe7\xd3\
\xaa\xbe\x27\xba\x3a\x06\xe5\x82\xbf\x03\x90\xfb\x2e\x03\xb1\x27\
\x10\xff\x02\x71\xfc\xe8\x18\x05\x87\xa1\xc1\xce\x46\x28\xfb\xd1\
\x2a\x17\x90\x54\x06\xd0\xdd\x01\xf4\x2a\x88\x08\x02\x3f\xa8\xc5\
\x4f\x06\xa2\x28\x1e\x79\x80\x71\xc4\xf7\x8e\x01\x02\x0c\x00\x49\
\x62\xdd\xf7\x46\x26\x8a\x67\x00\x00\x00\x00\x49\x45\x4e\x44\xae\
\x42\x60\x82\
\x00\x00\x04\xf2\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x37\x46\x32\x39\x46\x45\x31\
\x30\x32\x46\x44\x30\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x35\x30\x35\x32\x34\x37\x37\x38\x32\x46\x44\
\x30\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x30\x36\x38\x30\x31\x31\x37\x34\x30\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x3e\x5e\xaf\xbf\x00\x00\x01\x20\x49\x44\x41\
\x54\x78\xda\x62\x60\x20\x0e\x24\x00\xf1\x61\x20\x7e\x05\xc4\xff\
\xc9\xc4\x3f\x81\xf8\x00\x10\xe7\x00\x31\x1b\x91\xf6\x32\xf0\x03\
\xf1\x31\x0a\x2c\xc5\x85\xcf\x03\xb1\x34\x21\xcb\x99\x69\x64\x39\
\x0c\x5f\x22\x14\x12\x69\x34\xb4\x1c\x86\x73\x18\x68\x08\x40\xbe\
\xb3\x07\xe2\xc9\xd0\xf8\xc7\xe6\x80\xc3\x0c\x74\x02\x06\x40\xfc\
\x04\x47\xc2\x44\x71\x71\x0e\x34\xa5\xfe\xa4\x41\x6a\xd7\xc5\x61\
\x2e\x18\x48\x43\x53\x26\xad\x53\xfb\x64\x6c\x0e\x60\x83\xa6\x48\
\x7a\xa4\x76\x1b\x6c\x0e\xc8\xa1\x63\x6a\x67\xc3\xe6\x80\xc3\x68\
\x02\xa0\xf8\x33\xa3\x20\xc1\x99\x41\xcd\xc0\x95\xda\x31\x1c\x80\
\x9e\x30\x24\xa8\x90\xea\x25\xf0\xa4\x76\x14\x07\x30\x22\xa7\x44\
\x28\x60\xa4\x52\xd6\xc3\x65\x2e\x8a\x38\x13\xc3\x00\x83\x51\x07\
\x8c\x3a\x60\xd4\x01\x03\xed\x80\xef\x03\xed\x80\xcb\x03\x5d\x12\
\x26\x32\xe0\x6a\x20\x50\xc9\x01\xd8\xcc\x45\x6e\x2b\x30\x0f\x94\
\x03\xce\x20\x57\x7a\xf4\x76\x40\x2c\xba\x42\xf4\xea\x58\x9a\x0a\
\x96\x4b\xa3\x99\xf9\x19\x9f\x62\xf4\xce\xc7\x76\x0a\xdb\x04\x12\
\x50\x33\x90\xcd\x3c\x86\x4f\x43\x3e\x1d\x9a\x64\xf9\x84\x3a\x10\
\xf4\x6a\x94\xe2\x8d\xb3\x4b\x34\xb2\x9c\xe8\x34\xc5\x06\x0d\xaa\
\x03\x54\xb0\xf8\x30\xd4\x2c\x82\x3e\x07\x08\x30\x00\x3f\xf5\x78\
\x87\x69\x55\x35\xcf\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\
\x82\
\x00\x00\x01\x0e\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x18\x00\x00\x00\x18\x08\x06\x00\x00\x00\xe0\x77\x3d\xf8\
\x00\x00\x00\xd5\x49\x44\x41\x54\x48\x89\xed\x96\x41\x0a\x83\x30\
\x10\x45\x9f\xb6\x88\x4b\xaf\xd0\x8b\x94\x9e\xa5\x07\x70\x55\xba\
\x4c\xc1\x7b\x89\xd0\x6b\x74\x25\x78\x87\xae\x62\xbb\x30\x91\x51\
\x6c\x02\x4d\xb2\x28\xf4\xc3\x40\x32\x0e\xef\x43\x26\x31\x81\x49\
\x15\xa0\x80\x07\xa0\x81\xd7\x97\xa1\x0d\x43\x19\xe6\x0c\x6f\x03\
\xa0\x9f\xa2\x05\xaa\x1d\x70\x05\xce\xc4\xd7\x01\x78\x02\xf4\xc2\
\xf5\x02\x94\x01\xd0\xd2\x30\x2c\xaf\x87\xe5\x9a\x17\x01\x70\xab\
\x42\xf0\x74\x66\x06\x56\x59\x04\x03\x24\x33\xf7\x14\xd9\x80\x69\
\x67\xb8\x9a\xba\x29\x97\x41\x14\x25\x37\x48\xde\x83\xbd\xa3\x48\
\x89\x71\x03\x9c\x80\xa3\xa3\xbe\x71\xb9\x6d\x35\x2a\xa4\xc9\x73\
\x3e\x79\x0f\x5c\x4b\x74\x5b\xcd\xef\x1b\x39\xaf\xfe\x07\xcd\xab\
\xdf\x3f\x68\x39\x30\x8a\x0f\xb1\x7e\xd7\x56\x63\x0e\x0c\x22\x51\
\x13\x7e\xe1\xd4\x62\x3e\x80\x7f\x77\x84\x84\x82\xe9\xd2\xef\x12\
\xc0\x3b\x56\x2f\x8b\x24\xcf\x96\x37\x51\x4b\xac\x0a\x83\x5f\x6d\
\x62\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\x86\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x30\x00\x00\x00\x30\x08\x06\x00\x00\x00\x57\x02\xf9\x87\
\x00\x00\x05\x4d\x49\x44\x41\x54\x78\xda\xd5\x59\x09\x52\x1b\x47\
\x14\xed\xbe\x81\xa8\x54\x88\x21\x8e\x2d\x85\xd5\x6c\x96\xd8\x8c\
\x58\xc2\x88\x60\x0c\x04\x1c\x74\x82\x68\x6e\x80\x4e\x60\xe9\x04\
\xe8\x06\xe8\x06\xc8\x80\xd9\x4c\x90\x30\xfb\xaa\x61\x35\x9b\xcd\
\xd8\xb1\x21\x90\xa4\xac\x1b\x74\x7e\x4b\x62\x6c\x60\x46\xdd\x9a\
\x19\x84\xfd\xab\x54\x23\xe8\xdf\xef\xf7\xeb\xf9\x5b\xb7\x30\xfa\
\xc6\x05\xdf\xf6\x02\xbe\x4a\x02\x3d\x4f\x9f\x0a\xc9\xaf\x76\xf8\
\x48\xf4\x4b\x68\x70\x30\xf2\x4d\x10\xf8\xbd\xbb\xbb\x0f\x1e\xbd\
\x2a\x43\x91\xe7\x43\x43\xae\xaf\x9e\xc0\xd3\xae\xae\x4f\xf0\xb0\
\x68\x0c\xdb\x06\x87\x87\xe5\x5b\x23\xd0\xdd\xd9\x49\x17\x46\xdd\
\x42\x1e\x1a\x19\x91\x55\xc6\xed\x08\xe3\xa8\x26\x00\x21\x5e\x98\
\x17\x48\x81\x2b\xc1\x78\xec\x46\x08\x74\x75\x74\x08\xf0\xe8\x87\
\x19\x56\x44\x10\x35\xe2\x1d\x1e\x1d\x0d\x5e\xd1\xe9\x83\xf1\x5e\
\x4d\x10\x82\x42\x30\xc7\xad\x82\x3b\x00\xf3\x2c\x30\x4e\xe3\x45\
\x04\x1d\xc9\x54\x02\xbf\xb5\xb7\xfb\xe0\xf1\x4c\x65\x08\x08\x10\
\xef\x8b\xb1\xf1\x58\x52\x2f\x9a\xdc\x49\x4d\x79\x31\x36\x86\xd9\
\xb8\x71\xcc\x00\xe2\x10\x26\x81\xce\x27\x6d\x10\x94\x58\x7b\x57\
\x11\x91\x60\xe7\xdc\x09\x24\x7c\xcc\x36\x49\xdc\xa0\x1f\x01\xfd\
\x7e\xd0\xef\x49\xa1\x17\x18\x19\x9f\xf0\x1a\x26\xd0\xd1\xf6\x38\
\x0c\x0f\x21\xf5\x9a\xe2\x2e\x25\x01\x9a\xc0\xc2\x4b\xba\x89\x25\
\xee\x8a\x0c\x19\x9d\x78\xc9\x5c\x1f\x53\xa1\xbd\xb5\x55\x2b\x2d\
\xde\xb4\x84\xc6\x26\x27\xdd\x2c\x25\x26\x81\x27\xbf\xb6\x50\x9f\
\x8e\xb2\xf4\x6e\x40\xc4\xf1\x3f\xa7\x82\x86\x09\x50\x69\x6b\x71\
\x31\x83\xd3\x64\xa1\x2e\x69\x9b\x98\x0a\x33\x53\x2a\x17\x81\xc7\
\x2e\x81\xba\x50\x5f\x06\x09\x04\x5f\x86\x23\x22\x8f\x22\x17\x81\
\x56\xa1\x19\x72\x3f\xe1\xc8\x30\xa6\x89\x7b\x72\xfa\x55\x28\x6d\
\x02\x2d\x4d\x8d\x1e\xa8\xa4\xd6\xf8\x1f\x84\x24\x32\x4b\x42\x2c\
\x18\xe3\x81\x4c\xad\x9e\x10\x42\x83\xf7\xc2\x7d\xac\x5f\xac\x49\
\x9e\x9a\x99\x0d\xaa\x12\x70\x35\x36\xf4\x82\x62\x26\xdd\x44\x2f\
\x3b\x31\x3c\x3b\xa7\x90\x50\x08\x08\x0d\xf5\x66\x05\x6a\x08\xde\
\xd6\x73\x78\xca\x60\x28\x02\x1b\x23\xc0\x77\x2b\xec\xea\x1f\x88\
\x55\x4f\x38\xf1\x23\x73\xf3\x4a\x7a\x8d\x13\x68\xae\x77\x5a\xe1\
\x61\xd4\xc7\x25\x00\x13\x23\xf3\x0b\x9a\x7d\x8c\x50\xef\x14\x48\
\x22\x19\x18\xdb\x28\x42\xb2\xa6\x17\x16\x63\x0a\x81\x5f\x9c\x8f\
\x20\xcb\x18\x72\x9f\x20\x00\x79\x2f\x40\x53\x49\xb3\xb3\xce\x02\
\x24\xc2\xc6\x48\x10\xf7\xab\x85\xa5\x90\x42\xa0\xa9\xae\x96\x06\
\x68\x8f\x4e\x34\xd8\x71\xec\x9a\x59\x5c\xe2\x6e\x83\x9b\xea\x1e\
\x41\xfb\x4c\xa8\xcb\x5a\x75\xda\xf4\xcf\x2c\x2e\xfb\x14\x02\x8d\
\xb5\x35\x74\xf1\xfa\xb2\x0c\x46\xae\xd9\xa5\x95\x48\xba\xd3\x0c\
\xd9\x84\x34\x3b\xbb\xbc\xf2\xf9\x0d\x50\x69\xa8\xa9\xa6\xaf\x14\
\x3a\xc4\xb4\x5e\x6d\x64\x6e\x65\x55\xf7\x31\x11\x6c\xd2\xb8\xb3\
\xa6\x31\x85\xbe\x65\x37\xd8\x54\x36\xec\x5a\x21\xab\xaf\xae\xf2\
\xc1\xeb\x7d\xc6\x09\xe8\x9d\x5f\x5d\xe7\xea\xdb\xd5\xa4\xbe\xba\
\x32\x9d\x46\x11\x76\x1c\x8b\xf3\xab\x6b\x97\x5c\x55\xb5\x12\x3b\
\xab\x1c\x76\x68\x7b\x07\x38\x76\xc7\xb5\xb0\x1e\x8d\x20\x9d\xe2\
\xac\x74\x78\x50\xe2\xad\xa7\x92\x18\xac\x52\x5c\x58\x8b\xaa\x56\
\x66\xcd\x56\xa2\xce\xf1\x10\xc0\x31\x0b\xdc\xb5\x18\x95\x74\x13\
\xa8\x73\xd8\x05\x94\xc8\x48\x29\x84\xf8\x17\xa3\x1b\x3e\xad\x51\
\x55\x02\xb5\xf6\x0a\x1f\x56\x3f\x42\x5e\x86\x06\x7f\x5c\x96\x36\
\xb9\x7a\x16\x0d\x3b\xbd\x98\xa3\x49\x24\xd4\x7d\x08\x12\x97\x37\
\x36\xaf\x65\xba\x4b\x04\x6a\x2a\xca\xec\x74\xd7\x31\xe6\x0b\x64\
\x42\x50\x60\x65\x73\x8b\x79\xec\xd3\x92\x9a\x8a\x72\x6a\xcb\xc3\
\x69\x2b\x46\xf3\xff\xca\xe6\x76\x44\x95\x40\x75\x79\x29\xec\x3a\
\xe6\x0d\xde\xe4\xce\x10\x79\x75\x6b\xc7\xa6\x67\xf1\x60\x8f\x5e\
\xa5\x1c\x83\x4d\x4b\x3a\xf3\xc0\x26\x4d\x1a\x7e\xb0\xfb\xb9\x12\
\x57\x95\x95\xf8\x10\x87\xcb\x68\x88\x77\x6d\x7b\x37\xed\x4c\x64\
\x96\xcd\x38\x81\xca\xd2\x07\x34\x58\x3d\xba\xa0\x12\x6d\xb7\x6b\
\x7d\x77\x8f\xfb\x2e\xa7\xb2\xa4\x38\xf5\x05\x18\x5b\xfc\xeb\x3b\
\xaf\x7d\x0a\x01\xc7\x83\x22\xba\x78\x56\xc6\x49\x25\x71\x12\xd1\
\xd7\xfb\x4c\x12\x60\x4b\x40\x89\x0a\x9c\x96\xeb\x5c\x85\xb9\xb0\
\x95\x20\x50\x5c\x48\x1b\xac\x4f\x06\x00\x95\x9d\xa1\xf7\x39\xd2\
\xde\xe1\xb5\x6c\x61\x2f\x2e\xb0\x24\xef\x97\xf4\xba\xcd\x85\xc8\
\xd2\xde\x81\x12\x77\x4a\x10\x3f\x2c\x2a\x30\xd2\xd0\x5d\x11\x42\
\x53\xeb\xc6\x17\xff\x68\x06\x53\x82\x39\xd8\x28\xb0\xb1\x7f\xa8\
\x64\x3e\x85\x40\x45\x61\x3e\x18\x20\x61\x5d\x90\x99\x93\x18\xed\
\x7c\x37\x0f\x8e\x14\x57\xe5\xbe\xdc\x2d\xcf\xff\x99\x64\x6c\x99\
\x18\xd9\xb6\x0e\xdf\xca\x7c\xaa\x1c\x52\x96\x67\xa5\x05\x2e\x83\
\x97\x5b\x44\xdc\x7e\x23\x07\x4d\x23\x50\x6a\xbb\x9f\xfa\xda\xdc\
\xf4\xf5\x23\x69\xe7\xf8\x9d\xc3\x34\x02\x25\xd6\x7b\xa9\x7e\x75\
\xb9\x29\xb1\xed\xca\xef\x65\xc3\x04\x8a\xef\xdf\xed\x81\x72\x9f\
\xb1\x3b\xa1\x0b\x81\x96\xc1\xbb\xf7\xee\x03\xb3\xc2\x33\x09\x14\
\xdd\xfb\xb1\x1f\xeb\xad\xd2\x86\x08\x20\x79\xff\xfd\x47\x66\x9f\
\xc5\x26\xf0\x53\x2e\xfb\xf7\x01\x38\x5a\xc2\x67\x1a\xf1\x15\x29\
\x5a\x23\x68\xa1\x63\x6d\x4a\x6c\xff\xaf\x93\x2c\xc3\x04\x0a\xef\
\xe6\x78\xa0\x95\xd5\x6c\x33\xa0\x1d\xf6\x1f\x7c\x38\xf5\x81\x9e\
\x05\xf4\x78\xaa\xb9\xe3\xf0\xe3\xa9\x94\xc4\xa5\x67\x01\xb5\xd8\
\x8a\x01\xae\x17\x70\x83\x86\x09\x50\x29\xc8\xbd\x63\x07\x9f\xbc\
\x7a\xc4\x8c\x41\x6c\x88\x87\x27\x7f\x2b\x07\x9a\xfc\xdc\x1f\x58\
\xd5\x5c\x3e\x3a\x39\xb3\x31\x70\x25\xc0\x75\x03\xae\xcc\xb3\x36\
\xee\x42\x96\x9f\x93\x4d\xfb\x25\xea\x22\xbd\x09\x23\xc8\x7d\x74\
\x7a\x7e\xc9\x48\x5e\x4e\x36\xeb\x1a\x3e\xf8\xe6\xf4\x5c\x54\xc1\
\xa5\x73\x3c\xf0\x09\xc0\x78\x5a\x07\x24\x53\x7f\xe8\xce\xbb\xf3\
\xbd\x15\xdc\x22\xd5\x15\xa5\xfb\xed\xd9\x3f\xba\x8f\xa0\x37\x4e\
\x80\x8a\x2d\xfb\x3b\xed\x4b\x62\x8c\xb3\x8e\xcf\xfe\x4d\xeb\x87\
\xec\xdb\x20\xe0\x41\x48\x35\x38\xfd\xc7\xe7\xff\xf9\xcc\xb6\xf7\
\x3f\x4f\xb5\xe2\x40\x57\x64\x6f\x0b\x00\x00\x00\x00\x49\x45\x4e\
\x44\xae\x42\x60\x82\
\x00\x00\x05\x18\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x34\x38\x32\x35\x45\x31\x33\
\x30\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x33\x43\x38\x37\x33\x32\x39\x38\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x41\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x14\x10\xa2\xc6\x00\x00\x01\x46\x49\x44\x41\
\x54\x78\xda\xec\x57\x51\x0a\x83\x30\x0c\x2d\xee\x12\xbb\x8d\x5f\
\x82\x20\xf8\x25\x14\x84\x5d\x77\xe0\x0d\xf6\x25\x08\x4a\xcf\xb0\
\xe1\xd8\xea\x68\xc1\x49\xda\xa4\xb1\x65\xfb\x30\x10\x41\xda\xe4\
\x3d\x1b\xfb\x9a\x0a\x21\x44\xae\xbd\xd3\xfe\x72\xf8\x4d\x7b\x29\
\xc2\xad\x34\xb1\xae\xbc\x9d\xc1\x16\xca\x33\xc9\xba\x62\x10\x20\
\xe5\xcd\xf4\xe3\x4c\x48\xf6\x64\x10\xa0\xc4\x7c\xb0\x31\x96\xb3\
\xf6\x8a\x41\xa0\x36\xb1\x58\x7e\x14\xbc\x11\x7c\x93\x18\x89\x2c\
\xc1\xd2\x07\x1b\xb6\x44\x77\xe6\x2a\x48\x6e\x09\x66\x20\x30\x94\
\x04\x04\x3e\x3b\x08\x7d\x6d\x17\x5b\xf3\x66\x07\x89\xc6\x01\x0e\
\xe5\xfd\x6c\xef\x45\x0c\x06\xf3\x52\x23\x89\x30\x12\x3e\xf0\xf5\
\xee\x50\x06\x33\xe7\x7e\x0d\xb4\x2d\x2b\x02\x38\xcb\x20\x12\x90\
\x32\x0e\x29\xc0\x5d\x24\x46\x60\xce\x18\x61\xe7\xa0\xca\xa6\x8c\
\x17\xc0\x78\xb1\x1a\xaf\xc5\x61\x87\x25\xb0\xa5\xc3\xb9\x1a\x87\
\x04\x24\x5f\x8d\x97\xb1\xc1\x41\x09\xf5\xe8\x40\x52\x0d\xa0\xe8\
\x40\x34\x12\x32\x82\x14\x4b\x0c\xc4\x1e\x46\xdd\xa6\x76\x2d\x43\
\xdb\x5d\x67\x47\xbb\xf9\x97\xba\xf5\x61\xa4\x80\xc9\x1c\x70\x6c\
\xd5\xa0\xbc\x23\xb5\x21\x09\xad\x67\x50\x43\x42\xe9\x8a\x39\x3f\
\x53\xfb\x0f\x5d\xf1\xe5\xd7\x5d\xf1\x23\x56\x57\xcc\xb9\x98\x54\
\x31\x4a\xe0\x13\x1e\xcc\x46\x4a\xee\xa5\x04\x13\x21\xd9\x89\x41\
\x80\x12\x33\x59\x61\xf0\xdd\x64\xfb\x1d\x25\xe8\x91\x1b\x77\xf9\
\x16\x60\x00\x37\x5f\x6f\x36\xe4\x2e\xb8\xc6\x00\x00\x00\x00\x49\
\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x04\x9e\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x46\x42\x38\x32\x41\x44\x39\
\x33\x32\x46\x44\x30\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x46\x42\x38\x32\x41\x44\x39\x32\x32\x46\x44\
\x30\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x30\x37\x38\x30\x31\x31\x37\x34\x30\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x07\x15\x9b\x40\x00\x00\x00\xcc\x49\x44\x41\
\x54\x78\xda\x62\x60\x40\x00\x36\x20\xce\x01\xe2\x03\x40\xfc\x13\
\x88\xff\x53\x19\xff\x84\x9a\x9d\x03\xb5\x0b\x05\x48\x03\xf1\x79\
\x1a\x58\x8a\x0b\x9f\x87\xda\x09\xf7\xf9\x25\x3a\x5a\x0e\xc3\x97\
\x60\x21\x91\x33\x00\x96\xc3\x30\xc8\x6e\x86\xc3\x68\x82\xa0\x78\
\x32\x63\xa0\x3e\x30\x83\x9a\x8d\x6c\x17\xc8\x6e\x8c\x04\x27\xc1\
\x40\x3b\x20\x81\x9e\x30\x19\xa1\x0c\x64\xc0\xc8\x40\x5b\xf0\x1f\
\xdd\xb2\x41\xef\x80\xff\x38\x0c\xa2\x8a\xba\x21\xe5\x00\x46\x1c\
\x7c\x72\xd5\x0d\x3d\x07\x8c\xdc\x34\x30\x5a\x0e\x8c\x3a\x60\x34\
\x17\x8c\x96\x84\xa3\x69\x60\x78\x97\x03\xd8\x1a\xa5\xd2\x34\xb4\
\x5c\x1a\xcd\xae\xcf\x20\xc1\x63\x68\x82\xdb\x69\xd4\x32\x96\x80\
\x9a\x8d\x6c\x17\xc8\x6e\x86\xfc\x01\xec\x98\xe4\x0f\x8a\xae\x19\
\x2c\x6e\x2e\xd1\xd9\x72\x8c\xb4\xc6\x06\x0d\x92\x03\x34\xb4\xf8\
\x30\xd4\x0e\xb8\xcf\x01\x02\x0c\x00\xbd\xa2\x4b\x8d\x76\xdf\x38\
\xd4\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x04\x46\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x22\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x30\x20\x36\x31\
\x2e\x31\x33\x34\x37\x37\x37\x2c\x20\x32\x30\x31\x30\x2f\x30\x32\
\x2f\x31\x32\x2d\x31\x37\x3a\x33\x32\x3a\x30\x30\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\
\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x4d\x4d\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x6d\x6d\x2f\x22\x20\x78\x6d\
\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\
\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\x79\x70\x65\x2f\x52\x65\x73\
\x6f\x75\x72\x63\x65\x52\x65\x66\x23\x22\x20\x78\x6d\x70\x3a\x43\
\x72\x65\x61\x74\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\
\x65\x20\x50\x68\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x20\
\x4d\x61\x63\x69\x6e\x74\x6f\x73\x68\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x44\x39\x45\x32\x39\x32\x44\x42\x33\x41\x32\
\x39\x31\x31\x45\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\
\x45\x30\x43\x44\x32\x22\x20\x78\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\
\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\
\x3a\x44\x39\x45\x32\x39\x32\x44\x43\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\x3a\x44\x65\x72\x69\x76\
\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\x65\x66\x3a\x69\x6e\x73\
\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\x2e\x69\x69\x64\
\x3a\x44\x39\x45\x32\x39\x32\x44\x39\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x20\x73\x74\x52\x65\x66\x3a\x64\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x39\x45\
\x32\x39\x32\x44\x41\x33\x41\x32\x39\x31\x31\x45\x32\x39\x42\x38\
\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\x32\x22\x2f\x3e\
\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\
\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x52\x44\x46\x3e\x20\x3c\
\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\x3e\x20\x3c\x3f\x78\x70\
\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\x22\x72\x22\x3f\x3e\xc0\
\x5e\xdf\x6b\x00\x00\x00\xba\x49\x44\x41\x54\x78\xda\x62\xfc\xff\
\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\xa3\x0e\x18\x75\xc0\xa8\x03\
\x46\x01\xb9\x20\x01\x88\x0f\x03\xf1\x2b\x28\x3e\x0c\x15\xa3\x39\
\xe0\x07\xe2\x63\x40\xfc\x1f\x07\x3e\x06\x55\x43\x13\xc0\x4c\xc0\
\x72\x64\x47\x30\xd3\xc2\x01\x69\x44\x58\x0e\xc3\x69\xa4\x18\x6c\
\x0f\xc4\x27\x49\x30\x1c\x86\x0f\x00\xb1\x19\x14\x1f\x20\x43\xff\
\x49\xa8\xdd\x0c\xcf\xc9\xd0\x0c\xc2\x12\x48\x9e\x90\x20\xd3\x8c\
\xe7\xa0\x82\xe8\x2f\x05\x69\x02\x1b\x9b\x14\x00\xb6\xdb\x83\xcc\
\x50\xd8\x0e\xf5\xb9\x04\x94\x4d\xb2\xef\xa1\x76\x13\x0d\xa2\x48\
\x30\x3c\x76\x58\x66\x43\x58\x41\x74\x92\x40\xca\x16\xa2\x75\x69\
\xc8\x0c\x2d\x76\x41\x3e\x7d\x0b\xc5\xc7\xa0\x62\xcc\x0c\xa3\x80\
\x44\xc0\x38\xda\x2f\x18\x75\xc0\xa8\x03\x46\xbc\x03\x00\x02\x0c\
\x00\x95\xbc\x9b\xe1\x17\xa5\xac\x3b\x00\x00\x00\x00\x49\x45\x4e\
\x44\xae\x42\x60\x82\
\x00\x00\x04\x74\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x32\x37\x32\x34\x35\x42\x43\
\x32\x32\x46\x45\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x32\x37\x32\x34\x35\x42\x43\x31\x32\x46\x45\
\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x31\x33\x39\x37\x31\x33\x42\x41\x32\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x37\x5f\x21\xce\x00\x00\x00\xa2\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\xa3\
\x0e\x18\x75\xc0\xa8\x03\x46\x1d\x30\xe0\x0e\x60\x21\x45\x31\x23\
\x23\x23\xd1\x6a\x89\x2d\xe2\xc9\x0d\x01\x69\x20\x5e\x01\xc4\x4f\
\xa0\x78\x05\x54\x8c\x74\x00\x72\x29\xb1\x18\x0a\xd4\x81\xf8\x2d\
\x48\x2b\x1a\x06\x89\x69\x91\x6a\x2e\x39\x0e\x58\x8d\xc5\x72\x18\
\x5e\x47\xaa\x03\x18\x49\xa9\x8e\xa1\x69\xe0\x03\x10\xf3\xe3\x50\
\xf2\x05\x88\x79\xe9\x91\x06\x06\x34\x1b\xee\xc3\x23\xb7\x9b\x1e\
\x89\x50\x6b\xa0\x13\x21\x2c\x1b\x82\x12\xe3\x73\x28\x5e\x8d\x9e\
\x0d\x69\x99\x08\x07\x45\x41\x34\x30\x45\x31\x2d\x5a\xd0\xa3\xb5\
\xe1\xa8\x03\x46\x1d\x30\xea\x80\x01\x77\x00\x40\x80\x01\x00\xed\
\x71\x2e\x06\x0c\xcc\x59\x19\x00\x00\x00\x00\x49\x45\x4e\x44\xae\
\x42\x60\x82\
\x00\x00\x05\x06\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x37\x45\x38\x44\x45\x43\x32\
\x34\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x37\x45\x38\x44\x45\x43\x32\x33\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x42\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x56\x16\xe3\x26\x00\x00\x01\x34\x49\x44\x41\
\x54\x78\xda\x62\xfc\xff\xff\x3f\xc3\x40\x02\x26\x86\x01\x06\x03\
\xee\x80\x41\x0f\x12\x80\xf8\x00\x10\x3f\x07\xe2\xff\x24\xe2\x9f\
\x50\xbd\x39\x40\xcc\x46\xaa\xc5\x12\x40\x7c\x86\x0c\x4b\x71\xe1\
\xf3\x40\x2c\x4d\xac\xe5\xcc\x40\x7c\x92\x8a\x96\xc3\xf0\x25\x62\
\x43\x22\x9c\x06\x96\xc3\x70\x0e\x31\x0e\x50\x04\x62\x51\x20\x96\
\x83\x3a\xa6\x05\x88\xcd\xc8\x48\x3f\x66\xd0\x34\x80\xec\x80\xc3\
\xf4\x4e\xc4\x12\x58\x12\x26\x0a\x60\xa4\x83\x23\xfe\xe3\xb3\x73\
\x50\x96\x84\x7e\xd0\x7c\xff\x1c\xca\x26\x15\x78\x00\xf1\x0d\xa4\
\x60\xc7\x16\x22\xff\xa1\x6a\x3c\xd0\x25\x43\x80\xf8\x0f\x5a\x9c\
\x85\x90\xe8\x00\x52\x0a\xad\xe7\xf8\x2c\x27\xd7\x11\x4f\x48\x70\
\xc0\x13\x42\x96\xc3\xf0\x1f\x12\x1c\xe1\x47\xc0\x2c\x64\x33\xbd\
\x19\xa0\x04\x49\x1a\x88\x00\x24\x79\xe8\x39\x39\x41\x46\x81\x23\
\x30\xa2\xf4\x09\xb9\x89\x86\x0c\x47\x60\x4d\x4f\xa0\x60\xbd\x47\
\x84\xe5\xf7\x48\x88\x02\xa2\xb3\x35\x23\xa9\x25\xd7\x88\x68\x13\
\xfe\x42\xe3\x4b\xd3\xdb\x51\xc7\xd0\xe2\x7e\x3b\xb4\x56\x23\xa7\
\x26\xac\x06\x62\x27\x68\x43\x44\x08\x5a\xd5\x13\x04\xf9\x34\x6c\
\x90\x10\x55\x98\xb1\x41\x9b\x4f\xd4\xb6\xfc\x24\xb4\xb9\x47\x14\
\x90\xa6\xb2\x23\xce\x93\x93\x96\xd8\xa0\xd1\x71\x80\x4c\x4b\xdf\
\x42\x9b\x60\xc9\xa3\xbd\x1b\x7c\x80\x71\xc4\xf7\x8e\x01\x02\x0c\
\x00\x7e\x24\x3a\x5f\x35\xfd\xc2\x9e\x00\x00\x00\x00\x49\x45\x4e\
\x44\xae\x42\x60\x82\
\x00\x00\x05\xad\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x43\x36\x35\x33\x35\x45\x36\
\x36\x32\x46\x45\x38\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x43\x36\x35\x33\x35\x45\x36\x35\x32\x46\x45\
\x38\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x41\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x55\x0c\x2e\x11\x00\x00\x01\xdb\x49\x44\x41\
\x54\x78\xda\xc4\x56\xcf\x67\x03\x51\x10\xde\xb6\x84\x3d\xf5\x5a\
\x4a\x09\xa1\xa7\x52\x96\x25\xa7\x9e\x4a\xe8\xa9\x84\xb2\xe7\xfe\
\x05\x4b\xfe\x80\x92\x6b\xfe\x8f\x90\x4a\x4f\xd5\x43\x95\xb2\x24\
\x72\x6a\xa5\x4a\x68\xe5\xd4\x4a\x84\x2d\x3d\x45\x48\xb4\xb3\x35\
\xcb\x9a\xcc\xee\x7b\xaf\x3b\xdd\x7e\x7c\x87\xbc\xbc\x99\xf9\xf2\
\xe6\x47\x66\xc3\xd2\xc3\x01\xf0\x14\x78\x0c\xb4\x81\x87\x78\xfe\
\x08\x9c\x03\x6f\x81\x57\xc0\x27\x4b\x18\x51\xd0\x11\xf0\x4b\x93\
\xd1\xdd\xba\x44\xe0\x3d\x60\xcf\x20\x30\x65\x0f\x7d\xfc\x0a\x47\
\xc0\x30\x47\xf0\x98\x21\xfa\x32\x0e\xbe\x60\x9c\x2d\x81\x5d\xa0\
\x07\x74\x80\x5b\x48\x07\xcf\xba\x78\x87\xda\x2d\x4c\x44\x94\x81\
\xb3\x94\xbc\xba\x1a\xf6\x6e\x4a\xbd\xcc\xd0\xb7\x12\x01\x63\xdc\
\xc2\xaa\xd7\x85\x8d\x36\xd4\x4f\xa0\x32\xac\x33\x46\x8d\x1c\x45\
\xdc\x60\xfc\x65\x76\x07\x7d\xba\xb6\x40\x27\xb5\x99\x54\xa6\x16\
\x1e\x2d\x9c\x5d\x01\x01\x15\xa6\xa0\xd7\x0a\x72\x13\x58\x23\x67\
\x97\xc0\x77\x01\x01\xaf\xe8\x2b\x89\x1a\x27\xa0\x4a\xce\xae\x05\
\x27\x29\xf5\x55\xe5\x2e\xd1\xd6\xdb\x67\x52\x34\xd0\x18\x3c\x03\
\xe6\x89\xcb\x4c\x4b\xae\x81\xe6\x69\x9b\x7c\x3f\x31\x98\x7e\x13\
\x62\x5b\x62\xea\x6b\x2d\x05\x2a\xac\x0c\x9e\x7c\xc5\x08\xc8\xf4\
\x15\x09\xf8\x24\x67\x3b\xe4\xf3\x39\x70\xaa\x11\x7c\x8a\x77\x93\
\xa0\xdd\xf4\xc1\x19\xde\x93\x67\xf2\x04\x8b\xd0\x23\xbe\xef\xb8\
\x17\xe8\x93\xb3\x13\x41\x01\xd4\x57\x5f\x77\x10\x55\x04\x82\x97\
\x99\x02\x77\xff\x73\x14\x0f\x4d\xff\x8c\xfc\x1c\xc1\x7d\xc6\x9f\
\xb2\xb6\x02\x66\x09\x69\x32\xad\x94\x85\x12\xda\x2c\x99\xed\x48\
\xf9\xb7\x5e\x11\x58\x48\x9e\x33\x86\xd4\x8d\x8e\x88\xac\x95\xac\
\x03\x3c\xc3\x35\x2c\x86\x83\xe9\xeb\xa4\xac\x64\x9c\x88\x92\x8e\
\x08\x89\xa5\x34\x97\x88\xbc\x6b\x79\x28\x21\x22\xee\x8e\x17\x83\
\xc0\x43\x4c\x93\x8d\x41\x44\x44\xc4\xb9\xbe\xc0\x57\x79\x48\x38\
\x19\xe3\x78\x6d\x32\x85\xaa\x23\xa2\x65\xfd\x31\x54\x22\xde\xac\
\x02\x90\x25\x62\x64\x15\x84\x34\x11\xbe\x55\x20\xe2\x29\x39\xc6\
\x5f\xfe\x13\xfc\x5b\x80\x01\x00\x6c\xa2\x60\xae\xdf\xa1\x79\x7e\
\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\x94\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x31\x39\x34\x36\x42\x43\x35\
\x31\x32\x46\x45\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x31\x39\x34\x36\x42\x43\x35\x30\x32\x46\x45\
\x44\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x31\x33\x39\x37\x31\x33\x42\x41\x32\
\x37\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\xb1\x74\x57\xb4\x00\x00\x01\xc2\x49\x44\x41\
\x54\x78\xda\xb4\x97\xcd\x66\x03\x51\x14\xc7\x47\x87\x10\xca\x10\
\x42\x19\xba\xea\x03\x0c\x43\x29\x25\xbb\x3e\x41\xb7\xa5\xf4\x0d\
\xba\x2a\xa5\x64\x15\x4a\x5e\xa2\x84\x10\xb2\x0a\xb3\xea\x6a\x08\
\x53\x43\x29\x59\x95\x52\xb2\xea\xaa\x15\x52\xa9\xd0\x9e\xcb\x19\
\xae\xd3\xfb\x31\x99\x73\xef\xe1\x2f\xc4\xcd\xfd\xfd\xef\xc7\x39\
\xe7\x26\x08\xdc\xc7\x05\x68\x06\x5a\x82\x7e\x41\xef\xa0\x21\xa8\
\x1b\x78\x8e\x7d\x50\x86\x50\x95\x3e\x40\x3d\x9f\xf0\xdc\x00\xaf\
\xb4\xf6\x61\xa2\x2e\xdc\x8b\x89\xb6\x06\xfe\x8c\x90\x13\xbc\x0f\
\x2a\x13\xa7\x2e\x0c\x3c\x28\x26\x2f\x40\x91\x34\xa6\x05\x9a\x68\
\xee\xc4\x01\x07\x9e\xd6\x80\xdb\x4c\x0c\x39\x06\x06\x35\xe1\xb2\
\x09\x9a\x25\x6f\x1c\x03\xd9\x0e\xf0\x2a\x12\xba\x0b\x7b\x0c\x03\
\xdf\xf8\xf9\x04\x3a\x03\x7d\xe1\x99\x26\x96\x8c\x91\xe3\x87\xb3\
\x03\x77\x64\xe5\xe7\xa0\x15\xae\x6c\xa6\x18\x2f\xc6\x95\x64\x07\
\x72\x8e\x81\x98\xc0\xb7\x64\xf2\x84\xc0\x0b\xc5\x25\xbc\x74\x91\
\x8a\x2a\xf8\x0a\x0d\x9a\xe0\x62\x97\x42\x1f\xf0\x2d\x7e\x6f\x82\
\xe7\x8a\xfb\xf0\x2f\x8e\x41\x37\xa0\x31\xaa\x2f\xad\xca\x2b\x5c\
\x94\xd7\x91\xa5\xc8\x34\x85\xcf\x6d\xf0\x0e\x0e\xda\x15\xbe\xa9\
\x01\x1f\xe1\xe2\xb4\x71\x08\x5a\x30\xe1\x55\x76\xf4\xf1\xd8\x44\
\xb1\xba\xc7\xa6\x64\x8c\x23\x7c\xb9\x70\xe1\x8d\xfb\xf9\xab\x05\
\xde\x95\x8a\x8c\x53\xb8\x88\xdb\x1a\x5d\x2d\xb1\x9c\x79\xcc\x31\
\xf0\x42\x26\x2f\x15\x8d\x25\x94\x1e\x16\x6b\x02\x9f\x63\x59\x6e\
\x1c\xf4\x5c\x7b\x96\x37\x40\x4c\xe0\xe2\x37\x53\x8e\x81\x8d\xa1\
\x86\xeb\x22\x22\xe9\x9a\xb9\x3c\x02\x5b\x8d\x8e\x14\xb5\x62\xc0\
\x31\x70\xad\xb8\x84\x13\x8d\x89\x48\x53\xa8\x52\x8e\x81\x16\xbe\
\x62\x6d\x26\x74\xf0\xb1\x8b\x54\x8c\xf1\x7d\xa6\x6a\x99\x29\x5e\
\xcc\x52\x93\xae\x6d\x57\x6f\x7c\x9d\x09\x9d\x16\x3e\xfe\xef\xc5\
\x9a\xaa\x48\xb5\xe4\x16\x1f\x9b\x89\x47\x03\xbc\xc0\xde\xe1\x3d\
\xae\xd0\xc8\x27\x56\xbf\x1c\x33\x26\x74\x09\xf9\x13\x60\x00\xf1\
\x3d\x42\xb8\xfb\x69\x12\xcb\x00\x00\x00\x00\x49\x45\x4e\x44\xae\
\x42\x60\x82\
\x00\x00\x03\x31\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x18\x00\x00\x00\x18\x08\x06\x00\x00\x00\xe0\x77\x3d\xf8\
\x00\x00\x02\xf8\x49\x44\x41\x54\x78\xda\xad\x95\x5b\x4f\x13\x41\
\x14\xc7\x67\xb6\xb3\x5c\x44\x92\x9a\xf8\x60\x82\x84\x68\xe4\x22\
\x48\x40\x44\x50\xa1\x22\x04\x8d\x89\x26\xbe\x9b\x28\x1f\xc2\x67\
\xba\x2c\xf8\xea\x57\x30\x51\x1f\x7c\xf6\x0d\x95\x9b\x16\x08\x58\
\x50\x91\x8b\x22\x46\x43\xbc\xc4\x07\x12\x49\x10\xa1\xec\x76\xc6\
\xff\x4c\x77\xcb\xee\x76\x1f\xd0\x32\x49\xdb\x9d\x33\x73\xce\xef\
\x9c\x39\xff\xce\x52\x92\xc7\xb8\x7a\xb9\x9b\xea\x05\x85\xb3\x94\
\xd2\x46\x4c\x29\x3e\x02\x63\xd0\xda\x49\x5d\x1b\x7c\x3e\x24\x88\
\x63\xfc\xef\x71\xa5\xab\x53\x02\x6c\x02\x02\x02\x51\x44\x14\x12\
\x61\x5b\x16\x7b\x3a\x3c\x9c\x3f\xa0\xfb\x52\x07\x65\x7a\x81\x8d\
\xe8\x9a\x6b\x13\x82\x70\x9e\xb6\xd9\xb3\x91\xd1\xfc\x01\x5d\x17\
\x63\x34\xc2\x18\x00\xd4\x03\x10\x5c\x70\xce\x86\xc6\x5e\xfc\x1b\
\xa0\xe3\xc2\xf9\x46\x6c\x8e\x12\xaa\x5c\xd6\xc6\x26\x26\x17\x3a\
\xdb\xdb\xa8\x16\x89\xd8\x98\x6b\x9e\xad\x1c\x10\x36\xf2\x32\xb1\
\x77\x40\xac\xb5\xe5\x26\x32\x7d\xe4\xb5\x21\xcb\x5b\x88\xf0\x58\
\xd3\xb4\x9c\x0a\xf0\xc3\x90\xc0\xde\x01\x6d\x67\x9b\xe3\xc8\xb4\
\x2f\xd3\xcb\x8c\x54\x30\xfa\xf0\x70\x97\x6a\x9a\x15\x06\x48\x4c\
\x4d\x87\x03\x9a\xeb\xeb\x94\x0d\x8e\x24\x39\x37\x2f\x5a\x4f\x37\
\xe0\x51\x8b\xe3\x2c\x0c\x84\xcf\xec\x57\x5a\x11\x26\x1e\x06\xb0\
\xd1\xf2\x36\x19\x6b\x1c\xcb\x6c\x32\x39\x9b\x0b\x40\xb0\xfb\xc8\
\xa6\xc7\x93\xcd\x20\x4f\xf3\xeb\x20\xc4\x35\x8d\x1a\xc4\x69\x80\
\xd2\x8a\x20\x00\x10\x00\x88\x25\xd3\xf1\x1e\x1e\xd6\xd8\xd4\x9b\
\xb7\x7e\x40\x53\x6d\x0d\x14\xa1\x43\xd3\xc4\xa3\x69\x79\xd4\x69\
\x1d\xd0\x5e\x9c\x42\xb6\x02\x47\xef\x19\x00\xa1\x56\x50\xa6\xf8\
\x66\xc9\x77\x0b\x7e\x40\x63\x4d\x95\x52\x44\x88\xe4\x1c\x00\x2a\
\x70\x7a\xa0\xc2\x13\x17\x40\x42\x7b\xf0\x7a\xf1\xbd\x1f\xd0\x50\
\x5d\x29\x7b\x18\x26\x39\x1d\xbf\x19\xc0\xee\x7e\x59\x84\x09\xca\
\x00\xec\x56\x98\x4c\xe7\x96\x57\xfc\x80\xfa\x13\xc7\xe5\x73\x98\
\xe4\x14\x00\xd9\x1b\x94\x84\x1d\x51\x78\x05\xf3\x9f\x3e\xfb\x01\
\xb5\xc7\x2a\x1c\x40\xf0\x3c\x1d\x00\x21\xbb\x3d\x90\xdd\x21\xc4\
\x03\xc8\xf1\x61\x4b\x5f\x56\xfd\x80\x9a\x8a\xa3\x0a\x10\x2c\x17\
\xa1\xb2\x00\xe2\x91\x69\x16\xa0\x54\x14\xf0\x01\xe0\xc3\xea\x37\
\x3f\xa0\xba\xbc\x8c\xa2\x3a\x1b\x19\x50\xd7\x8e\xcc\xb0\x99\x2a\
\x00\x62\x1a\x44\x38\xfb\x29\x11\x60\x39\x15\x08\x0b\x3e\x2e\x40\
\x50\xb9\x46\x29\x5b\xfe\xfa\x23\x58\x41\x39\xb5\x52\xa9\x6d\xe7\
\x48\xb2\x15\x33\x5d\x67\x9c\xf3\x5e\x9e\x4e\xfb\x9a\x0c\xc5\x99\
\xf8\x03\x0e\xe0\x6a\xb6\x03\xff\x27\x4b\x2f\x2c\x28\xca\xa9\x40\
\x8e\xca\xb2\x23\x71\x38\xf4\x78\x4c\x4f\x4a\xa3\x87\xee\xa4\x6d\
\xbb\xea\xcf\xef\x8d\x87\x98\x1f\x76\xec\x6b\x07\x0e\x96\xde\xc6\
\xfd\xf4\x71\x63\xfd\xd7\x3d\xcc\x6f\xb8\x0e\x48\xe8\xc1\xca\xf7\
\x9f\xfd\xee\x3c\xe7\xaa\x38\x73\xaa\x56\x5d\x39\x50\x03\x99\x5d\
\x58\x12\x61\x6b\x33\xf3\x8b\x3e\x7b\x53\xdd\x49\x79\x9d\xc8\x06\
\xc3\xc7\xbf\x96\xd7\xfb\x60\x2f\x43\x01\x62\xe7\x5a\xa2\x78\xf5\
\x45\x73\x80\x42\x04\x76\xbb\x57\x51\xf6\x6b\x37\x4c\xc0\x73\x27\
\x95\x5a\x1f\x9f\x7e\xb5\xae\xcc\x9d\xb1\xf6\x78\x61\x51\xb1\xb1\
\x9f\x99\xa7\xb6\xb7\xcc\xd1\xc4\x78\xbf\x02\xe0\xbe\x37\x70\xe6\
\x5e\x95\xe4\x3b\xa4\x54\xcd\x89\xe4\x8c\xa9\x02\xe2\x1d\x60\x6c\
\x6d\x6e\xee\x2b\xa0\xb8\xa4\xc4\x84\x18\xcc\xbf\x4b\xc5\x7c\x28\
\x52\x36\xbf\xa4\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\
\x00\x00\x00\xe3\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x12\x00\x00\x00\x12\x08\x06\x00\x00\x00\x56\xce\x8e\x57\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x00\x85\x49\x44\x41\x54\x78\xda\x62\
\x60\x18\x6c\x80\x99\x48\x75\xf6\x50\xfa\x23\x25\x96\xd5\x03\xf1\
\x7f\x20\x7e\x4f\xa9\xab\xf7\x43\x0d\xfa\x4f\xa9\x41\xf6\x50\xc3\
\xe2\xa9\x11\x96\xf2\x40\xcc\x4f\xa9\x01\x30\xaf\xdd\x27\xd7\x10\
\x7d\x68\x00\xc3\xc2\x67\x3f\x3e\xc5\x4c\x78\xe4\x36\x00\xf1\x03\
\x20\x3e\x80\xc4\x27\x0b\xf0\x43\xbd\x06\x73\x91\x3e\xb9\x2e\x02\
\x25\x3e\x03\x28\xfb\x01\x14\xf3\x93\x63\x10\x08\x38\x40\xe9\x0b\
\x50\x2f\x6e\xa0\x46\x62\x7c\x8f\xcf\x7b\x84\x5c\xf4\x01\xc9\x6b\
\x20\xd7\x5d\xa4\x24\xc0\xed\x19\x86\x24\x00\x08\x30\x00\x6f\x0c\
\x20\x5d\x80\x42\x66\x5a\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\
\x60\x82\
\x00\x00\x04\xa0\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x22\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x30\x20\x36\x31\
\x2e\x31\x33\x34\x37\x37\x37\x2c\x20\x32\x30\x31\x30\x2f\x30\x32\
\x2f\x31\x32\x2d\x31\x37\x3a\x33\x32\x3a\x30\x30\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\
\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x4d\x4d\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x6d\x6d\x2f\x22\x20\x78\x6d\
\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\
\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\x79\x70\x65\x2f\x52\x65\x73\
\x6f\x75\x72\x63\x65\x52\x65\x66\x23\x22\x20\x78\x6d\x70\x3a\x43\
\x72\x65\x61\x74\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\
\x65\x20\x50\x68\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x20\
\x4d\x61\x63\x69\x6e\x74\x6f\x73\x68\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x38\x30\x46\x38\x35\x36\x30\x34\x33\x41\x32\
\x39\x31\x31\x45\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\
\x45\x30\x43\x44\x32\x22\x20\x78\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\
\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\
\x3a\x38\x30\x46\x38\x35\x36\x30\x35\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\x3a\x44\x65\x72\x69\x76\
\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\x65\x66\x3a\x69\x6e\x73\
\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\x2e\x69\x69\x64\
\x3a\x38\x30\x46\x38\x35\x36\x30\x32\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x20\x73\x74\x52\x65\x66\x3a\x64\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x38\x30\x46\
\x38\x35\x36\x30\x33\x33\x41\x32\x39\x31\x31\x45\x32\x39\x42\x38\
\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\x32\x22\x2f\x3e\
\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\
\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x52\x44\x46\x3e\x20\x3c\
\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\x3e\x20\x3c\x3f\x78\x70\
\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\x22\x72\x22\x3f\x3e\xc2\
\x2a\xc3\x28\x00\x00\x01\x14\x49\x44\x41\x54\x78\xda\xec\x97\x41\
\x0a\x83\x30\x10\x45\x13\x29\x9e\x41\xf0\x1e\xdd\x76\xdb\xc3\x08\
\x1e\xaf\xdd\x14\x02\x82\xe0\x4a\xf0\x12\x5e\xa0\x1b\x0b\x92\xc6\
\x32\x01\x3b\x35\x21\x13\x8d\x2d\x25\x03\x7f\x93\x98\xfc\x97\x31\
\xe8\x0c\x93\x52\x32\x62\xa4\x4a\x85\x92\x50\x1a\x40\x02\xc6\x52\
\xea\x66\x8c\x08\x90\x2b\xb5\xd3\x32\x83\x5a\x78\x26\x08\xc0\x74\
\xba\xce\x62\xae\xd5\x91\x32\x41\x00\x28\x1c\xcc\xb5\x8a\x10\x00\
\x15\x32\x99\xde\xfb\x11\x24\xd0\x5c\x15\x02\x60\x40\x26\xd9\x6c\
\x2e\x43\x73\x83\x0f\xc0\x49\xa9\x21\xa4\xd9\x57\x0d\x78\x7d\x00\
\xf4\x3b\x98\x6b\xf5\xda\x34\x99\x25\x63\x64\xfb\xc5\xb8\x94\x81\
\xf3\x4e\x59\xe8\xc1\xeb\x15\x7c\x02\xe0\x9c\xbf\x5d\x0b\x44\xcb\
\x57\x9e\xd6\xba\x5f\xc2\xbe\x1c\x11\xc0\x07\xe0\x06\x72\x1d\xb7\
\x86\xcf\x25\x94\x9e\xe3\xf1\x12\x46\x80\x08\xf0\x9b\x00\x07\x8f\
\x35\x57\xe2\xf8\xe6\x1f\xa2\xff\xff\x1b\x3e\x16\x9a\x11\xdf\xc0\
\x6b\xef\x2e\x55\x71\x8d\x2a\x98\x0b\xaa\x80\x5d\x23\x83\xb5\xf3\
\xbd\x6a\x17\x80\x32\x60\x39\x56\xba\x00\xb8\xb6\x60\x54\x2d\xb7\
\x6c\x86\xc6\x24\xdf\x18\xa2\x33\xde\x25\x4b\x67\x94\x42\xca\xc4\
\x0a\xe3\x0a\xf6\x30\x36\xab\x4f\x01\x06\x00\xb9\xc0\x1f\x4e\xd5\
\xb4\x84\xa4\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x04\x49\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x22\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x30\x20\x36\x31\
\x2e\x31\x33\x34\x37\x37\x37\x2c\x20\x32\x30\x31\x30\x2f\x30\x32\
\x2f\x31\x32\x2d\x31\x37\x3a\x33\x32\x3a\x30\x30\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\
\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x4d\x4d\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x6d\x6d\x2f\x22\x20\x78\x6d\
\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\
\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\x79\x70\x65\x2f\x52\x65\x73\
\x6f\x75\x72\x63\x65\x52\x65\x66\x23\x22\x20\x78\x6d\x70\x3a\x43\
\x72\x65\x61\x74\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\
\x65\x20\x50\x68\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x20\
\x4d\x61\x63\x69\x6e\x74\x6f\x73\x68\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x42\x38\x31\x32\x34\x34\x41\x44\x33\x41\x32\
\x39\x31\x31\x45\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\
\x45\x30\x43\x44\x32\x22\x20\x78\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\
\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\
\x3a\x42\x38\x31\x32\x34\x34\x41\x45\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\x3a\x44\x65\x72\x69\x76\
\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\x65\x66\x3a\x69\x6e\x73\
\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\x2e\x69\x69\x64\
\x3a\x42\x38\x31\x32\x34\x34\x41\x42\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x20\x73\x74\x52\x65\x66\x3a\x64\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x42\x38\x31\
\x32\x34\x34\x41\x43\x33\x41\x32\x39\x31\x31\x45\x32\x39\x42\x38\
\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\x32\x22\x2f\x3e\
\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\
\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x52\x44\x46\x3e\x20\x3c\
\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\x3e\x20\x3c\x3f\x78\x70\
\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\x22\x72\x22\x3f\x3e\xcc\
\xbc\x0a\x97\x00\x00\x00\xbd\x49\x44\x41\x54\x78\xda\xec\x97\x31\
\x0a\x80\x30\x0c\x45\x8d\xb8\x7b\x01\xef\xe2\x24\x78\x13\x0f\xe1\
\x51\x05\xc1\x41\x9c\xf4\x06\x52\x23\x38\x88\xb4\x6a\x6b\xda\x8f\
\xd0\xc0\x5b\xea\xd0\x47\xa9\x3f\x29\x29\xa5\x12\x64\xa5\x09\xb8\
\xa2\x00\x5c\x20\xd3\x2d\x12\x91\x97\xcd\x74\x17\xfe\xee\x04\x4a\
\xa6\x63\x06\xa6\x65\x72\x6f\x56\x57\x8e\x9a\xf6\xcf\x27\xe6\xaf\
\x22\xda\xbd\x6e\x04\x94\x01\x67\x11\x29\x01\x67\x11\x69\x01\x6b\
\x11\x5f\x02\xaf\x45\x7c\x0b\x3c\x8a\x84\x12\x30\x8a\x84\x16\x38\
\x8b\xd4\x26\x01\xd2\xa5\xd3\x91\x84\x92\x7d\x7a\x64\x0a\xdb\x24\
\x94\xac\x15\xd9\x8c\x16\xa6\x71\x89\xe2\xdf\x5e\x42\xd8\x6f\x08\
\x0b\x22\x58\x14\xc3\x9a\x51\xb0\x76\x0c\x1f\x48\x2a\xa6\x97\x1c\
\xc9\x6c\xa3\x18\x3e\x94\xe2\xc6\xf2\x90\xef\xc5\xf8\x32\x82\x0b\
\x6c\x02\x0c\x00\xdf\x69\xaf\xa2\xf6\xc7\xa8\x9a\x00\x00\x00\x00\
\x49\x45\x4e\x44\xae\x42\x60\x82\
\x00\x00\x05\xc5\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x68\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x31\x20\x36\x34\
\x2e\x31\x34\x30\x39\x34\x39\x2c\x20\x32\x30\x31\x30\x2f\x31\x32\
\x2f\x30\x37\x2d\x31\x30\x3a\x35\x37\x3a\x30\x31\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x4d\x4d\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\
\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\
\x6d\x6d\x2f\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\
\x79\x70\x65\x2f\x52\x65\x73\x6f\x75\x72\x63\x65\x52\x65\x66\x23\
\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x4f\x72\x69\x67\x69\x6e\x61\x6c\x44\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x46\x34\
\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\x38\x31\x31\x38\x38\x43\
\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\x38\x31\x32\x22\x20\x78\
\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\
\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x30\x31\x32\x39\x34\x33\x41\
\x39\x32\x46\x45\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\
\x33\x46\x37\x31\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x30\x31\x32\x39\x34\x33\x41\x38\x32\x46\x45\
\x41\x31\x31\x45\x32\x38\x43\x30\x39\x39\x39\x32\x33\x46\x37\x31\
\x37\x31\x31\x41\x34\x22\x20\x78\x6d\x70\x3a\x43\x72\x65\x61\x74\
\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\x65\x20\x50\x68\
\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x2e\x31\x20\x4d\x61\
\x63\x69\x6e\x74\x6f\x73\x68\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\
\x3a\x44\x65\x72\x69\x76\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\
\x65\x66\x3a\x69\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\
\x6d\x70\x2e\x69\x69\x64\x3a\x34\x41\x45\x42\x37\x43\x34\x36\x32\
\x32\x32\x30\x36\x38\x31\x31\x38\x46\x36\x32\x42\x38\x38\x42\x42\
\x44\x44\x31\x46\x34\x46\x46\x22\x20\x73\x74\x52\x65\x66\x3a\x64\
\x6f\x63\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\
\x69\x64\x3a\x44\x46\x34\x39\x41\x45\x36\x46\x33\x44\x32\x30\x36\
\x38\x31\x31\x38\x38\x43\x36\x43\x42\x36\x33\x31\x44\x37\x36\x46\
\x38\x31\x32\x22\x2f\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\
\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\
\x52\x44\x46\x3e\x20\x3c\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\
\x3e\x20\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\
\x22\x72\x22\x3f\x3e\x26\x10\xd0\xea\x00\x00\x01\xf3\x49\x44\x41\
\x54\x78\xda\xbc\x56\x41\x67\x03\x51\x10\xde\xb6\x84\x9c\x7a\x2d\
\xa5\x84\xd0\x53\x29\xcb\x92\x53\x4f\x25\xf4\x54\x42\xc9\xb9\xbf\
\x60\xc9\x0f\x28\xb9\xe6\x7f\x84\x54\x7a\xaa\x1e\xaa\x94\x90\xe8\
\xa9\x95\x2a\xa1\x95\xd3\xd6\x46\x48\xe9\xa9\x42\xa2\x9d\xad\x59\
\xd6\xec\xbc\x79\x6f\xbb\xaf\xfb\xf1\x11\xcf\xcc\x37\x93\x37\xf3\
\x66\x67\xc3\x31\xc3\x01\xf0\x14\x78\x0c\x2c\x03\x0f\xf1\xfc\x09\
\xf8\x05\xbc\x05\x5e\x01\x9f\x1d\xcb\x88\x82\x4e\x80\xdf\x86\x8c\
\x6c\x1b\x36\x02\xef\x01\x87\x19\x02\x53\x0e\x51\xe3\x4f\x38\x02\
\x2e\x72\x04\x8f\xb9\x40\xad\xcc\xc1\x97\x8c\xd8\x0a\xd8\x07\x36\
\x81\x2e\x70\x0b\xe9\xe2\x59\x1f\x6d\xa8\xdf\x32\x4b\x12\x15\xe0\
\x5c\x51\x57\xcf\xc0\xdf\x53\xf4\xcb\x1c\xb5\xb5\x18\x30\xce\x1d\
\xec\x7a\x53\x94\xd1\x87\xea\x0c\x74\x8e\x0d\xc6\xa9\x95\xa3\x89\
\x5b\x8c\x9e\xf8\x3a\xe8\xd5\x75\x35\x7d\xf2\x80\x94\xea\xdb\x65\
\x4a\xa9\x14\xa4\x8d\xb3\x2b\x08\x87\x09\xdb\x50\xb0\xab\x32\x0d\
\x9d\x4a\x78\x13\x58\x27\x67\x97\xc0\x77\x41\x78\x47\xf1\x9b\xe2\
\x0d\xb5\x92\xa8\x73\x09\xd4\xc8\xd9\xb5\xc5\x49\x4a\xb5\x6a\x9c\
\x11\x7d\x7a\xfb\x8a\x9a\xeb\x06\x0f\xd7\x13\x15\xe6\x49\xa6\x40\
\xeb\xb4\x2d\xd4\x5c\x47\xda\x13\x25\xa6\xbf\x52\x25\xd0\x61\x9d\
\xe1\xca\xd7\x4c\x02\xa2\x56\x94\xc0\xa7\xd0\x64\x11\xce\x81\x33\
\x83\xe0\x33\xb4\x4d\x82\xbe\xa6\x0f\xce\xf1\x9e\x5c\x53\x53\x13\
\x88\x5e\xbb\x84\x26\xb1\xbd\xe3\x6e\x60\x44\xce\x4e\x2c\xbe\x02\
\xaa\x35\x32\x1d\x44\x55\x41\x34\x48\xd8\x06\x9a\x8f\x1b\x6d\x70\
\xcf\xc6\x28\xae\x63\xb7\x87\xdc\x60\x11\x46\xf1\x38\xeb\xc7\xc8\
\xcf\x71\xf5\x3e\xa3\xa7\xeb\xad\xd4\xe7\x38\x5a\x30\xda\xcc\x53\
\x92\x50\x42\x9f\x15\xb3\x1d\x69\x3f\xeb\x55\x0b\x0b\xc9\x8b\x30\
\xa4\x6e\x4c\x92\x90\x56\xb2\x1e\xf0\x0c\xd7\xb0\x18\x2e\x96\xaf\
\xa7\x58\xc9\xb8\x24\x4a\x26\x49\xd8\x58\x4a\x73\x25\x91\x77\x2d\
\x5f\xd8\x48\x22\x7e\x1d\xaf\x19\x02\x8f\xb1\x4c\x65\x0c\x62\x25\
\x89\xb8\xd6\x17\x78\x2b\x8f\x09\x91\x29\x8e\xd7\x36\xd3\xa8\x26\
\x49\x74\x9c\x7f\x86\x2e\x89\xc0\x29\x00\x52\x12\x13\xa7\x20\xa8\
\x92\xf0\x9d\x02\x11\x4f\xc9\x29\xfe\xf3\xdf\xe0\x3f\x02\x0c\x00\
\x12\x4a\x70\xda\x17\x63\xb5\xbb\x00\x00\x00\x00\x49\x45\x4e\x44\
\xae\x42\x60\x82\
\x00\x00\x04\xf5\
\x89\
\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\
\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\
\x00\x00\x00\x19\x74\x45\x58\x74\x53\x6f\x66\x74\x77\x61\x72\x65\
\x00\x41\x64\x6f\x62\x65\x20\x49\x6d\x61\x67\x65\x52\x65\x61\x64\
\x79\x71\xc9\x65\x3c\x00\x00\x03\x22\x69\x54\x58\x74\x58\x4d\x4c\
\x3a\x63\x6f\x6d\x2e\x61\x64\x6f\x62\x65\x2e\x78\x6d\x70\x00\x00\
\x00\x00\x00\x3c\x3f\x78\x70\x61\x63\x6b\x65\x74\x20\x62\x65\x67\
\x69\x6e\x3d\x22\xef\xbb\xbf\x22\x20\x69\x64\x3d\x22\x57\x35\x4d\
\x30\x4d\x70\x43\x65\x68\x69\x48\x7a\x72\x65\x53\x7a\x4e\x54\x63\
\x7a\x6b\x63\x39\x64\x22\x3f\x3e\x20\x3c\x78\x3a\x78\x6d\x70\x6d\
\x65\x74\x61\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x3d\x22\x61\x64\x6f\
\x62\x65\x3a\x6e\x73\x3a\x6d\x65\x74\x61\x2f\x22\x20\x78\x3a\x78\
\x6d\x70\x74\x6b\x3d\x22\x41\x64\x6f\x62\x65\x20\x58\x4d\x50\x20\
\x43\x6f\x72\x65\x20\x35\x2e\x30\x2d\x63\x30\x36\x30\x20\x36\x31\
\x2e\x31\x33\x34\x37\x37\x37\x2c\x20\x32\x30\x31\x30\x2f\x30\x32\
\x2f\x31\x32\x2d\x31\x37\x3a\x33\x32\x3a\x30\x30\x20\x20\x20\x20\
\x20\x20\x20\x20\x22\x3e\x20\x3c\x72\x64\x66\x3a\x52\x44\x46\x20\
\x78\x6d\x6c\x6e\x73\x3a\x72\x64\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x77\x77\x77\x2e\x77\x33\x2e\x6f\x72\x67\x2f\x31\x39\x39\
\x39\x2f\x30\x32\x2f\x32\x32\x2d\x72\x64\x66\x2d\x73\x79\x6e\x74\
\x61\x78\x2d\x6e\x73\x23\x22\x3e\x20\x3c\x72\x64\x66\x3a\x44\x65\
\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x72\x64\x66\x3a\x61\x62\
\x6f\x75\x74\x3d\x22\x22\x20\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\
\x3d\x22\x68\x74\x74\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\
\x65\x2e\x63\x6f\x6d\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x22\x20\
\x78\x6d\x6c\x6e\x73\x3a\x78\x6d\x70\x4d\x4d\x3d\x22\x68\x74\x74\
\x70\x3a\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\
\x2f\x78\x61\x70\x2f\x31\x2e\x30\x2f\x6d\x6d\x2f\x22\x20\x78\x6d\
\x6c\x6e\x73\x3a\x73\x74\x52\x65\x66\x3d\x22\x68\x74\x74\x70\x3a\
\x2f\x2f\x6e\x73\x2e\x61\x64\x6f\x62\x65\x2e\x63\x6f\x6d\x2f\x78\
\x61\x70\x2f\x31\x2e\x30\x2f\x73\x54\x79\x70\x65\x2f\x52\x65\x73\
\x6f\x75\x72\x63\x65\x52\x65\x66\x23\x22\x20\x78\x6d\x70\x3a\x43\
\x72\x65\x61\x74\x6f\x72\x54\x6f\x6f\x6c\x3d\x22\x41\x64\x6f\x62\
\x65\x20\x50\x68\x6f\x74\x6f\x73\x68\x6f\x70\x20\x43\x53\x35\x20\
\x4d\x61\x63\x69\x6e\x74\x6f\x73\x68\x22\x20\x78\x6d\x70\x4d\x4d\
\x3a\x49\x6e\x73\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\
\x2e\x69\x69\x64\x3a\x44\x39\x45\x32\x39\x32\x44\x37\x33\x41\x32\
\x39\x31\x31\x45\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\
\x45\x30\x43\x44\x32\x22\x20\x78\x6d\x70\x4d\x4d\x3a\x44\x6f\x63\
\x75\x6d\x65\x6e\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\
\x3a\x44\x39\x45\x32\x39\x32\x44\x38\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x3e\x20\x3c\x78\x6d\x70\x4d\x4d\x3a\x44\x65\x72\x69\x76\
\x65\x64\x46\x72\x6f\x6d\x20\x73\x74\x52\x65\x66\x3a\x69\x6e\x73\
\x74\x61\x6e\x63\x65\x49\x44\x3d\x22\x78\x6d\x70\x2e\x69\x69\x64\
\x3a\x44\x39\x45\x32\x39\x32\x44\x35\x33\x41\x32\x39\x31\x31\x45\
\x32\x39\x42\x38\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\
\x32\x22\x20\x73\x74\x52\x65\x66\x3a\x64\x6f\x63\x75\x6d\x65\x6e\
\x74\x49\x44\x3d\x22\x78\x6d\x70\x2e\x64\x69\x64\x3a\x44\x39\x45\
\x32\x39\x32\x44\x36\x33\x41\x32\x39\x31\x31\x45\x32\x39\x42\x38\
\x36\x46\x43\x39\x37\x44\x38\x42\x45\x30\x43\x44\x32\x22\x2f\x3e\
\x20\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\
\x6f\x6e\x3e\x20\x3c\x2f\x72\x64\x66\x3a\x52\x44\x46\x3e\x20\x3c\
\x2f\x78\x3a\x78\x6d\x70\x6d\x65\x74\x61\x3e\x20\x3c\x3f\x78\x70\
\x61\x63\x6b\x65\x74\x20\x65\x6e\x64\x3d\x22\x72\x22\x3f\x3e\x96\
\x5b\x4e\x6f\x00\x00\x01\x69\x49\x44\x41\x54\x78\xda\xec\x57\x4d\
\xaa\xc2\x30\x10\x8e\x15\x7a\x86\x80\x97\x70\xf5\xb6\x6f\xfb\x40\
\x70\xe5\x56\xf0\x00\x85\x1e\xcf\x6d\x41\x28\x04\x04\xa1\x37\x70\
\xe5\x0d\x04\xc1\x52\xa7\x92\xf2\xe2\x90\x9f\xf9\xac\xe0\xa6\x03\
\x1f\x94\x24\xf3\xcd\x64\x32\xcd\x4c\x94\xfa\x97\x9c\x50\x10\x2a\
\xc2\xcd\xa2\xb2\x63\xb9\x4a\xcb\x1f\xc1\x10\xce\x84\x5f\x05\xca\
\x82\x70\x22\x74\x01\x9c\xec\x9a\x90\x6c\x08\x77\x67\xfd\x05\x31\
\xde\xef\xae\x89\x18\x1f\xd0\x04\x22\xc1\x8d\x0f\x10\x4b\x21\x30\
\x3e\xa0\x10\x1a\x87\x1c\x38\x30\xc5\xfe\xdc\x7f\x2c\x2a\x36\x77\
\x70\xf4\x56\x11\xe3\x1d\x92\x43\x37\xa6\xa8\x9d\x39\xed\x21\x1d\
\xe4\x02\x44\x2e\x98\x43\x19\xa1\x65\x63\xf3\xc0\xb7\x62\x6b\x5b\
\x20\xca\x4b\xc2\x3e\x14\x09\xc3\xbc\xdd\xdb\x9d\x6b\xfb\xed\xce\
\x19\x47\x6f\x9d\x38\x02\x49\x0e\x3d\x65\x07\x10\xec\x80\x24\x4c\
\xe5\xd0\x4b\x98\x8f\xc2\x73\x9c\x83\xbf\x61\x2c\x87\x5e\x44\x27\
\x9c\x38\xb2\xe4\x44\x2e\x22\xe8\xf7\xdc\x7a\x14\xb6\xc2\x44\xeb\
\xaf\xe2\xda\x73\x15\x47\x1d\x98\x79\x88\x3a\xc1\x1a\x44\xa2\x7c\
\x99\xfa\xb2\x4c\x0e\x4c\x0e\x4c\x0e\x64\x9e\x8b\x48\x09\xc6\xa4\
\x22\xe6\xd3\x9e\xaa\xc8\xab\xa0\x06\x0c\x43\x7c\x63\x8b\xd1\xe8\
\xe2\x36\xa6\x1c\x7f\xa4\xbc\x1b\xa0\x27\x34\x02\x07\x60\xbe\x2b\
\xd0\x13\x5e\x05\x0e\x40\x7c\x99\xe7\x5c\xa5\x3d\x61\x2c\x07\x20\
\xbe\x1a\xe8\x09\x6b\x81\x03\x30\x5f\x09\x24\x4d\x29\x70\x00\xe6\
\x1b\xfb\x34\xfb\xc8\x53\x6f\x91\x50\x6a\x12\x8f\x53\xdf\x63\x17\
\xe6\xcb\x6d\x48\x2a\xd6\x46\x97\xc2\x9d\xbf\xc5\xf7\x10\x60\x00\
\x80\x31\x6b\x23\x79\x2c\x08\xa3\x00\x00\x00\x00\x49\x45\x4e\x44\
\xae\x42\x60\x82\
"

qt_resource_name = "\
\x00\x06\
\x07\x03\x7d\xc3\
\x00\x69\
\x00\x6d\x00\x61\x00\x67\x00\x65\x00\x73\
\x00\x05\
\x00\x6f\xa6\x53\
\x00\x69\
\x00\x63\x00\x6f\x00\x6e\x00\x73\
\x00\x0e\
\x00\x0f\x42\x87\
\x00\x70\
\x00\x68\x00\x6f\x00\x65\x00\x62\x00\x65\x00\x2d\x00\x67\x00\x75\x00\x69\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x0c\x33\x5a\x87\
\x00\x68\
\x00\x65\x00\x6c\x00\x70\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x07\
\x06\xc1\x57\xa7\
\x00\x70\
\x00\x65\x00\x6e\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0c\
\x07\x84\x37\xa7\
\x00\x65\
\x00\x6c\x00\x6c\x00\x69\x00\x70\x00\x73\x00\x69\x00\x73\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x07\
\x07\xa7\x57\x87\
\x00\x61\
\x00\x64\x00\x64\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x07\
\x0c\xf8\x57\x87\
\x00\x65\
\x00\x79\x00\x65\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0c\
\x05\xc9\x18\x47\
\x00\x64\
\x00\x61\x00\x74\x00\x61\x00\x62\x00\x61\x00\x73\x00\x65\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x08\x0b\x2f\xe7\
\x00\x6d\
\x00\x65\x00\x6e\x00\x75\x00\x2d\x00\x32\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x0c\xad\x0f\x07\
\x00\x64\
\x00\x65\x00\x6c\x00\x65\x00\x74\x00\x65\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0b\
\x0c\x6a\x2c\x47\
\x00\x72\
\x00\x65\x00\x66\x00\x72\x00\x65\x00\x73\x00\x68\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x0c\x9e\x4e\x27\
\x00\x72\
\x00\x65\x00\x74\x00\x75\x00\x72\x00\x6e\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x07\
\x07\x63\x57\xa7\
\x00\x70\
\x00\x6f\x00\x70\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x00\xa7\x59\x27\
\x00\x6c\
\x00\x69\x00\x73\x00\x74\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0c\
\x0b\xdf\x21\x47\
\x00\x73\
\x00\x65\x00\x74\x00\x74\x00\x69\x00\x6e\x00\x67\x00\x73\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x08\x4a\xc9\x87\
\x00\x65\
\x00\x78\x00\x70\x00\x61\x00\x6e\x00\x64\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x08\xf7\x5a\x87\
\x00\x67\
\x00\x72\x00\x69\x00\x64\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x04\x0a\x1b\xc7\
\x00\x63\
\x00\x6f\x00\x6d\x00\x6d\x00\x69\x00\x74\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0a\
\x02\xca\x6f\x27\
\x00\x62\
\x00\x75\x00\x6c\x00\x6c\x00\x65\x00\x74\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0c\
\x08\x1d\x80\x47\
\x00\x72\
\x00\x65\x00\x70\x00\x65\x00\x61\x00\x74\x00\x2d\x00\x32\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0c\
\x06\xeb\x97\xe7\
\x00\x7a\
\x00\x6f\x00\x6f\x00\x6d\x00\x2d\x00\x6f\x00\x75\x00\x74\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x00\x4e\x59\x27\
\x00\x6c\
\x00\x69\x00\x6e\x00\x6b\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x09\
\x08\x97\x84\x87\
\x00\x63\
\x00\x68\x00\x61\x00\x72\x00\x74\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x04\xd2\x59\x47\
\x00\x69\
\x00\x6e\x00\x66\x00\x6f\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x09\
\x01\x06\x85\x47\
\x00\x62\
\x00\x69\x00\x6e\x00\x2d\x00\x33\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x02\x8c\x59\xa7\
\x00\x70\
\x00\x6c\x00\x61\x00\x79\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x0b\
\x03\x03\x9b\x47\
\x00\x7a\
\x00\x6f\x00\x6f\x00\x6d\x00\x2d\x00\x69\x00\x6e\x00\x2e\x00\x70\x00\x6e\x00\x67\
\x00\x08\
\x0c\x2f\x59\xa7\
\x00\x70\
\x00\x75\x00\x6c\x00\x6c\x00\x2e\x00\x70\x00\x6e\x00\x67\
"

qt_resource_struct = "\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x02\
\x00\x00\x00\x22\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
\x00\x00\x00\x12\x00\x02\x00\x00\x00\x1a\x00\x00\x00\x04\
\x00\x00\x02\x24\x00\x00\x00\x00\x00\x01\x00\x00\x60\x6e\
\x00\x00\x01\x50\x00\x00\x00\x00\x00\x01\x00\x00\x3c\x97\
\x00\x00\x02\x68\x00\x00\x00\x00\x00\x01\x00\x00\x6a\x22\
\x00\x00\x02\x80\x00\x00\x00\x00\x00\x01\x00\x00\x6e\xc6\
\x00\x00\x01\xce\x00\x00\x00\x00\x00\x01\x00\x00\x51\x3b\
\x00\x00\x02\x96\x00\x00\x00\x00\x00\x01\x00\x00\x73\x13\
\x00\x00\x01\xb4\x00\x00\x00\x00\x00\x01\x00\x00\x4c\xf1\
\x00\x00\x02\x52\x00\x00\x00\x00\x00\x01\x00\x00\x69\x3b\
\x00\x00\x00\xb4\x00\x00\x00\x00\x00\x01\x00\x00\x20\xc8\
\x00\x00\x00\x5a\x00\x00\x00\x00\x00\x01\x00\x00\x0f\xc4\
\x00\x00\x02\x06\x00\x00\x00\x00\x00\x01\x00\x00\x5a\xbd\
\x00\x00\x01\x3c\x00\x00\x00\x00\x00\x01\x00\x00\x37\xa1\
\x00\x00\x00\x6e\x00\x00\x00\x00\x00\x01\x00\x00\x14\x88\
\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x01\x00\x00\x18\xe8\
\x00\x00\x00\xd2\x00\x00\x00\x00\x00\x01\x00\x00\x25\xfd\
\x00\x00\x01\xe8\x00\x00\x00\x00\x00\x01\x00\x00\x55\xb3\
\x00\x00\x01\x84\x00\x00\x00\x00\x00\x01\x00\x00\x43\x33\
\x00\x00\x02\x3a\x00\x00\x00\x00\x00\x01\x00\x00\x66\x06\
\x00\x00\x01\x9e\x00\x00\x00\x00\x00\x01\x00\x00\x48\x4f\
\x00\x00\x01\x66\x00\x00\x00\x00\x00\x01\x00\x00\x3d\xa9\
\x00\x00\x02\xb2\x00\x00\x00\x00\x00\x01\x00\x00\x78\xdc\
\x00\x00\x00\x44\x00\x00\x00\x00\x00\x01\x00\x00\x0e\xdf\
\x00\x00\x01\x06\x00\x00\x00\x00\x00\x01\x00\x00\x2d\x83\
\x00\x00\x01\x22\x00\x00\x00\x00\x00\x01\x00\x00\x32\xe9\
\x00\x00\x00\xec\x00\x00\x00\x00\x00\x01\x00\x00\x2a\x25\
\x00\x00\x00\xa0\x00\x00\x00\x00\x00\x01\x00\x00\x1b\x64\
"

def qInitResources():
    QtCore.qRegisterResourceData(0x01, qt_resource_struct, qt_resource_name, qt_resource_data)

def qCleanupResources():
    QtCore.qUnregisterResourceData(0x01, qt_resource_struct, qt_resource_name, qt_resource_data)

qInitResources()
