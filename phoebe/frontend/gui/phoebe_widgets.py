from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

import ui_phoebe_pyqt4 as gui

##### Added for the OpenGL renderer mesh_widget
##### The pyphoebe.parameters import create line 
##### should be removed once we establish the system's
##### mesh as an argument to the widget
try:
    from PyQt4.QtOpenGL import *
    from OpenGL.GL import *
except:
    pass
from phoebe.parameters import create
from phoebe.utils import callbacks
from phoebe.units import constants
from phoebe.frontend import phcompleter
################################################

from collections import OrderedDict

Signal = pyqtSignal
Slot = pyqtSlot
Property = pyqtProperty

import os
import re
import sys
import code
import rlcompleter
import readline
import inspect
import json

import time

class CreatePopPriorEdit(QDialog, gui.Ui_popPriorEdit_Dialog):
    #in ui_phoebe_pyqt4.py, this needs to come before the import statements for phoebe_widgets
    def __init__(self, parent=None):
        super(CreatePopPriorEdit, self).__init__(parent)
        self.setupUi(self)

class CreatePopParamEdit(QDialog, gui.Ui_popParamEdit_Dialog):
    #in ui_phoebe_pyqt4.py, this needs to come before the import statements for phoebe_widgets
    def __init__(self, parent=None):
        super(CreatePopParamEdit, self).__init__(parent)
        self.setupUi(self)
        
class StatusPushButton(QPushButton):
    def __init__(self, parent=None, server=None, job=None, enabled=True, from_last_known=False):
        """
        provide server (object) and job (script name string) for job
        or just server (object) for server
        """
        super(StatusPushButton, self).__init__(parent)
        
        icon = QIcon()
        icon.addPixmap(QPixmap(":/images/icons/bullet.png"), QIcon.Normal, QIcon.Off)
        self.setIcon(icon)
        
        self.setFlat(True)
        self.setIconSize(QSize(32, 32))
        self.setMaximumSize(QSize(18,18))
        self.setEnabled(enabled)
    
        self.server = server
        self.job = job
        
        self.connect(self, SIGNAL("clicked()"), self.update_status)
        self.update_status(from_last_known=from_last_known)
            
        
        #~ self.status_icon = QIcon()
        #~ self.status_icon.addPixmap(QPixmap(":/images/icons/bullet.png"), QIcon.Normal, QIcon.Off)
        #~ self.setIcon(self.status_icon)
        
    def update_status(self,from_last_known=False):
        if self.job is not None and self.server is not None:
            status = self.server.check_script_status(self.job)
            self.setToolTip('job: %s' % status)
            if status == 'running':
                self.colorize('yellow')
            elif status == 'complete':
                self.colorize('green')
            else:
                self.colorize('red')
            return
        elif self.server is not None:
            # then this is for a server
            if from_last_known:
                status = self.server.last_known_status['status']
            else:
                status = self.server.check_status()
        else:
            status = True

        if status:
            self.setToolTip("ok")
        else:
            lks = self.server.last_known_status
            if lks['mount'] and lks['ping']:
                failed = [f for f in lks.keys() if lks[f]==False and f not in ['status']]
            else:
                failed = [f for f in lks.keys() if lks[f]==False and f not in ['status','test']]
            self.setToolTip("failed: %s" % " ".join(failed))
        self.colorize('green' if status else 'red')
        
        self.emit(SIGNAL("serverStatusChanged"),False)
    
    def colorize(self,color):
        colorize = QGraphicsColorizeEffect()
        qcolor = QColor()                                
        qcolor.setNamedColor(color)
        colorize.setColor(qcolor)
        self.setGraphicsEffect(colorize)
        
        self.setStyleSheet('QPushButton {background-color: blue};QPushButton:pressed {background-color: blue};')
        
class GeneralParameterTreeWidget(QTreeWidget):
    """
    probably not used on its own, but all versions of the parameter tree widget can be subclassed from this
    
    each subclass of this at least needs to have:
    set_data()
    
    and optionally:
    item_clicked()
    item_changed()
    """
    def __init__(self,parent = None):
        super(GeneralParameterTreeWidget, self).__init__()
        
        self.myparent = parent
        self.items = None
        self.selected_item = None
        self.style = []
        
        self.info_icon = QIcon()
        self.info_icon.addPixmap(QPixmap(":/images/icons/info.png"), QIcon.Normal, QIcon.Off)

        self.edit_icon = QIcon()
        self.edit_icon.addPixmap(QPixmap(":/images/icons/pen.png"), QIcon.Normal, QIcon.Off)
        
        self.link_icon = QIcon()
        self.link_icon.addPixmap(QPixmap(":/images/icons/link.png"), QIcon.Normal, QIcon.Off)
        
        self.list_icon = QIcon()
        self.list_icon.addPixmap(QPixmap(":/images/icons/list.png"), QIcon.Normal, QIcon.Off)
        
        self.reload_icon = QIcon()
        self.reload_icon.addPixmap(QPixmap(":/images/icons/refresh.png"), QIcon.Normal, QIcon.Off)
        
        self.delete_icon = QIcon()
        self.delete_icon.addPixmap(QPixmap(":/images/icons/bin-3.png"), QIcon.Normal, QIcon.Off)
        
        self.data_icon = QIcon()
        self.data_icon.addPixmap(QPixmap(":/images/icons/database.png"), QIcon.Normal, QIcon.Off)

        self.plot_icon = QIcon()
        self.plot_icon.addPixmap(QPixmap(":/images/icons/chart.png"), QIcon.Normal, QIcon.Off)
        
        self.obs_icon = QIcon()
        self.obs_icon.addPixmap(QPixmap(":/images/icons/ellipsis.png"), QIcon.Normal, QIcon.Off)
        
        self.syn_icon = QIcon()
        self.syn_icon.addPixmap(QPixmap(":/images/icons/commit.png"), QIcon.Normal, QIcon.Off)
        
        self.add_icon = QIcon()
        self.add_icon.addPixmap(QPixmap(":/images/icons/add.png"), QIcon.Normal, QIcon.Off)
        
        self.eye_icon = QIcon()
        self.eye_icon.addPixmap(QPixmap(":/images/icons/eye.png"), QIcon.Normal, QIcon.Off)
        
        self.installEventFilter(self)
        self.connect(self, SIGNAL("itemClicked(QTreeWidgetItem*, int)"), self.item_clicked)
        self.connect(self, SIGNAL("currentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)"), self.item_changed) 
        self.connect(self, SIGNAL("itemSelectionChanged()"), self.item_changed) # should handle focus being stolen by other items
        
    def eventFilter(self,object,event):
        if event.type()== QEvent.FocusIn:
            self.emit(SIGNAL("focusIn"))
        #~ print event.type()
        #~ if event.type()==QEvent.FocusOut:
            #~ self.item_changed()
        return False
        

    def get_parent_label(self,param):
        unique_label = param.get_unique_label()
        for obj,name in zip(self.system_ps,self.system_names):
            if unique_label in [obj.get_parameter(k).get_unique_label() for k in obj.keys()]:
                return name
                
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.item_changed()
        elif event.key() == Qt.Key_Escape:
            self.item_changed(change_value=False)
        elif event.key() == Qt.Key_Down:
            item_below = self.itemBelow(self.currentItem())
            if item_below:
                self.setCurrentItem(item_below)
        elif event.key() == Qt.Key_Up:
            item_above=self.itemAbove(self.currentItem ())
            if item_above:
                self.setCurrentItem(item_above)
                
    def item_clicked(self, item, col):
        pass
        
    def item_changed(self,change_value=True):
        pass

class AdjustableTreeWidget(GeneralParameterTreeWidget):
    """
    tree view in rightpane that shows parameters set for adjustment
    and allows editing of their priors
    """
    # required functions
    def set_data(self,data,system_ps,system_names):
        '''
        data - system.get_adjustable_parameters()
        '''
        self.clear()
        #~ self.items = OrderedDict()
        self.params = []
        self.items = []
        self.system_ps = system_ps
        self.system_names = system_names
        
        ncol = 2
        self.setColumnCount(ncol)
        self.setHeaderLabels(['Parameter','Prior'])
        
        for n,param in enumerate(data):
            parentLabel = self.get_parent_label(param)
            
            if (parentLabel, param.get_qualifier()) in self.params or param.get_qualifier()=='pblum' or param.get_qualifier()=='l3':
                continue
                
            self.params.append((parentLabel, param.get_qualifier()))
                            
            # create general item
            #~ item = QTreeWidgetItem(['','%s: %s' % (parentLabel, param.get_qualifier()),''])
            item = QTreeWidgetItem([])
            item.info = {'param': param, 'distribution': param.get_prior().get_distribution()}
            self.addTopLevelItem(item)
            
            # create adjustable widget
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(0)
            frame.setLayout(HBox)
            
            check = QCheckBox()
            item.info['check'] = check
            check.setToolTip('unmark for adjustment/fitting')
            check.info = {'param': param, 'parentLabel': parentLabel, 'state': param.get_adjust()}
            check.setMaximumSize(QSize(18, 18))
            check.setCheckState(2 if param.get_adjust() is True else 0)
            check.setVisible(False)
            HBox.addWidget(check)
            
            label = QLabel(' %s@%s        ' % (param.get_qualifier(), parentLabel))
            HBox.addWidget(label)
            
            self.setItemWidget(item,0,frame)
            
            # create prior widget
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(0)
            frame.setLayout(HBox)
            
            #~ combo = QComboBox()
            #~ combo.setMaximumSize(QSize(300,18))
            #~ combo.addItems(['uniform','normal'])
            #~ HBox.addWidget(combo)
            distribution = param.get_prior().get_distribution()
            label = QLabel(self.getLabelText(distribution))
            item.info['label'] = label
            HBox.addWidget(label)
            
            button=QPushButton()
            item.info['edit'] = button
            button.setToolTip('edit prior')
            button.setIcon(self.edit_icon)
            button.setIconSize(QSize(12, 12))
            #~ button.info = {'param': par, 'widget': widget}
            button.setMaximumSize(QSize(18,18))
            QObject.connect(button, SIGNAL("clicked()"), self.on_editbutton_clicked)
            button.setVisible(False)
            HBox.addWidget(button)
            
            self.items.append(item)
            self.setItemWidget(item,1,frame)
            
        for i in range(ncol):
            self.resizeColumnToContents(i)
        self.resizeColumnToContents(0)
        
    def item_clicked(self, item, col):
        check = item.info['check']
        param = check.info['param']
        check.setCheckState(2 if param.get_adjust() is True else 0)
        check.setVisible(True)
        item.info['edit'].setVisible(True)
        self.selected_item = item
        
    def item_changed(self,change_value=True):
        # check if adjust has changed
        if self.selected_item:
            item = self.selected_item
            check = item.info['check']
            param = item.info['param']
            label = check.info['parentLabel']
            
            new_check = True if check.checkState()==2 else False        
            
            # check if prior changed
            if new_check and change_value:
                if param.get_prior().get_distribution() != item.info['distribution']:
                    self.emit(SIGNAL("priorChanged"),label,param,item.info['distribution'])

            # reset label in case of esc
            if not change_value:
                item.info['label'].setText(self.getLabelText(param.get_prior().get_distribution()))

            # remove from list if check is false
            if change_value and new_check == False:
                self.emit(SIGNAL("parameterChanged"),self,label,param,True,new_check,None,None,True)
                
            # reset visibility
            if new_check or not change_value: #item will have been deleted if new_check == False
                #~ for item in self.items:
                item.info['check'].setVisible(False)
                item.info['edit'].setVisible(False)
            self.selected_item = None
            
    # functions for popup windows, etc
    def on_editbutton_clicked(self):
        item = self.selected_item
        param = item.info['param']
        lims = param.get_limits()
        distribution = item.info['distribution']
        
        pop = CreatePopPriorEdit()
        pop.distributionCombo.setCurrentIndex(pop.distributionCombo.findText(distribution[0]))

        pop.uniform_lower.setRange(*lims)
        pop.uniform_lower.setValue(lims[0])
        pop.uniform_upper.setRange(*lims)
        pop.uniform_upper.setValue(lims[1])
        
        pop.normal_mu.setRange(*lims)
        pop.normal_mu.setValue(param.get_value())
        pop.normal_sigma.setRange(0,1E6)
        pop.normal_sigma.setValue(1.)

        if distribution[0]=='uniform':
            pop.uniform_lower.setValue(distribution[1]['lower'])
            pop.uniform_upper.setValue(distribution[1]['upper'])
        elif distribution[0]=='normal':
            pop.normal_mu.setValue(distribution[1]['mu'])
            pop.normal_sigma.setValue(distribution[1]['sigma'])
        
        result = pop.exec_()

        if result:
            disttype = str(pop.distributionCombo.currentText())
            if disttype=='uniform':
                item.info['distribution'] = [disttype, {'lower': pop.uniform_lower.value(), 'upper': pop.uniform_upper.value()}]
            elif disttype=='normal':
                item.info['distribution'] = [disttype, {'mu': pop.normal_mu.value(), 'sigma': pop.normal_sigma.value()}]
            #~ item.info['label'].setText(self.getLabelText(item.info['distribution']))
            self.item_changed()
            
    def getLabelText(self,distribution):
        # TODO more intelligent float formatting (scientific notation?)
        if distribution[0]=='uniform':
            return "%s: %.2f,%.2f" % (distribution[0], distribution[1]['lower'], distribution[1]['upper'])
        elif distribution[0]=='normal':
            return "%s: %.2f,%.2f" % (distribution[0], distribution[1]['mu'], distribution[1]['sigma'])
        else:
            return distribution[0]
            
class DictDataTreeWidget(GeneralParameterTreeWidget):
    """
    tree view designed to show versions/feedbacks
    simple subclassing of this will allow those two to be separate
    
    """
    def _set_data(self,data,kind='version'):
        '''
        data - bundle.versions or bundle.feedbacks
             - list of dicts
        '''
        self.kind = kind
        self.clear()
        self.versions = []
        
        self.setColumnCount(1)
        
        for n,version in enumerate(data):
            stackedwidget = QStackedWidget()
            
            
            ## normal text view
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            label = QLabel(QString(version['name']))
            font = QFont()
            #~ font.setBold(dataset.get_enabled() is True)
            label.setFont(font)
            #~ label.setMinimumSize(QSize(400,18))
            HBox.addWidget(label)
            
            stackedwidget.addWidget(frame)
            
            
            ## selected view with widgets
            
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            restore_button = QPushButton()
            restore_button.setIcon(self.eye_icon)
            restore_button.setMaximumSize(QSize(18, 18))
            restore_button.info = {'version': version}
            if kind=='version':
                restore_button.setToolTip("restore this version as the working version")
                QObject.connect(restore_button, SIGNAL('clicked()'), self.on_restoreversion_clicked)
            elif kind=='feedback':
                restore_button.setToolTip("examine this feedback")
                QObject.connect(restore_button, SIGNAL('clicked()'), self.on_examinefeedback_clicked)
            HBox.addWidget(restore_button)
            
            label = QLabel(QString(version['name']))
            #~ font = QFont()
            #~ font.setBold(dataset.get_enabled() is True)
            #~ label.setFont(font)
            #~ label.setMinimumSize(QSize(400,18))
            HBox.addWidget(label)
            
            edit_button = QPushButton()
            edit_button.setIcon(self.edit_icon)
            edit_button.setMaximumSize(QSize(18, 18))
            edit_button.setToolTip("change name for this %s" % kind)
            edit_button.info = {'stackedwidget': stackedwidget}
            #~ edit_button.setEnabled(False) #until signals attached
            QObject.connect(edit_button, SIGNAL('clicked()'), self.on_edit_clicked)
            HBox.addWidget(edit_button)

            delete_button = QPushButton()
            delete_button.setIcon(self.delete_icon)
            delete_button.setMaximumSize(QSize(18, 18))
            delete_button.setToolTip("remove this %s" % (kind))
            delete_button.info = {'version': version}
            QObject.connect(delete_button, SIGNAL('clicked()'), self.on_remove_clicked) 
            HBox.addWidget(delete_button)
            
            stackedwidget.addWidget(frame)
            
            
            ## edit name
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            name_edit = QLineEdit(version['name'])
            name_edit.setMaximumSize(QSize(2000,18))            
            HBox.addWidget(name_edit)
            
            stackedwidget.addWidget(frame)
            
            ## add item to tree view
            
            item = QTreeWidgetItem()
            item.info = {}
            item.info = {'stackedwidget': stackedwidget, 'version': version, 'name_edit': name_edit}
            self.addTopLevelItem(item)
            self.setItemWidget(item,0,stackedwidget)
            
    def item_clicked(self, item, col):
        self.selected_item = (item,col)
        
        stackedwidget = item.info['stackedwidget']
        stackedwidget.setCurrentIndex(1)
        
    def item_changed(self,change_value=True):
        if not self.selected_item:
            return

        item, col = self.selected_item            
        stackedwidget = item.info['stackedwidget']
        version = item.info['version']
        
        if change_value and stackedwidget.currentIndex()==2:
            # get info and send signals to create commands
            name_edit = item.info['name_edit']
            new_name = str(name_edit.text())
            if new_name != version['name']:
                do_command = "bundle.rename_%s('%s','%s')" % (self.kind,version['name'],new_name)
                undo_command = "bundle.rename_%s('%s','%s')" % (self.kind,new_name,version['name'])
                description = "change name of %s to %s" % (self.kind,new_name)
                self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                
        # reset state
        stackedwidget.setCurrentIndex(0)
        self.selected_item = None
        
    def on_restoreversion_clicked(self):
        version = self.sender().info['version']
                
        do_command = "bundle.restore_version('%s')" % version['name']
        undo_command = "print 'undo is not available for this action'"
        # TODO - try to make this undoable
        description = "restore %s version" % version['name']
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
 
    def on_examinefeedback_clicked(self):
        feedback = self.sender().info['version']
        
        self.emit(SIGNAL("feedbackExamine"),feedback['name'])
        # this signal should change the stackedwidget and load this feedback in the treeview/plots
        
    def on_remove_clicked(self):
        version = self.sender().info['version']
                
        do_command = "bundle.remove_%s('%s')" % (self.kind,version['name'])
        undo_command = "print 'undo is not available for this action'"
        # TODO - try to make this undoable
        description = "remove %s %s" % (version['name'],self.kind)
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
        
    def on_edit_clicked(self):
        stackedwidget = self.sender().info['stackedwidget']
        
        stackedwidget.setCurrentIndex(2)
            
class VersionsTreeWidget(DictDataTreeWidget):
    """
    treeview that shows stored versions in the bundle
    and allows restoring/renaming
    """
    def set_data(self,data):
        # we do this so that we can easily subclass this for versions and feedbacks
        self._set_data(data,'version')
    
class FeedbacksTreeWidget(DictDataTreeWidget):
    """
    treeview that shows stored feedbacks in the bundle
    and allows restoring/renaming
    """
    def set_data(self,data):
        # we do this so that we can easily subclass this for versions and feedbacks
        self._set_data(data,'feedback')

class FittingTreeWidget(GeneralParameterTreeWidget):
    """
    treeview that shows the results from feedback
    namely old and new values for all parameters set for adjustment
    """
    # required functions
    def set_data(self,data,system_ps,system_names):
        '''
        data - system.get_adjustable_parameters()
        '''
        self.clear()
        self.params = []
        self.system_ps = system_ps
        self.system_names = system_names
        
        
        if data is None or len(data.keys())==0:    return

        feedback = data['feedback']

        ncol = 2
        self.setColumnCount(ncol)
        #~ self.setHeaderLabels(['Parameter','Old Value','New Value'])
        self.setHeaderLabels(['Parameter','New Value'])
        
        for param,value in zip(feedback['parameters'],feedback['values']):
            parentLabel = self.get_parent_label(param)
            
            stackedwidget = QStackedWidget()
            
            ## normal text view
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            label = QLabel(QString('%s@%s  ' % (param.get_qualifier(),parentLabel)))
            font = QFont()
            #~ font.setBold(dataset.get_enabled() is True)
            label.setFont(font)
            #~ label.setMinimumSize(QSize(400,18))
            HBox.addWidget(label)
            
            stackedwidget.addWidget(frame)
            
            
            ## selected view with widgets
            
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            examine_button = QPushButton()
            examine_button.setIcon(self.plot_icon)
            examine_button.setMaximumSize(QSize(18, 18))
            #~ examine_button.info = {}
            examine_button.setToolTip("view correlations and fitting statistics")
            #~ QObject.connect(examine_button, SIGNAL('clicked()'), self.on_examineversion_clicked)
            examine_button.setEnabled(False) # until signal connected
            HBox.addWidget(examine_button)
            
            label = QLabel(QString('%s@%s  ' % (param.get_qualifier(), parentLabel)))
            #~ font = QFont()
            #~ font.setBold(dataset.get_enabled() is True)
            #~ label.setFont(font)
            #~ label.setMinimumSize(QSize(400,18))
            HBox.addWidget(label)
            
            stackedwidget.addWidget(frame)
                
            # create general item
            item = QTreeWidgetItem(['',''])
            item.info = {'stackedwidget': stackedwidget}
            self.addTopLevelItem(item)
            self.setItemWidget(item,0,stackedwidget)
            
            
            # create label for second column
            label = QLabel(QString(str(value)))
            font = QFont()
            #~ font.setBold(dataset.get_enabled() is True)
            label.setFont(font)
            #~ label.setMinimumSize(QSize(400,18))
            self.setItemWidget(item,1,label)
            
        #~ for i in range(ncol):
            #~ self.resizeColumnToContents(i)
        self.resizeColumnToContents(0)
        
    def item_clicked(self, item, col):
        self.selected_item = (item,col)
        
        stackedwidget = item.info['stackedwidget']
        stackedwidget.setCurrentIndex(1)
        
    def item_changed(self,change_value=True):
        if not self.selected_item:
            return

        item, col = self.selected_item            
        stackedwidget = item.info['stackedwidget']
        
        # reset state
        stackedwidget.setCurrentIndex(0)
        self.selected_item = None
        
class ServerListTreeWidget(GeneralParameterTreeWidget):
    """
    parameter tree view in preferences:server
    """
    def set_data(self,servers):
        
        self.clear()
        self.servers = servers
        self.statusbuttons = []
        
        ### build rows
        
        for servername,server in servers.items():
            item = QTreeWidgetItem()
            #~ item.info = {'dataset': dataset, 'widgets': []}
            self.addTopLevelItem(item)
            
            # col 1 (status & servername)
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(4)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            statusbutton = StatusPushButton(server=server,from_last_known=True)        
            self.statusbuttons.append(statusbutton)  
            HBox.addWidget(statusbutton)
            self.connect(statusbutton, SIGNAL("serverStatusChanged"), self.on_server_status_changed)
            #~ statusbutton = StatusPushButton(server=server,server_kind='ping',from_last_known=True)        
            #~ self.statusbuttons.append(statusbutton)  
            #~ HBox.addWidget(statusbutton)
            #~ statusbutton = StatusPushButton(server=server,server_kind='test')        
            #~ self.statusbuttons.append(statusbutton)  
            #~ HBox.addWidget(statusbutton)

            label = QLabel(QString(servername)+"    ")
            HBox.addWidget(label)

            self.setItemWidget(item,0,frame)
            
            # col 2 (servername)
            mpi_ps = server.settings['mpi']
            if mpi_ps is None:
                mpi_type = "No MPI"
            else:
                mpi_type = "%s:np=%d" % (mpi_ps.context,mpi_ps.get_value('np'))
            
            label = QLabel(QString(mpi_type))
            self.setItemWidget(item,1,label)
            
            # col 3 (edit button)
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox)  
            
            spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
            HBox.addItem(spacer)           
            
            edit_button = QPushButton()
            edit_button.setIcon(self.edit_icon)
            edit_button.setMaximumSize(QSize(18, 18))
            edit_button.setToolTip("edit %s server" % servername)
            edit_button.info = {'server': server, 'servername': servername}
            QObject.connect(edit_button, SIGNAL('clicked()'), self.on_edit_clicked)
            HBox.addWidget(edit_button)

            delete_button = QPushButton()
            delete_button.setIcon(self.delete_icon)
            delete_button.setMaximumSize(QSize(18, 18))
            delete_button.setToolTip("remove %s server" % servername)
            delete_button.info = {'server': server, 'servername': servername}
            QObject.connect(delete_button, SIGNAL('clicked()'), self.on_delete_clicked)
            HBox.addWidget(delete_button)
            
            self.setItemWidget(item,2,frame)
            
            self.resizeColumnToContents(0)
            
    def on_edit_clicked(self,*args):
        w = self.sender()
        server, servername = w.info['server'], w.info['servername']

        self.emit(SIGNAL('edit_server_clicked'),servername,server)
        
    def on_delete_clicked(self,*args):
        w = self.sender()
        server, servername = w.info['server'], w.info['servername']
        
        self.emit(SIGNAL('delete_server_clicked'),servername,server)
        
    def on_server_status_changed(self,*args):
        self.emit(SIGNAL("serverStatusChanged"),False)
            
def has_ydata(dataset):
    category = dataset.context[:-3]
    if category in ['lc','sp']:
        ydata = dataset['flux']
    elif category == 'rv':
        ydata = dataset['rv']
    elif category == 'etv':
        ydata = dataset['etv']
    else:
        ydata = []
        
    return len(ydata)>0
            

class DatasetTreeWidget(GeneralParameterTreeWidget):
    """
    parameter tree view in bottom panel and any plot popups
    """
       
    def set_data(self,data_obs,data_syn,types,plots,bundle,system_ps,system_names,style=[]):
        '''
        data_obs,data_syn - list of data ParameterSets (datasets) to be shown in the treeview
            these should already be filtered to exclude duplicates (each entry will create a new row)
        '''
        self.clear()
        self.style=style
        self.system_ps = system_ps
        self.system_names = system_names
        
        self.selected_item = None
        
        if plots=='' or types=='':
            #then gui is not ready
            return
        
        ### filter which rows we want to show
        axes_incl = bundle._get_dict_of_section('axes', kind='Container').values() #default
        self.style = 'data' #default
        if plots!='all plots' or types!='all categories':
            if plots!='all plots':
                #~ axes_incl = bundle._get_by_search(plots, kind='Container', section='axes', all=True, ignore_errors=True)
                axes_incl = [bundle.get_axes(plots)]
                typ = axes_incl[0].get_value('category') # will be rv, lc, etc 
                self.style = 'plot'
            elif types!='all categories': #plot will automatically handle filtering by type
                typ = types
                
            # override the input lists for data_obs and data_syn to only show the requested rows
            data_obs = [dso for dso in data_obs if dso.context[:-3]==typ]
            data_syn = [dss for dss in data_syn if dss.context[:-3]==typ]
        
        # get plotted datasets, for all axes or current axes depending on selection
        plotted_obs = [] # holds datasets that are plotted
        plotted_syn = []
        plotted_obs_ps = [] # hold the plotoptions in axes_incl
        plotted_syn_ps = []
        for ax in axes_incl:
            for p in ax._get_dict_of_section('plot').values():
                if p['type'][-3:]=='obs':
                    plotted_obs.append(bundle.get_obs(objref=p['objref'],dataref=p['dataref']))
                    plotted_obs_ps.append(p)
                elif p['type'][-3:]=='syn':
                    plotted_syn.append(bundle.get_syn(objref=p['objref'],dataref=p['dataref']))
                    plotted_syn_ps.append(p)
        
        # setup tree view
        self.setColumnCount(1+len(system_names))
        self.setHeaderLabels(['Dataset']+['%s' % name for name in system_names])
        
        ### build rows
        for n,dataset in enumerate(data_obs):
            item = QTreeWidgetItem()
            item.info = {'dataset': dataset, 'widgets': []}
            self.addTopLevelItem(item)
            
            # create left (dataset) entry
            leftstackedwidget = QStackedWidget()
            
            # view mode
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(2)
            frame.setLayout(HBox) 
            
            label = QLabel(QString(str(dataset.get_value('ref'))+'          '))
            font = QFont()
            font.setBold(dataset.get_enabled() is True)
            #~ font.setItalic(par.get_qualifier() in self.data[col-1].constraints.keys())
            label.setFont(font)
            label.setMinimumSize(QSize(400,18))
            HBox.addWidget(label)
            
            if has_ydata(dataset):
                for key in ['l3','pblum']:
                    label = QLabel(key[:2] if key in dataset.keys() and dataset.get_adjust(key) else '')
                    label.setMinimumSize(QSize(18,18))
                    HBox.addWidget(label)
            
            leftstackedwidget.addWidget(frame)
            
            # edit mode
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(2)
            HBox.setMargin(0)
            frame.setLayout(HBox) 
            
            check = EnabledCheck(dataset.get_enabled() is True)
            check.setToolTip("toggle whether %s dataset is enabled for computing" % dataset['ref'])
            item.info['check'] = check
            HBox.addWidget(check)
            
            label = QLabel(QString(str(dataset.get_value('ref'))))
            HBox.addWidget(label)
            
            spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
            HBox.addItem(spacer)

            adjust_checks = []
            if has_ydata(dataset):
                for key in ['l3','pblum']:  # move these to the edit window?
                    if key in dataset.keys():
                        check = EnabledButton(dataset.get_adjust(key))
                        check.setText(key[:2])
                        check.key = key
                        check.setToolTip("toggle whether %s is marked for adjustment/fitting" % key)
                        adjust_checks.append(check)
                        HBox.addWidget(check)
            item.info['adjust_checks'] = adjust_checks
                    
            edit_button = QPushButton()
            edit_button = QPushButton()
            edit_button.setIcon(self.edit_icon)
            edit_button.setMaximumSize(QSize(18, 18))
            edit_button.setToolTip("edit %s dataset" % dataset['ref'])
            edit_button.setEnabled(False) #until signals attached
            HBox.addWidget(edit_button)

            if has_ydata(dataset):
                reload_button = QPushButton()
                reload_button.setIcon(self.reload_icon)
                reload_button.setMaximumSize(QSize(18, 18))
                reload_button.setToolTip("reload %s dataset" % dataset['ref'])
                #~ reload_button.setEnabled(False) #until signals attached
                reload_button.info = {'dataset': dataset}
                QObject.connect(reload_button, SIGNAL('clicked()'), self.on_reload_clicked)
                HBox.addWidget(reload_button)
            else:
                reload_button = None
            
            export_button = QPushButton()
            export_button.setIcon(self.list_icon)
            export_button.setMaximumSize(QSize(18, 18))
            export_button.setToolTip("export synthetic model to ascii")
            export_button.info = {'dataset': dataset}
            QObject.connect(export_button, SIGNAL('clicked()'), self.on_export_clicked)
            HBox.addWidget(export_button)
            
            delete_button = QPushButton()
            delete_button.setIcon(self.delete_icon)
            delete_button.setMaximumSize(QSize(18, 18))
            delete_button.setToolTip("remove %s dataset" % dataset['ref'])
            delete_button.info = {'dataset': dataset}
            QObject.connect(delete_button, SIGNAL('clicked()'), self.on_remove_clicked) 
            HBox.addWidget(delete_button)
            
            # store buttons on right so they can be shown for click on col 0 and hidden otherwise
            leftstackedwidget.info = {'buttons': [edit_button, reload_button, delete_button]}
            leftstackedwidget.addWidget(frame)           
            
            item.info['widgets'].append(leftstackedwidget)
            self.setItemWidget(item,0,leftstackedwidget)
            
            # build individual items (columns 1+)
            for i,name in enumerate(system_names):
                stackedwidget = QStackedWidget()
                
                # these two bools are used in the data view                
                has_obs = len(bundle.get_obs(name, dataset['ref'], all=True)) > 0
                #~ if has_obs:
                    # need to check that its not empty
                    #~ has_obs = has_ydata(bundle.get_obs(objref=name, dataref=dataset['ref'], return_type='all')[0])
                has_syn = len(bundle.get_syn(objref=name, dataref=dataset['ref'], all=True)) > 0
                
                # get the dataset for this row AND column
                if has_obs:
                    col_obs = bundle.get_obs(objref=name, dataref=dataset['ref'], all=True).values()[0]
                else:
                    col_obs = None
                if has_syn:
                    col_syn = bundle.get_syn(objref=name, dataref=dataset['ref'], all=True).values()[0]
                else:
                    col_syn = None
                
                # the following bools are used in the plots view
                #~ is_plotted_obs = dataset['ref'] in po_refs
                #~ is_plotted_syn = dataset['ref'] in ps_refs
                is_plotted_obs = col_obs in plotted_obs
                is_plotted_syn = col_syn in plotted_syn
                is_plotted = is_plotted_obs or is_plotted_syn
                
                # we want to know the names of all the plots that this dataset appears in
                plotted_names = []
                for ax in axes_incl:
                    for p in ax._get_dict_of_section('plot').values():
                        if ax.get_value('title') not in plotted_names and p['dataref']==dataset['ref']:
                            plotted_names.append(ax.get_value('title'))
                
                # and now we get the actually plotoptions parameter sets for the plots view
                if is_plotted_obs and self.style=='plot':
                    plotted_obs_ps_curr = plotted_obs_ps[plotted_obs.index(col_obs)]
                    plotted_obs_ind_curr = axes_incl[0].get_plot().values().index(plotted_obs_ps_curr)  # the index used in axes.get_plot()
                else:
                    plotted_obs_ind_curr = None
                    plotted_obs_ps_curr = None
                if is_plotted_syn and self.style=='plot':
                    plotted_syn_ps_curr = plotted_syn_ps[plotted_syn.index(col_syn)]
                    plotted_syn_ind_curr = axes_incl[0].get_plot().values().index(plotted_syn_ps_curr)
                else:
                    plotted_syn_ind_curr = None
                    plotted_syn_ps_curr = None
                
                # default view
                frame = QWidget()
                HBox = QHBoxLayout()
                HBox.setSpacing(2)
                HBox.setMargin(0)
                frame.setLayout(HBox)   
                
                #~ print "***", dataset.get_value('ref'), name, has_obs, has_syn
                             
                if has_obs:
                    #~ if has_obs and has_ydata(col_obs):
                    if has_obs:
                        # then data exists for this object and dataset
                        label = QLabel()
                        label.setPixmap(QPixmap(":/images/icons/database.png"))
                        label.setScaledContents(True)
                        label.setMaximumSize(QSize(18,18))
                        label.setMaximumSize(QSize(18,18))
                        label.setToolTip("%s includes data for %s" % (dataset.get_value('ref'),name))
                        HBox.addWidget(label)


                    #~ if has_syn:
                        # then a syn exists for this object and dataset
                        #~ label = QLabel("S")
                        #~ label.setPixmap(QPixmap(":/images/icons/database.png"))
                        #~ label.setScaledContents(True)
                        #~ label.setMaximumSize(QSize(18,18))
                        #~ label.setMaximumSize(QSize(18,18))
                        #~ label.setToolTip("%s includes synthetic data for %s" % (dataset.get_value('ref'),name))
                        #~ HBox.addWidget(label)

                    if True:
                        if self.style=='data':
                            if is_plotted:
                                label = QLabel()
                                label.setPixmap(QPixmap(":/images/icons/chart.png"))
                                label.setScaledContents(True)
                                label.setMaximumSize(QSize(18,18))
                                label.setToolTip("%s has plotoptions for %s" % (dataset.get_value('ref'),name))
                                HBox.addWidget(label)
                        if self.style=='plot':
                            if is_plotted_obs and has_ydata(col_obs):
                                label = QLabel()
                                label.setPixmap(QPixmap(":/images/icons/ellipsis.png"))
                                label.setScaledContents(True)
                                label.setMaximumSize(QSize(18,18))
                                label.setToolTip("%s has obs plotoptions for %s on %s" % (dataset.get_value('ref'),name,plots))
                                
                                if plotted_obs_ps_curr.get_value('color') != 'auto':
                                    colorize = QGraphicsColorizeEffect()
                                    color = QColor()                                
                                    color.setNamedColor(plotted_obs_ps_curr.get_value('color'))
                                    colorize.setColor(color)
                                    label.setGraphicsEffect(colorize)
                                if not plotted_obs_ps_curr.get_value('active'):
                                    #~ effect = QGraphicsBlurEffect()
                                    #~ effect.setBlurRadius(2)
                                    effect = QGraphicsOpacityEffect()
                                    effect.setOpacity(0.5)
                                    label.setGraphicsEffect(effect)
                                
                                HBox.addWidget(label)
                            
                            if is_plotted_syn:
                                label = QLabel()
                                label.setPixmap(QPixmap(":/images/icons/commit.png"))
                                label.setScaledContents(True)
                                label.setMaximumSize(QSize(18,18))
                                label.setToolTip("%s has syn plotoptions for %s on %s" % (dataset.get_value('ref'),name,plots))
                                
                                #~ if plotted_syn_ps_curr.get_value('color') != 'auto':
                                if True: # auto is set to red
                                    colorize = QGraphicsColorizeEffect()
                                    color = QColor()
                                    color.setNamedColor(plotted_syn_ps_curr.get_value('color') if plotted_syn_ps_curr.get_value('color') != 'auto' else '#FF0000')
                                    colorize.setColor(color)
                                    label.setGraphicsEffect(colorize)
                                if not plotted_syn_ps_curr.get_value('active'):
                                    #~ effect = QGraphicsBlurEffect()
                                    #~ effect.setBlurRadius(2)
                                    effect = QGraphicsOpacityEffect()
                                    effect.setOpacity(0.5)
                                    label.setGraphicsEffect(effect)
                                
                                HBox.addWidget(label)

                spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                HBox.addItem(spacer)
                    
                stackedwidget.addWidget(frame)
                
                # data view
                frame = QWidget()
                HBox = QHBoxLayout()
                HBox.setSpacing(2)
                HBox.setMargin(0)
                frame.setLayout(HBox)
                
                if has_obs:
                    if is_plotted:
                        # build button for each existing plot
                        for plotname in plotted_names:
                            plot_button = QPushButton()
                            plot_button.setIcon(self.plot_icon)
                            plot_button.setMaximumSize(QSize(18, 18))
                            plot_button.setToolTip(plotname)
                            plot_button.info = {'plotname': plotname}
                            QObject.connect(plot_button, SIGNAL("clicked()"), self.axes_goto)
                            HBox.addWidget(plot_button)
                    
                    # build button to create new axes
                    newplot_button = QPushButton()
                    newplot_button.setIcon(self.add_icon)
                    newplot_button.setMaximumSize(QSize(18, 18))
                    newplot_button.setToolTip("create new %s axes" % dataset.context[:-3])
                    newplot_button.info = {'category': dataset.context[:-3], 'datasetref': dataset['ref'], 'objref': name}
                    QObject.connect(newplot_button, SIGNAL("clicked()"), self.axes_add)
                    #~ newplot_button.setEnabled(False) # until signal connected
                    HBox.addWidget(newplot_button)
                    
                    spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                    HBox.addItem(spacer)
                    
                stackedwidget.addWidget(frame)
                
                # plot view
                frame = QWidget()
                HBox = QHBoxLayout()
                HBox.setSpacing(2)
                HBox.setMargin(0)
                frame.setLayout(HBox)
                
                if has_obs:
                    # obs
                    if has_ydata(col_obs):
                        obs_color = ColorChooserButton('obs',plotted_obs_ps_curr.get_value('color') if plotted_obs_ps_curr is not None else 'auto')
                        obs_marker = MarkerChooserCombo('obs',plotted_obs_ps_curr.get_value('marker') if plotted_obs_ps_curr is not None else 'auto')
                        obs_errorbars = EnabledCheck(plotted_obs_ps_curr.get_value('errorbars') if plotted_obs_ps_curr is not None else True)
                        obs_errorbars.setToolTip('show errorbars')
                        obs_toggle = PlotEnabledToggle('obs',plotted_obs_ps_curr.get_value('active') if plotted_obs_ps_curr is not None else False,[obs_color,obs_marker])
                        
                        HBox.addWidget(obs_toggle)
                        HBox.addWidget(obs_color)
                        HBox.addWidget(obs_errorbars)
                        HBox.addWidget(obs_marker)
                    else:
                        obs_color, obs_marker, obs_errorbars, obs_toggle = None, None, None, None
                    
                    # syn
                    syn_color = ColorChooserButton('syn',plotted_syn_ps_curr.get_value('color') if plotted_syn_ps_curr is not None else 'auto')
                    syn_marker = MarkerChooserCombo('syn',plotted_syn_ps_curr.get_value('linestyle') if plotted_syn_ps_curr is not None else 'auto')
                    syn_toggle = PlotEnabledToggle('syn',plotted_syn_ps_curr.get_value('active') if plotted_syn_ps_curr is not None else False,[syn_color,syn_marker])

                    HBox.addWidget(syn_toggle)
                    HBox.addWidget(syn_color)
                    HBox.addWidget(syn_marker)
                else:
                    obs_toggle, obs_color, obs_marker, obs_errorbars = None, None, None, None
                    syn_toggle, syn_color, syn_marker = None, None, None

                # add info to stacked widget so we can access and compare values
                stackedwidget.info = {}
                stackedwidget.info['itemname'] = name
                stackedwidget.info['dataset'] = dataset
                if len(axes_incl)>0: # this will fail if we have no axes created
                    stackedwidget.info['plotoptions'] = {axes_incl[0].get_value('title'): {plotted_obs_ind_curr if plotted_obs_ind_curr is not None else 'new_obs': {'color': obs_color, 'marker': obs_marker, 'errorbars': obs_errorbars, 'active': obs_toggle}, plotted_syn_ind_curr if plotted_syn_ind_curr is not None else 'new_syn': {'color': syn_color, 'linestyle': syn_marker, 'active': syn_toggle}}}
                
                spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                HBox.addItem(spacer)
                
                stackedwidget.addWidget(frame)
                
                item.info['widgets'].append(stackedwidget)
                self.setItemWidget(item,i+1,stackedwidget)
        
        for col in range(1,len(self.system_names)):
            self.resizeColumnToContents(col)
        self.resizeColumnToContents(0)


    def item_clicked(self, item, col):
        dataset = item.info['dataset']
        self.selected_item = (item,col)
        
        #~ if col == 0:
        leftstackedwidget = item.info['widgets'][0]
        leftstackedwidget.setCurrentIndex(1)
        
        #show buttons on right only if col 0 was clicked
        #else we'll just see the enabled check
        for button in leftstackedwidget.info['buttons']:
            if button is not None:
                button.setVisible(col==0)
        
        if col != 0:          
            obj = self.system_names[col-1]
            stackedwidget = item.info['widgets'][col]
            
            if self.style == 'data':
                stackedwidget.setCurrentIndex(1)
            else: # then plots
                stackedwidget.setCurrentIndex(2)
        
        
    def item_changed(self,change_value=True):
        if not self.selected_item:
            return

        item, col = self.selected_item            
        if change_value:
            # get stuff from col 0 but do it last
            #~ leftstackedwidget = item.info['widgets'][0]
            dataset = item.info['dataset']
            check = item.info['check']
            adjust_checks = item.info['adjust_checks']
            
            if col!=0 and self.style=='plot':
                # access stackedwidget.info
                stackedwidget = item.info['widgets'][col]
                plinfo = stackedwidget.info['plotoptions']
                dataset = stackedwidget.info['dataset']
                itemname = stackedwidget.info['itemname']
                #~ print "***", info

                already_created = []
                for axes_name in plinfo.keys():
                    for plot_no in plinfo[axes_name].keys():
                        for key in plinfo[axes_name][plot_no].keys():
                            widget = plinfo[axes_name][plot_no][key]
                            if widget is not None and widget.new_value != widget.orig_value:
                                #~ print "*", plot_no, isinstance(plot_no,int)
                                if isinstance(plot_no,int):
                                    do_command = "bundle.get_axes(\'%s\').get_plot(%d).set_value(\'%s\', \'%s\')" % (axes_name, plot_no, key, widget.new_value)
                                    undo_command = "bundle.get_axes(\'%s\').get_plot(%d).set_value(\'%s\', \'%s\')" % (axes_name, plot_no, key, widget.orig_value)
                                    description = 'change plotting option'
                                    
                                    self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                                
                                elif plot_no not in already_created: # need to create new plot
                                    new_plot_command = "type=\'%s\',dataref=\'%s\',objref=\'%s\'" % (dataset.context[:-3]+plot_no.split('_')[1],dataset.get_value('ref'),itemname)
                                    # get all the other new values, set them in one single command, and then skip this plot in future loops
                                    for nkey in plinfo[axes_name][plot_no].keys():
                                        nwidget = plinfo[axes_name][plot_no][nkey]
                                        if nwidget is not None and ((nkey!='active' and nwidget.new_value != nwidget.orig_value) or (nkey=='active' and nwidget.new_value==True)):
                                            new_plot_command = ",".join([new_plot_command,"%s=\'%s\'" % (nkey, nwidget.new_value)])
                                    do_command = "bundle.get_axes(\'%s\').add_plot(%s)" % (axes_name, new_plot_command)
                                    undo_command = "bundle.get_axes(\'%s\').remove_plot(%d)" % (axes_name, -1)
                                    description = 'add new plotting command'
                                    
                                    already_created.append(plot_no)
                                
                                    self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                                
            if check.new_value != check.orig_value:
                do_command = "bundle.%s_obs(\'%s\')" % ('enable' if check.new_value else 'disable', dataset['ref'])
                undo_command = "bundle.%s_obs(\'%s\')" % ('enable' if check.orig_value else 'disable', dataset['ref'])
                description = "%s %s dataset" % ('enable' if check.new_value else 'disable', dataset['ref'])
                self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                
            for check in adjust_checks:
                if check.new_value != check.orig_value:
                    do_command = "bundle.set_adjust('%s@%s@lcobs', %s)" % (check.key, dataset['ref'],check.new_value)
                    undo_command = "bundle.set_adjust('%s@%s@lcobs', %s)" % (check.key, dataset['ref'],check.orig_value)
                    #~ do_command = "bundle.adjust_obs(dataref=\'%s\',%s=%s)" % (dataset['ref'],check.key,check.new_value)
                    #~ undo_command = "bundle.adjust_obs(dataref=\'%s\',%s=%s)" % (dataset['ref'],check.key,check.orig_value)
                    description = "%s dataset changed adjust on %s" % (dataset['ref'],check.key)
                    self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                    #~ print "emitted signal"

        # this can probably be in else after system call is in place (will redraw treeview)
        leftstackedwidget = item.info['widgets'][0]
        leftstackedwidget.setCurrentIndex(0)
        
        stackedwidget = item.info['widgets'][col]
        stackedwidget.setCurrentIndex(0)
        self.selected_item = None
        
    def on_remove_clicked(self):
        ref = self.sender().info['dataset']['ref']
        
        do_command = "bundle.remove_data('%s')" % ref
        undo_command = "print 'undo is not available for this action'"
        # TODO - try to make this undoable
        description = "remove %s dataset" % ref
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
        
    def on_reload_clicked(self):
        ref = self.sender().info['dataset']['ref']
        
        do_command = "bundle.reload_obs('%s')" % ref
        undo_command = "print 'undo is not available for this action'"
        # TODO - try to make this undoable
        description = "reload %s dataset" % ref
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)        
        
    def on_export_clicked(self):
        ref = self.sender().info['dataset']['ref']
        
        # need popup file chooser
        latest_dir = self.myparent.myparent.latest_dir
        filename = QFileDialog.getSaveFileName(self, 'Save File', latest_dir if latest_dir is not None else './', **self.myparent.myparent._fileDialog_kwargs)
        
        if len(filename)>0:
            do_command = "bundle.write_syn(dataref='%s', output_file='%s')" % (ref, filename)
            undo_command = "print 'undo is not available for this action'"
            # TODO - try to make this undoable
            description = "export %s dataset" % ref
            self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
        
    def axes_add(self):
        info = self.sender().info
        
        self.emit(SIGNAL("axes_add"),info['category'],info['objref'],info['datasetref'])
    
    def axes_goto(self):
        info = self.sender().info
        self.emit(SIGNAL("axes_goto"),info['plotname'])
        
        
class PlotEnabledToggle(QPushButton):
    """
    toggle button to change whether a particular plotoptions is enabled/active
    toggle.orig_value and toggle.new_value are available for comparison and necessary system calls
    """
    
    def __init__(self,kind,state=None,children=[]):
        super(PlotEnabledToggle, self).__init__()
        
        self.setMaximumSize(QSize(18,18))
        self.setCheckable(True)
        
        obs_icon = QIcon()
        obs_icon.addPixmap(QPixmap(":/images/icons/ellipsis.png"), QIcon.Normal, QIcon.Off)
        
        syn_icon = QIcon()
        syn_icon.addPixmap(QPixmap(":/images/icons/commit.png"), QIcon.Normal, QIcon.Off)

        #children will hold all widgets that belong to this plot
        #when toggled, the children will be enabled/disabled
        self.children = children
        
        if kind=='obs':
            self.setToolTip('plot observables')
            self.setIcon(obs_icon)
        elif kind=='syn':
            self.setToolTip('plot synthetic')
            self.setIcon(syn_icon)

        if state is not None:
            # in this case if state is False, the signal will not originally be emitted
            # so we'll manually call on_toggled and connect the signal afterwards
            self.setChecked(state) 
            self.on_toggled(state,True)
        
        #connect the signal last
        QObject.connect(self, SIGNAL("toggled(bool)"), self.on_toggled)
        
    def setChildren(self,children):
        self.children = children
        
    def on_toggled(self,state,orig=False):
        
        if orig:
            self.orig_value = self.isChecked()
        self.new_value = self.isChecked()
        
        for child in self.children:
            child.setEnabled(state)

class EnabledCheck(QCheckBox):
    """
    check box to handle enabling/disabling dataset
    check.orig_value and check.new_value are available for comparison and necessary system calls
    """
    def __init__(self,enabled=False):
        super(EnabledCheck, self).__init__()
        
        self.setMaximumSize(QSize(18,18))
        
        self.setChecked(enabled)
        self.set_enabled(enabled,True)
        
        QObject.connect(self, SIGNAL("toggled(bool)"), self.set_enabled)
        
    def set_enabled(self,enabled,orig=False):
        if orig:
            self.orig_value = enabled
        self.new_value = enabled
        
class EnabledButton(QPushButton):
    """
    check box to handle enabling/disabling dataset
    check.orig_value and check.new_value are available for comparison and necessary system calls
    """
    def __init__(self,enabled=False):
        super(EnabledButton, self).__init__()
        
        self.setMaximumSize(QSize(18,18))
        
        self.setCheckable(True)
        
        self.setChecked(enabled)
        self.set_enabled(enabled,True)
        
        QObject.connect(self, SIGNAL("toggled(bool)"), self.set_enabled)
        
    def set_enabled(self,enabled,orig=False):
        if orig:
            self.orig_value = enabled
        self.new_value = enabled
    
class ColorChooserButton(QPushButton):
    """
    push button that when clicked will show a color dialog
    button.orig_value and button.new_value are available for comparison and necessary system calls
    """
    def __init__(self,kind,color=None):
        super(ColorChooserButton, self).__init__()
        
        self.setMaximumSize(QSize(24,18))
        QObject.connect(self, SIGNAL("clicked()"), self.on_clicked)
        
        if kind == 'obs':
            self.setToolTip("color for observables")
        elif kind=='syn':
            self.setToolTip("color for synthetic")
        
        if color is not None:
            self.set_color(color,True)
        
    def set_color(self,color,orig=False):
        if color != 'auto':
            self.setStyleSheet('QPushButton {background-color: %s}' % color)
        if orig:
            self.orig_value = color
        # either way, set new_value
        self.new_value = color
            
    def on_clicked(self,*args):
        initialcolor = QColor()
        initialcolor.setNamedColor(self.orig_value)
        dialog = QColorDialog()
        color = dialog.getColor(initialcolor)
        if color.isValid(): # will be false if canceled
            self.set_color(color.name()) # will also set self.new_color
            
class MarkerChooserCombo(QComboBox):
    """
    combo to choose marker/linestyle for plotoptions
    combo will automatically build with correct options based on kind
    marker.orig_value and marker.new_value are available for comparison and necessary system calls
    """
    def __init__(self,kind,marker=None):
        super(MarkerChooserCombo, self).__init__()
        
        self.setMaximumSize(QSize(70,18))
        self.setMinimumSize(QSize(56,18))
        
        if kind=='syn':
            self.addItems(['auto','-','--','-.',':'])
            self.setToolTip('linestyle for synthetic')
        elif kind=='obs':
            self.addItems(['auto','.',',','o','+','x','*','s','p','h','H','d','D','v','^','<','>','1','2','3','4'])
            self.setToolTip('marker for observables')
               
        if marker is not None:
            self.setCurrentIndex(self.findText(marker))
            self.set_marker(marker,True)
        
        QObject.connect(self, SIGNAL("currentIndexChanged(QString)"), self.set_marker)
            
    def set_marker(self,marker,orig=False):
        # this time we should have an entry for 'auto'
        # the widgets state will already be set, we just need to handle the orig_value and new_value
        if orig:
            self.orig_value = self.currentText()
        self.new_value = self.currentText()
        
        
class ParameterTreeWidget(GeneralParameterTreeWidget):
    """
    parameter tree view used for:
        components
        orbits
        meshes
        compute options
        fitting options
        plot orbit options
        plot mesh options
    """
     
    def set_data(self,data,style=[],hide_params=[]):
        '''
        data - list of ParameterSets to be shown in the treeview
        
        style is a list of rules to apply from the following
            'nofit' to never have adjust checkboxes
            'noadd' to disable adding a new parameter at the bottom
        '''
        self.clear()
        self.items = OrderedDict()
        self.style=style
        self.data=data #to access parametersets
        
        try:
            nobj = len(data)
            ncol = nobj + 1
        except TypeError:
            self.setHeaderLabels(['Parameter'])
            return
        
        self.setColumnCount(ncol)
        self.setHeaderLabels(['Parameter']+['Value']*nobj)
        
        for n,params in enumerate(data):
            for k,v in params.items():
                if ('label' not in k or 'incl_label' in self.style) and k not in ['feedback'] and k not in hide_params:
                    par = params.get_parameter(k)
                    adj = params.get_adjust(k)
                    
                    try:
                        self.items[k].append([v,par,adj])
                    except KeyError:
                        if n>0:
                            self.items[k]=[['',''] for i in range(n)]
                            self.items[k].append([v,par,adj])
                        else:
                            self.items[k]=[[v,par,adj]]
        
        if 'incl_label' in self.style and 'label' in self.items.keys():
            # make sure label is first
            labelitem = self.items.pop('label')
            iteritems = [('label',labelitem)] + self.items.items()
            # add back to self.items
            self.items['label'] = labelitem
        else:
            iteritems = self.items.items() 
        
        for k,v in iteritems:
            item = QTreeWidgetItem()
            try:
                item.setToolTip(0,v[0][1].get_description())
            except AttributeError:
                item.setToolTip(0,'no description...')
            self.addTopLevelItem(item)
            
            #create frame and labels for col 0
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(0)
            frame.setLayout(HBox)

            infobutton = QPushButton()
            infobutton.setToolTip('get info about %s' % k)
            infobutton.setIcon(self.info_icon)
            infobutton.setIconSize(QSize(12, 12))
            infobutton.info = {'param': v[0][1]}
            infobutton.setMaximumSize(QSize(18,18))
            QObject.connect(infobutton, SIGNAL("clicked()"), self.on_infobutton_clicked)
            infobutton.setVisible(False)
            HBox.addWidget(infobutton)

            label = QLabel(QString(k))
            HBox.addWidget(label)
            self.setItemWidget(item,0,frame)
            
            item.info = {'isparam': True, 'paramname': k, 'infobutton': infobutton} #access par through self.getItem(item.info['paramname'], col)
            
            for col in range(1,nobj+1):
                self.createLabel(item,col)
                
        if 'add' in self.style:
            item = QTreeWidgetItem()
            self.addTopLevelItem(item)
            
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(0)
            frame.setLayout(HBox)
            addlabel = QLabel('add...')
            addlabel.setToolTip('add new parameter')
            HBox.addWidget(addlabel)
            addparam = QPushButton('add...')
            addparam.setToolTip('add new parameter')
            addparam.setMaximumSize(QSize(300,18))
            addparam.setEnabled(False)
            addparam.setVisible(False)
            HBox.addWidget(addparam)
            
            item.info = {'isparam': False, 'label': addlabel, 'button': addparam}
            self.setItemWidget(item,0,frame)
                
        for i in range(ncol):
            self.resizeColumnToContents(i)
            
    def getItem(self, itemname, col):
        for k,v in self.items.items():
            if k==itemname:
                if len(v) >= col:
                    return v[col-1][1]
                else:
                    return None
        return None

    def createLabel(self, item, col):
        frame = QWidget()
        HBox = QHBoxLayout()
        HBox.setSpacing(0)
        HBox.setMargin(0)
        frame.setLayout(HBox)
        par = self.getItem(item.info['paramname'], col)
        if hasattr(par,'qualifier'): # need this for the cases where a parameter exists in one column but not another
            value = par.get_value()
            if par.cast_type=='return_string_or_list' and not isinstance(value, str) and len(str(value)) > 4:
                value = 'list'
            label = QLabel(QString(str(value)))
            font = QFont()
            font.setBold(par.get_adjust() is True)
            font.setItalic(par.get_qualifier() in self.data[col-1].constraints.keys())
            label.setFont(font)
            HBox.addWidget(label)
        self.setItemWidget(item,col,frame)
        
    def createFloatWidget(self,par,value,typ=float,widget=None):
        lims = par.get_limits()
        step = par.get_step()
        if widget is None:
            if typ==int:
                widget=QSpinBox()
            else:
                widget=QDoubleSpinBox()
            widget.setMaximumSize(QSize(200,18))
        if lims != (None, None):
            widget.setRange(*lims)
        else:
            widget.setRange(0,1E6)
        if step is not None:
            widget.setSingleStep(step)
        if typ==int:
            widget.setValue(int(value))
        else:
            currentdecimals = len(value.split('.')[1])
            widget.setDecimals(5 if currentdecimals<=5 else currentdecimals)
            widget.setValue(float(value))

        return widget

    def item_clicked(self, item, col):
        if self.selected_item is not None and item == self.selected_item[0]: #then we haven't changed items
            return
        if not item.info['isparam']: #then is the add... row
            item.info['label'].setVisible(False)
            item.info['button'].setVisible(True)
            self.selected_item = None
            return
            
        item.info['infobutton'].setVisible(True)
        self.selected_item = (item,col) #will override later if not col 0
            
        if col>0:
            
            currenttext = str(self.itemWidget(item,col).children()[1].text())
            
            par = self.getItem(item.info['paramname'], col)
            try:
                typ = par.cast_type
            except AttributeError:
                #cell does not hold a phoebe parameter
                return
                
            frame = QWidget()
            HBox = QHBoxLayout()
            HBox.setSpacing(0)
            HBox.setMargin(0)
            frame.setLayout(HBox)
            if typ==float and 'nofit' not in self.style and hasattr(par, 'adjust'): # only floats are fitable
                is_adjust = par.get_adjust()
                check = QCheckBox()
                check.setMaximumSize(QSize(18, 18))
                check.setCheckState(2 if is_adjust else 0)
                check.info = {'originalvalue': is_adjust}
                if is_adjust:
                    check.setToolTip('unmark for adjustment/fitting')
                else:
                    check.setToolTip('mark for adjustment/fitting')
                    
                HBox.addWidget(check)
            else:
                check = None #for consistency in the length of self.selected_item

            if typ==float or typ==int:
                widget = self.createFloatWidget(par,currenttext,typ=typ)
                HBox.addWidget(widget)
                
                if par.qualifier in self.data[col-1].constraints.keys(): #par.has_constraint()
                    widget.setEnabled(False)
                    #should check be unchecked and disabled? can you fit a constrained parameter?
                    
                    # removing link for now - its behavior seems somewhat inconsistent
                    #~ link=QPushButton()
                    #~ link.setCheckable(True)
                    #~ link.setChecked(True)
                    #~ link.setToolTip('remove constraint')
                    #~ link.setIcon(self.link_icon)
                    #~ link.setIconSize(QSize(12,12))
                    #~ link.setMaximumSize(QSize(18,18))
                    #~ QObject.connect(link, SIGNAL("toggled(bool)"), self.on_link_toggled)
                    #~ HBox.addWidget(link)


                #~ if par.has_unit():
                if True:
                    button=QPushButton()
                    button.setToolTip('edit parameter')
                    button.setIcon(self.edit_icon)
                    button.setIconSize(QSize(12, 12))
                    button.info = {'param': par, 'widget': widget, 'check': check}
                    button.setMaximumSize(QSize(18,18))
                    QObject.connect(button, SIGNAL("clicked()"), self.on_editbutton_clicked)
                    HBox.addWidget(button)
                    
                #~ spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                #~ HBox.addItem(spacer)

                self.setItemWidget(item,col,frame)
                self.selected_item = (item,col,widget,check,currenttext)
                
            elif typ==str or typ=='return_string_or_list':
                #don't use currenttext here because sometimes we override that for the label
                widget=QLineEdit(str(par.get_value()))
                widget.setMaximumSize(QSize(2000,18))
                HBox.addWidget(widget)

                if typ=='return_string_or_list':
                    button=QPushButton()
                    button.setIcon(self.edit_icon)
                    button.setIconSize(QSize(12, 12))
                    button.info = {'param': par, 'widget': widget, 'check': None}
                    button.setMaximumSize(QSize(18,18))
                    QObject.connect(button, SIGNAL("clicked()"), self.on_editbutton_clicked)
                    if par.get_qualifier() in ['refs','types','samprate']:
                        button.setEnabled(False) # until implemented
                    HBox.addWidget(button)
                
                #~ spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                #~ HBox.addItem(spacer)

                self.setItemWidget(item,col,frame)
                widget.setFocus()
                
                self.selected_item = (item,col,widget,check,currenttext)
                
            elif typ=='make_bool' or typ==bool:
                widget=QComboBox()
                widget.setMaximumSize(QSize(2000,18))
                widget.addItem('True')
                widget.addItem('False')
                widget.setCurrentIndex(widget.findText(currenttext))
                HBox.addWidget(widget)
                #~ spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                #~ HBox.addItem(spacer)
                self.setItemWidget(item,col,frame)
                #~ widget.showPopup() # disabled because of check
                self.selected_item = (item,col,widget,check,currenttext)
                
            elif typ=='choose':
                choices = par.choices
                widget=QComboBox()
                widget.setMaximumSize(QSize(2000,18))
                for c in choices:
                    widget.addItem(c)

                widget.setCurrentIndex(widget.findText(currenttext))
                HBox.addWidget(widget)
                #~ spacer = QSpacerItem(0, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)
                #~ HBox.addItem(spacer)
                self.setItemWidget(item,col,frame)
                #~ widget.showPopup() # disabled because of check
                self.selected_item = (item,col,widget,check,currenttext)
                
            #~ self.resizeColumnToContents(col)
                   
    def item_changed(self,change_value=True):
        
        if self.selected_item:
            self.selected_item[0].info['infobutton'].setVisible(False)
            if self.selected_item[1]==0: #then first column and we can exit
                self.selected_item = None
                return

            # get this info first since access to the widget will disappear after emitting a signal
            if len(self.selected_item) <= 2: return
            item=self.selected_item[0]
            col=int(self.selected_item[1])
            widget=self.selected_item[2]
            check=self.selected_item[3]
            currenttext=self.selected_item[4]
            
            className=widget.metaObject().className()
            label=str(self.headerItem().text(col))
            paramname=item.info['paramname']
            param = self.getItem(item.info['paramname'], col)
            old_value=currenttext
            
            self.removeItemWidget(item,col)
            self.createLabel(item,col)
            
            self.selected_item=None
            if not change_value:
                return

            # get check state only if type is float (fitable)
            # again, doing this now because it will disappear after emitting a signal
            if className=='QDoubleSpinBox' and 'nofit' not in self.style and hasattr(param,'adjust'): # only floats are fitable
                new_check = True if check.checkState()==2 else False
                old_check = check.info['originalvalue']
            
            elif className=='QDoubleSpinBox' or className=='QSpinBox':
                new_value=str(widget.value())
                if new_value!=old_value:
                    item.setData(col,0,new_value)
                    self.emit(SIGNAL("parameterChanged"),self,label,param,old_value,new_value)
                
            elif className=='QLineEdit':
                new_value=widget.text()
                if new_value and new_value!=old_value:
                    item.setData(col,0,new_value)
                    self.emit(SIGNAL("parameterChanged"),self,label,param,old_value,new_value)
                    
            elif className=='QComboBox':
                new_value=widget.currentText()
                if new_value!=old_value:
                    item.setData(col,0,new_value)
                    self.emit(SIGNAL("parameterChanged"),self,label,param,old_value,new_value)
            
            elif className=='QDoubleSpinBox' and 'nofit' not in self.style and hasattr(param,'adjust'): #only floats are fitable
                if new_check!=old_check: 
                    self.emit(SIGNAL("parameterChanged"),self,label,param,old_check,new_check,None,None,True)
            
            return
            
    def on_infobutton_clicked(self):
        button = self.sender()
        param = button.info['param'] #note: we don't know the column
        
        text="Description: %s" % (param.get_long_description(width=50,initial_indent="            |",subsequent_indent='     ',force=True).split('|')[1])
        
        if param.cast_type == float:
            if param.has_unit():
                text+="\n\nDefault Unit: %s\n" % param.get_unit()
            else:
                text+="\n\n"
            text+="Lower Limit: %s\nUpper Limit: %s" % (param.get_limits()[0], param.get_limits()[1])
        
        text+="\n"
        
        QMessageBox.information(None, "PHOEBE - Parameter Info: %s" % param.get_qualifier(), text)         
        
        #this could also include a link button to the documentation on the parameter

    def on_editbutton_clicked(self):
        tools_constraints = {}
        tools_constraints['sma'] = [['add_asini', '{asini}/np.sin({incl})']]
        tools_constraints['incl'] = [['add_asini', 'np.arcsin({asini}/{sma})']]
        #~ tools_constraints['rotperiod'] = [['add_rotfreqcrit', '2*np.pi*np.sqrt(27*{radius}**3/(8*constants.GG*{mass}))/{rotfreqcrit}']]
        tools_constraints['teff'] = [['add_teffpolar', '{teffpolar}']]
        tools_constraints['mass'] = [['add_surfgrav', '{surfgrav}/constants.GG*{radius}**2']]
        tools_constraints['radius'] = [['add_surfgrav', 'np.sqrt((constants.GG*{mass})/{surfgrav})']]
        #~ tools_constraints['rotperiod'] += [['add_vsini', '(2*np.pi*{radius})/{vsini}*np.sin({incl})']]
        #~ tools_constraints['incl'] += [['add_vsini', 'np.arcsin({rotperiod}*{vsini}/(2*np.pi*{radius}))']]
        #~ tools_constraints['radius'] += [['add_vsini', '({rotperiod}*{vsini})/(2*constants.pi*np.sin({incl}))']]
        
        
        button = self.sender()
        par = button.info['param']
        widget = button.info['widget']
        check = button.info['check']
        col = int(self.selected_item[1])
        parset = self.data[col-1]
        label=self.headerItem().text(int(self.selected_item[1]))

        if par.cast_type != float and par.cast_type != 'return_string_or_list':
            return

        pop = CreatePopParamEdit()
                
        pop.kindLabel.setText("%s:" % par.get_context().title())
        pop.objectLabel.setText(self.headerItem().text(int(self.selected_item[1])))
        pop.parameterLabel.setText(par.get_qualifier())
        pop.descriptionLabel.setText(par.get_description())
        
        if parset.get_context().split(':')[0] not in ['compute','fitting']:
            pop.defaults_frame.setVisible(False)
        
        if not hasattr(par, 'long_description'):
            pop.descriptionLongToggle.setVisible(False)
        else:
            pop.descriptionLongLabel.setText(par.get_long_description())
        pop.descriptionLongLabel.setVisible(False)
                
        if par.cast_type == float:
            pop.stackedWidget.setCurrentIndex(0)
            
            self.editdialog_setvalues(pop, par, widget)
            
            QObject.connect(pop.float_infobutton, SIGNAL("clicked()"), self.on_infobutton_clicked)
            pop.float_infobutton.info = {'param': par}

            pop.constraintText.info = {'pop': pop}
            QObject.connect(pop.constraintText, SIGNAL("textChanged(QString)"), self.on_constraint_changed)
            
            pop.constraintQual.setText("{%s} = " % par.get_qualifier())
            
            pop.constraintReset.info = {'constraint': [None,''], 'pop': pop}
            QObject.connect(pop.constraintReset, SIGNAL("clicked()"), self.on_preset_clicked)
            
            if par.get_qualifier() in tools_constraints.keys():
                constraints = tools_constraints[par.get_qualifier()]
                HBox = QHBoxLayout()
                HBox.setMargin(0)
                for i,constraint in enumerate(constraints):
                    button = QPushButton(constraint[0])
                    button.info = {'constraint': constraint, 'pop': pop}
                    HBox.addWidget(button)
                    QObject.connect(button, SIGNAL("clicked()"), self.on_preset_clicked)
                
                pop.presetWidget.setLayout(HBox)
            else:
                pop.presetLabel.setVisible(False)
                pop.presetHelp.setVisible(False)
                
            if par.get_qualifier() in parset.constraints.keys():
                constraint = parset.constraints[par.get_qualifier()]
                pop.info = {'orig_constraint': [None, constraint]} #to check to see if changed when ok clicked
                pop.constraintText.setText(constraint)
            else:
                pop.info = {'orig_constraint': [None, '']}
            pop.info['constraint'] = pop.info['orig_constraint']
            
            pop.constraintHelp.info = {'helpTitle': 'PHOEBE - constraint help','helpText': 'Constrain this parameter using other parameters in the same parameter set.  Refer to parameter names as {qualifier}.\nAs long as the constraint is in place, the value will be computed automatically, and you will not be able to manually change the value of this parameter.\n\nNOTE: these must always be in SI units.'}
            pop.presetHelp.info = {'helpTitle': 'PHOEBE - constraint preset help','helpText': 'These presets will add a new parameter and also create the constraint.'}
            QObject.connect(pop.constraintHelp, SIGNAL("clicked()"), self.on_help_clicked)
            QObject.connect(pop.presetHelp, SIGNAL("clicked()"), self.on_help_clicked)
        
            if hasattr(par, 'adjust') and 'nofit' not in self.style:
                pop.check.setCheckState(check.checkState())
            else:
                pop.check.setVisible(False)
        
            pop.float_resetbutton.setVisible(False) #not consistent behavior, so removed for now
            #~ pop.float_resetbutton.info = {'param': par, 'pop': pop}
            #~ QObject.connect(pop.float_resetbutton, SIGNAL("clicked()"), self.on_reset_clicked)

            result = pop.exec_()
            
            if result:
                old_value=float(self.selected_item[4])
                new_value=pop.float_value.value()

                if str(pop.constraintText.text()) != pop.info['orig_constraint'][1]:
                    # check to see if one of the presets
                    function = pop.info['constraint'][0]
                    if function:
                        if function == 'add_asini':
                            value = parset.get_value('sma')*np.sin(np.pi/180. * parset.get_value('incl'))  #note that these don't come from the widget, but the current set value
                        elif function == 'add_teffpolar':
                            value = parset.get_value('teff')
                        elif function == 'add_surfgrav':
                            value = constants.GG*parset.get_value('mass')/parset.get_value('radius')**2
                        else:
                            print("WARNING: function %s not implemented yet" % function)
                            return
                        do_command = 'tools.%s(bundle.get_%s(\'%s\'),%f,derive=\'%s\')' % (function,par.get_context()[0],label,value,par.get_qualifier())
                        undo_command = 'bundle.get_%s(\'%s\').remove_constraint(\'%s\')' % (par.get_context()[0],label,par.get_qualifier())
                        description = "add \'%s\' preset constraint to %s" % (function,label)
                        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
                    else:
                        self.emit(SIGNAL("parameterChanged"),self,label,par,pop.info['orig_constraint'][1],str(pop.constraintText.text()),None,None,False,True)
                
                if pop.float_value.isEnabled():
                    if old_value != new_value:
                        self.emit(SIGNAL("parameterChanged"),self,label,par,old_value,new_value,None,None)
                        
                new_check = True if pop.check.checkState() == 2 else False
                old_check = par.get_adjust()
                if new_check != old_check:
                    self.emit(SIGNAL("parameterChanged"),self,label,par,old_check,new_check,None,None,True)
                
                
                # force to unfocus item
                self.item_changed(False)


        elif par.cast_type == 'return_string_or_list':
            if par.get_qualifier() in ['time','times']:
                index = 2
                if par.get_value()=='auto':
                    pop.strlist_time_auto.setChecked(True)
                else:
                    pop.strlist_time_custom.setChecked(True)
                    pop.strlist_time_custom_text.setText(str(list(par.get_value())))
            elif par.get_qualifier() in ['refs','types']:
                index = 3
                pop.strlist_refstypes_auto.setChecked(True) #will eventually be correct based on parameter
            else:
                index = 1
                
            pop.stackedWidget.setCurrentIndex(index)
            
            result = pop.exec_()

            if result:
                if index==2:
                    #then dealing with time
                    old_value = par.get_value()
                    if old_value != 'auto':
                        old_value = list(old_value)
                    is_auto = pop.strlist_time_auto.isChecked()
                    if is_auto and par.get_value()!='auto':
                        new_value = 'auto'
                    elif pop.strlist_time_linspace.isChecked():
                        new_value = 'np.linspace(%f,%f,%f)' % (pop.strlist_linspace_min.value(), pop.strlist_linspace_max.value(), pop.strlist_linspace_num.value())
                    else: # assume custom
                        new_value = pop.strlist_time_custom_text.text()
                    
                    if str(old_value) != str(new_value):
                        self.emit(SIGNAL("parameterChanged"),self,label,par,old_value,new_value,None,None,False)
                         
                else:
                    print("WARNING: not implemented yet")
                    
                # force to unfocus item
                self.item_changed(False)


            
    def editdialog_setvalues(self, pop, par, widget=None):
        value = par.get_value() if widget is None else widget.value()
        pop.float_value = self.createFloatWidget(par,str(value),widget=pop.float_value)

        if par.has_unit():
            #~ pop.float_units.addItems(par.list_available_units()[1])
            #~ pop.float_units.setCurrentIndex(pop.float_units.findText(par.get_unit()))
            pop.float_units.setText(par.get_unit())
        else:
            pop.float_units.setVisible(False)
            pop.float_infobutton.setVisible(False)
            
    def on_link_toggled(self):
        link = self.sender()
        item = self.selected_item[0]
        col = int(self.selected_item[1])
        
        label = self.headerItem().text(int(self.selected_item[1]))
        par = self.getItem(item.info['paramname'], col)
        parset = self.data[col-1]
        constrainttext = parset.constraints[par.get_qualifier()]
        
        do_command = 'bundle.get_%s(\'%s\').remove_constraint(\'%s\')' % (par.get_context()[0],label,par.get_qualifier())
        undo_command = 'bundle.get_%s(\'%s\').add_constraint(\'{%s} = %s\')' % (par.get_context()[0],label,par.get_qualifier(),constrainttext)
        description = "remove constraint on %s:%s" % (label,par.get_qualifier())
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
        
        #since this is only visible when there is a constraint, its guaranteed to be removing one if toggled
        self.item_changed(False) #immediately apply and redraw
        
    def on_reset_clicked(self):
        button = self.sender()
        param = button.info['param']
        pop = button.info['pop']
        label = self.headerItem().text(int(self.selected_item[1]))
        
        do_command = 'bundle.get_%s(\'%s\').reset(\'%s\')' % (param.get_context()[0],label,param.get_qualifier())
        undo_command = 'bundle.get_%s(\'%s\').set_value(\'%s\',\'%s\')' % (param.get_context()[0],label,param.get_qualifier(),param.get_value())  #won't reset units 
        description = 'reset parameter %s:%s' % (label,param.get_qualifier())
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description)
        
        self.editdialog_setvalues(pop, param)
        
    def on_preset_clicked(self):
        button = self.sender()
        constraint = button.info['constraint']
        pop = button.info['pop']
        
        pop.constraintText.setText(constraint[1])
        
        if constraint[0] is not None:
            pop.constraintText.setEnabled(False)
        else:
            pop.constraintText.setEnabled(True)
            
        pop.info['constraint'] = constraint
        
        #change this part to be in textchanged so it constantly updates
        #~ if constraint[1].strip() == '':
            #~ pop.float_value.setEnabled(True)
        #~ else:
            #~ pop.float_value.setEnabled(False)
            
    def on_constraint_changed(self,text):
        pop = self.sender().info['pop']
        
        if str(text).strip() == '':
            pop.float_value.setEnabled(True)
        else:
            pop.float_value.setEnabled(False)
        
            
    def on_help_clicked(self):
        button = self.sender()
        
        QMessageBox.information(None, button.info['helpTitle'], button.info['helpText'])  
        
########################################################################


class AddPlotHoverLabelWidget(QWidget):
    def __init__(self, parent = None):
        super(AddPlotHoverLabelWidget, self).__init__(parent)

    def enterEvent(self, event):
        super(AddPlotHoverLabelWidget, self).enterEvent(event)
        self.emit(SIGNAL("hover_enter"))

    def leaveEvent(self, event):
        super(AddPlotHoverLabelWidget, self).leaveEvent(event)
        self.emit(SIGNAL("hover_leave"))
        
class SystemTreeWidget(QTreeWidget):
    '''
    custom widget to show the tree structure of the system
    and handles sending signals when the selection changes
    '''
    def __init__(self, parent = None):
        super(SystemTreeWidget, self).__init__(parent)

    def mouseReleaseEvent(self, event):
        super(SystemTreeWidget, self).mouseReleaseEvent(event)
        # using native signals for changed selection, we get the signal before the drag action is completed
        self.emit(SIGNAL("selectionChanged")) #note: the new method of declaring and emitting a signal was laggy for some reason

    def keyPressEvent(self, event):
        super(SystemTreeWidget, self).keyPressEvent(event)
        if event.key() in [Qt.Key_Up, Qt.Key_Down]:
            self.emit(SIGNAL("selectionChanged"))
    
    def expand_all(self, treeitems):
        for item in treeitems:
            self.expandItem(item)


class SchematicItem(QGraphicsEllipseItem):
    def __init__(self, parent = None):
        super(SchematicItem, self).__init__(parent)
        self.setFlag(self.ItemIsSelectable, True)

        self.children = 0
        #~ self.setFlag(self.ItemIsMovable, True)

    def highlight(self):  # not implemented yet
        if self._isOrbit:
            self.setPen(QPen(QColor(0,0,0),3))
        else:
            self.setPen(QPen(QColor(0,0,0),2))
            
    def _getShape(self):
        return self.rect().center().x(), self.rect().center().y(), self.rect().width()/2.

    def _determineShape(self, parent, pos):
        cx, cy, r = 0, 0, 50
        if parent is not None:
            if pos is None:
                pos = parent.children + 1
            pcx, pcy, pr = parent._getShape()
            if pos == 1:
                cx = pcx - pr
                cy = pcy
            if pos == 2:
                cx = pcx + pr
                cy = pcy
            if pos == 3:
                cx = pcx
                cy = pcy - pr
            if pos == 4:
                cx = pcx
                py = pcy + pr
            r = 3./4 * pr
        return cx, cy, r

    def setObject(self, kind="orbit", parent=None, pos=None):
        cx, cy, r = self._determineShape(parent, pos)
        if kind=='star':
            rx, ry = 10, 10
        if kind=='rochestar':
            rx, ry = 12, 6
        if kind=='orbit':
            rx, ry = r, r
            self.setRect(cx-r, cy-r, r*2, r*2)
            self.setPen(QPen(QColor(0,0,0),2))
        else:
            self.setRect(cx-rx, cy-ry, rx*2, ry*2)
            self.setPen(QPen(QColor(0,0,0),1))
            self.setBrush(QBrush(QColor(100,100,100)))

        if parent is not None:
            parent.children += 1
            
class SysWebView(QWebView):
    def __init__(self):
        super(SysWebView, self).__init__()
        
    def contextMenuEvent(self,*args):
        #disable contextmenu (with reload option)
        pass
        
    def keyPressEvent(self,event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers==Qt.ControlModifier:
            self.emit(SIGNAL("ctrl_pressed"))
            
    def keyReleaseEvent(self,event):
        self.emit(SIGNAL("ctrl_released"))
            

class JavaScriptMessenger(QObject):  
    def __init__(self):
        super(JavaScriptMessenger, self).__init__()
        self.sysitems = []
        self.sysitems_flat = []
        self.sysitems_ps = []
        self.sysitems_nchild = []
        self.sysitems_sel = []
        self.sysitems_sel_flat = [] #this is what will be sent from JQuery (compare to list(utils.traverse(self.sysitems_ps)))
        self.ctrl = False

    @pyqtSlot(str)  
    def printToTerm(self, msg):
        #~ pass
        print(msg)

    @pyqtSlot(str)
    def showMessage(self, msg):  
        """Open a message box and display the specified message."""  
        QMessageBox.information(None, "Info", msg)  
    
    @pyqtSlot(str)
    def sendSignal(self, signal="signal"):
        self.emit(SIGNAL(signal))
        
    @pyqtSlot(str)
    def updateSelected(self, selected):
        self.sysitems_sel_flat = json.loads(str(selected))
        self.emit(SIGNAL("selectionUpdate"))

    def _get_sysitems(self):
        return json.dumps(self.sysitems)
    
    def _get_sysitems_nchild(self):
        return json.dumps(self.sysitems_nchild)
        
    def _get_sysitems_sel(self):
        return json.dumps(self.sysitems_sel)
        
    def _get_ctrl(self):
        return str(self.ctrl)
        
    def _get_test(self):
        return 'test'

    """Python interpreter version property."""  
    get_sysitems = pyqtProperty(str, fget=_get_sysitems)
    get_sysitems_nchild = pyqtProperty(str, fget=_get_sysitems_nchild)
    get_sysitems_sel = pyqtProperty(str, fget=_get_sysitems_sel)
    get_ctrl = pyqtProperty(str, fget=_get_ctrl)
    get_test = pyqtProperty(str, fget=_get_test)
    
            
class CommandRun(QUndoCommand):
    def __init__(self,PythonEdit,redocommand,undocommand,write=True,thread=False,kind=None,description=None):
        super(CommandRun, self).__init__(description)
        self.PythonEdit = PythonEdit
        self.redocommand = redocommand
        self.undocommand = undocommand
        self.write = write
        self.thread = thread
        self.kind = kind
        
        self.skip_first = False # skip running the redo command on first call
        
    def redo(self):
        if not self.skip_first:
            self.run(self.redocommand,self.write,self.thread,self.kind)
        self.skip_first = False
        
    def undo(self):
        self.run(self.undocommand,self.write,self.thread,self.kind)
        
    def run(self,command,write,thread,kind):
        if kind=='sys':
            write = self.PythonEdit.write_sys
        if kind=='plots':
            write = self.PythonEdit.write_plots
        if kind=='settings':
            write = self.PythonEdit.write_settings
        if write:
            PyInterp.write(self.PythonEdit, command+"\n")
        PyInterp.run(self.PythonEdit, command, write=write, thread=thread)
        if write:
            self.PythonEdit.history.insert(0, command)
        if write and not thread:
            PyInterp.marker(self.PythonEdit)

class StdBuffer(QObject):
    def write(self, text):
        self.emit(SIGNAL('newtext'), text)

class PyInterpThread(QThread):
    buff = StdBuffer() #create buffer which will communicate stdout through signals
    def __init__(self,parent,command,debug=[False, False]):
        QThread.__init__(self)
        self.parent = parent
        self.command = command
        self.comm = {} #to send locals back to PyInterp

        self.debug = debug

        if not debug[0]:
            self.stdoutbkp = sys.stdout
            sys.stdout = self.buff
        if not debug[1]:
            self.stderrbkp = sys.stderr        
            sys.stderr = self.buff
            
        #~ if 'bundle' in self.parent.comm.keys():
            #~ bundle = self.parent.comm['bundle']
            #~ bundle.attach_signal(bundle.get_system(), 'set_time', self.on_set_time)
            
    def run(self):
        # attach signal each time a command is run, and purge afterwards
        # this allows us to still safely pickle or deepcopy the system
        if 'bundle' in self.parent.comm.keys():
            bundle = self.parent.comm['bundle']
            if '.add_version' not in self.command:
                # don't attach signal if we know we need to pickle or deepcopy during this command
                bundle.attach_signal(bundle.get_system(), 'set_time', self.on_set_time)
                

        try:
            code = compile(self.command, '<string>', 'single')
            exec(code)
        except Exception as message:
            self.message = str(message)
            self.emit(SIGNAL('threadFailed'))
        else:
            self.emit(SIGNAL('threadComplete'),self.command)
        
        for key in locals().keys():
            self.comm[key] = locals()[key]
            
        # restore stdout and stderr as they were
        if not self.debug[0]:
            sys.stdout = self.stdoutbkp
        if not self.debug[1]:
            sys.stderr = self.stderrbkp
        
        # remove signal if attached    
        if 'bundle' in self.parent.comm.keys():
            bundle = self.parent.comm['bundle']
            if len(bundle.attached_signals) > 0:
                #~ print "*** before", bundle.attached_signals, bundle.get_system().signals
                bundle.purge_signals([bundle.get_system()]) 
                # just remove the set_time signal
                # note: this will still result in duplicates in bundle.attached_signals
                # but should be cleared soon enough through an entire purge
                
                #~ bundle.get_system().signals = {}
                #~ print "*** after", bundle.attached_signals, bundle.get_system().signals
 
            #~ callbacks.purge_signals(bundle.get_system())
            #~ bundle.get_system().signals={}
            
    def on_set_time(self,*args):
        self.emit(SIGNAL('set_time'))
        
class PyInterp(QTextEdit):
    # modified greatly from https://github.com/JeffMGreg/PyInterp
    class InteractiveInterpreter(code.InteractiveInterpreter):
        def __init__(self, locals):
            code.InteractiveInterpreter.__init__(self, locals)

        def runIt(self, command):
            code.InteractiveInterpreter.runsource(self, command)
            #~ code.InteractiveInterpreter.showsyntaxerror(command)
            
    def __init__(self,  parent, debug=0):
        super(PyInterp,  self).__init__(parent)

        #TODO smarter line wrap (can be buggy if an item in history needs to wrap)

        self.debug = debug
        if not debug[0]:
            sys.stdout              = self
        if not debug[1]:
            sys.stderr              = self

        self.thread_enabled     = True
        self.write_sys          = True
        self.write_plots        = True
        self.write_settings     = True
        
        self.refreshMarker      = False # to change back to >>> from ...
        self.multiLine          = False # code spans more than one line
        self.prevcommand        = ''    # previously run command
        self.command            = ''    # command to be ran
        
        self.printBanner()              # print sys info
        self.marker()                   # make the >>> or ... marker        
        self.history            = []    # list of commands entered
        self.historyIndex       = -1
        self.interpreterLocals  = {}

        self.comm               = {}    # communicate with gui

        # setting the color for bg and text
        #TODO get pallette from profile
        palette = QPalette()
        palette.setColor(QPalette.Base, QColor("light gray"))
        palette.setColor(QPalette.Text, QColor("black"))
        self.setPalette(palette)
        self.setFont(QFont('Courier', 10))
        
        readline.set_completer(phcompleter.Completer().complete)
        readline.set_completer_delims(' \t\n`~!#$%^&*)-=+[{]}\\|;:,<>/?')
        #~ readline.parse_and_bind("tab: complete")

        # initilize interpreter with self locals
        self.initInterpreter(locals())
    
    def run_from_gui(self, command, write=False, thread=False, kind=None):
        # need to check against custom commands first
        if kind=='sys':
            write = write and self.write_sys
        if kind=='plots':
            write = write and self.write_plots
        if kind=='settings':
            write = write and self.write_settings
        if write:
            self.write(command+"\n")
        if self.customCommands(command):
            return None
        else:
            self.run(command,write,thread)
        if write:
            self.history.insert(0, command)
        if write and not thread:
            self.marker()
                
    def run(self, command, write=False, thread=False):
        if 'plt.' in command or 'plot' in command or 'load_data' in command or 'remove_axes' in command: #mpl will not work if threaded
            thread = False
            
        self.prevcommand = command
            
        if thread and self.thread_enabled:
            self.thread = PyInterpThread(self,command,self.debug)
            self.emit(SIGNAL("GUILock"),command,self.thread)
            #~ self.thread.bundle.attach_signal(self.thread.bundle.get_system(), "set_time", self.on_set_time)
            #~ print hasattr(self, 'bundle'), 'bundle' in globals().keys()
            QObject.connect(self.thread, SIGNAL("set_time"), self.on_set_time)
            QObject.connect(self.thread, SIGNAL("threadComplete"), self.update_from_thread)
            QObject.connect(self.thread, SIGNAL("threadFailed"), self.update_from_threadfailed)
            QObject.connect(self.thread.buff, SIGNAL("newtext"), self.update_textfromthread)
            
            self.thread.start()
        else:
            try:
                code = compile(command, '<string>', 'single')
                exec(code)
            except Exception as message:
                print(str(message))
                
            for key in locals().keys():
                if key is not 'self':
                    globals()[key] = locals()[key] # so everything is available from interpreter without self
                    self.comm[key]=locals()[key]   # so everything is available from the gui

            self.emit(SIGNAL("command_run"))
            self.marker()
        
            self.comp = rlcompleter.readline.get_completer()
            
    def on_set_time(self,*args):
        self.emit(SIGNAL("set_time"))
        
    def update_textfromthread(self, text):
        #~ self.insertPlainText(text)
        self.write(text)
        
    def update_from_thread(self,command):
        QObject.disconnect(self.thread, SIGNAL("threadComplete"), self.update_from_thread)
        QObject.disconnect(self.thread, SIGNAL("threadFailed"), self.update_from_threadfailed)
        QObject.disconnect(self.thread.buff, SIGNAL("newtext"), self.update_textfromthread)
        QObject.disconnect(self.thread, SIGNAL("set_time"), self.on_set_time)
        for key in self.thread.comm.keys():
            if key is not 'self' and key is not 'command':
                globals()[key] = self.thread.comm[key] # so everything is available from interpreter without self
                self.comm[key] = self.thread.comm[key]
        # changing system should create callback to main gui
        # until then...
        self.emit(SIGNAL("command_run"))
        self.emit(SIGNAL("GUIUnlock"))
        self.marker()
        self.comp = rlcompleter.readline.get_completer() 
        
    def update_from_threadfailed(self):
        QObject.disconnect(self.thread, SIGNAL("threadComplete"), self.update_from_thread)
        QObject.disconnect(self.thread, SIGNAL("threadFailed"), self.update_from_threadfailed)      
        QObject.disconnect(self.thread.buff, SIGNAL("newtext"), self.update_textfromthread) 
        QObject.disconnect(self.thread, SIGNAL("set_time"), self.on_set_time)
        self.write(str(self.thread.message)+"\n")
        self.emit(SIGNAL("GUIUnlock"))
        self.emit(SIGNAL("GUIthrowerror"),self.thread.message)
        self.marker()
        self.comp = rlcompleter.readline.get_completer() 

    def update_from_comm(self):
        for key in self.comm.keys():
            globals()[key] = self.comm[key]

    def printBanner(self):
        msg = '\nPHOEBE Python Interface\n'
        self.write(msg + '\n')
        #~ self.insertHtml('<a href="http://www.phoebe-project.org">PHOEBE Website</a>')

    def marker(self):
        if self.textCursor().positionInBlock()<4: #just to make sure
        #~ if True:
            if self.multiLine:
                self.insertPlainText('... ')
            else:
                self.insertPlainText('>>> ')

    def initInterpreter(self, interpreterLocals=None):
        if interpreterLocals:
            # when we pass in locals, we don't want it to be named "self"
            # so we rename it with the name of the class that did the passing
            # and reinsert the locals back into the interpreter dictionary
            selfName = interpreterLocals['self'].__class__.__name__
            interpreterLocalVars = interpreterLocals.pop('self')
            self.interpreterLocals[selfName] = interpreterLocalVars
        else:
            self.interpreterLocals = interpreterLocals
        self.interpreter = self.InteractiveInterpreter(self.interpreterLocals)

    def updateInterpreterLocals(self, newLocals):
        className = newLocals.__class__.__name__
        self.interpreterLocals[className] = newLocals

    def write(self, line):
        # TODO check line.split('.') and link to docs if available insertHtml('<a href="">%s</a>' % linesection)
        self.insertPlainText(line)
        self.ensureCursorVisible()

    def clearCurrentBlock(self):
        # block being current row
        length = len(self.document().lastBlock().text()[4:])
        if length == 0:
            return None
        else:
            self.textCursor().setPosition(len(self.document().toPlainText()))
            [self.textCursor().deletePreviousChar() for x in xrange(length)]
        return True

    def recallHistory(self):
        # used when using the arrow keys to scroll through history
        self.clearCurrentBlock()
        if self.historyIndex != -1:
            self.insertPlainText(self.history[self.historyIndex])
        return True

    def customCommands(self, command):
        # Disable forbidden terms
        for disabled in ['quit', 'exit', 'SystemExit', 'locals', 'globals', 'self']:
            if disabled in command:
                self.write("\n%s forbidden\n" % disabled)
                self.command=''
                self.multiLine=False
                self.marker()
                return True
            # comment out to access main GUI with self.parent.interpreterLocals['PhoebeGUI'].setVisible(False)
            
        #~ if command=='undo()':
            #~ self.write("\n")
            #~ self.marker()
            #~ self.emit(SIGNAL('undo'))
            #~ return True
            
        #~ if command=='redo()':
            #~ self.write("\n")
            #~ self.marker()
            #~ self.emit(SIGNAL('redo'))
            #~ return True

        #~ if command == 'clear()':
            #~ self.setText("")
            #~ self.marker()
            #~ return True
            
        if command == 'redraw()':
            self.write("\n")
            self.marker()
            self.emit(SIGNAL('plots_changed'))
            return True

        if command == '!hist': # display history
            self.append('') # move down one line
            # vars that are in the command are prefixed with ____CC and deleted
            # once the command is done so they don't show up in dir()
            backup = self.interpreterLocals.copy()
            history = self.history[:]
            history.reverse()
            for i, x in enumerate(history):
                iSize = len(str(i))
                delta = len(str(len(history))) - iSize
                line = line  = ' ' * delta + '%i: %s' % (i, x) + '\n'
                self.write(line)
            self.updateInterpreterLocals(backup)
            self.marker()
            return True

        if re.match('!hist\(\d+\)', command): # recall command from history
            backup = self.interpreterLocals.copy()
            history = self.history[:]
            history.reverse()
            index = int(command[6:-1])
            self.clearCurrentBlock()
            command = history[index]
            if command[-1] == ':':
                self.multiLine = True
            self.write(command)
            self.updateInterpreterLocals(backup)
            return True

        return False

    #~ def resizeEvent (self, event):
        #~ # TODO shrinking parent should scroll to end not beginning of viewable selection
        #~ return None

    def dragEnterEvent (self, event):
        # disables all drag events
        return None

    def keyPressEvent(self, event):   
        if not isinstance(event,str):
            key = event.key()
        else:
            key = event
            
        if self.textCursor().positionInBlock()<4:
            # ensure that we have a marker (in case of previous error)
            self.command=''
            self.multiLine=False
            self.marker()
            
        if self.textCursor().blockNumber()!=self.document().lastBlock().blockNumber() or key == Qt.Key_Home:
            # set cursor to position 4 in current block (end of marker)
            blockLength = len(self.document().lastBlock().text()[4:])
            lineLength  = len(self.document().toPlainText())
            position = lineLength - blockLength
            textCursor  = self.textCursor()
            textCursor.setPosition(position)
            self.setTextCursor(textCursor)
            if key == Qt.Key_Home:
                return None

        if key == Qt.Key_Escape:
            self.customCommands('clear()')
            return None

        if key == Qt.Key_Tab:
            #TODO fix for middle-of-line tab-completion - still buggy
            #TODO consider changing to use QCompleter
            pos = self.textCursor().positionInBlock()-4
            line = str(self.document().lastBlock().text())[4:]
            line.rstrip()
            eol = line[pos:]
            term = line[:pos]
            term_orig = term
            
            for sep in rlcompleter.readline.get_completer_delims():
                term = term.split(sep)[-1]
            bol = line[:term_orig.rfind(term)]   # this should work for all (most) cases

            results=[]

            if line.endswith('('):
                term = line.split('(')[-2]
                try:
                    results = inspect.getargspec(eval(term))
                except TypeError:
                    try:
                        results = inspect.getargspec(eval(term+".__init__"))
                    except TypeError or KeyError:
                        pass

                if len(results)>0:
                    self.write("\n")
                    self.write("%s( " % (term))
                    if results[0][0] == 'self':    results[0].pop(0) 
                    for i in range(len(results[0])):  #TODO clean this up
                        if results[3] is not None and len(results[3]) > i and results[3][i] is not None:
                            self.write("%s=%s, " % (results[0][i], results[3][i]))
                        else:
                            self.write("%s, " % (results[0][i]))
                    if results[1] is not None:
                        self.write("%s, " % results[1])
                    if results[2] is not None:
                        self.write("*%s, " % results[2])
                    self.write(")\n")
                    self.marker()
                    self.write(line)

                return None
                
            elif len(term)!=0:
                #~ self.run("comp = rlcompleter.readline.get_completer()")
                #~ comp = self.comm['comp']
                comp = self.comp
                #~ comp = rlcompleter.readline.get_completer()  
                
                i=0

                #TODO fix using history as well
                if '.' in term: # really should just edit self.comp to take all of these cases
                    term_a = term.rstrip(term.split('.')[-1])
                    bol+=term_a
                    term_b = term.lstrip(term_a) #TODO there is most definitely a better way to do this
                    term_a = term_a.rstrip('.') #TODO there is probably a better way to do this
                   

                    try:
                        results += [a for a in dir(globals()[term_a]) if ((len(term_b)==0 or a.startswith(term_b) and "_" not in a))] #get class from string
                    except KeyError:
                           pass
                           
                if len(term.split('.'))==1:
                #~ if True:
                    out = comp(term, i)
                    while isinstance(out, str):
                        results.append(out)
                        i+=1
                        out = comp(term, i)

            if len(results)==1:
                self.clearCurrentBlock()
                #~ if inspect.isfunction(eval(results[0])):
                    #~ results[0]+=")"
                #TODO check isfunc to see whether to add a (
                self.write("%s%s%s" % (bol, results[0], eol))
            if len(results)>1:
                self.write("\n")
                for out in results:
                    self.write("   %s" % out)
                self.write("\n")
                self.marker()
                self.write(line)

            if len(results)>0:
                return None 
            if not self.multiLine:  #allow blank tab for multiline only
                return None
            
        if key == Qt.Key_Down:
            if self.historyIndex == len(self.history):
                self.historyIndex -= 1
            try:
                if self.historyIndex > -1:
                    self.historyIndex -= 1
                    self.recallHistory()
                else:
                    self.clearCurrentBlock()
            except:
                pass
            return None

        if key == Qt.Key_Up:
            try:
                if len(self.history) - 1 > self.historyIndex:
                    self.historyIndex += 1
                    self.recallHistory()
                else:
                    self.historyIndex = len(self.history)
            except:
                pass
            return None

        if key in [Qt.Key_Left, Qt.Key_Backspace]:
            # don't allow deletion of marker
            if self.textCursor().positionInBlock() == 4:
                return None

        if key in [Qt.Key_Return, Qt.Key_Enter] or key=='run':
            # set cursor to end of line to avoid line splitting
            textCursor = self.textCursor()
            position   = len(self.document().toPlainText())
            textCursor.setPosition(position)
            self.setTextCursor(textCursor)

            line = str(self.document().lastBlock().text())[4:] # remove marker
            line.rstrip()
            self.historyIndex = -1

            if self.customCommands(line):
                return None
            else:
                try:
                    line[-1]
                    self.haveLine = True
                    if line[-1] == ':':
                        self.multiLine = True
                    self.history.insert(0, line)
                except:
                    self.haveLine = False

                if self.haveLine and self.multiLine: # multi line command
                    self.command += line + '\n' # + command and line
                    self.append('') # move down one line
                    self.marker() # handle marker style
                    return None

                if self.haveLine and not self.multiLine: # one line command
                    self.command = line # line is the command
                    self.append('') # move down one line
                    self.run(self.command, thread=True)
                    self.command = '' # clear command
                    #~ self.marker() # handle marker style - move because of threading
                    return None

                if self.multiLine and not self.haveLine: #  multi line done
                    self.append('') # move down one line
                    self.run(self.command, thread=True)
                    self.command = '' # clear command
                    self.multiLine = False # back to single line
                    #~ self.marker() # handle marker style - move because of threading
                    return None

                if not self.haveLine and not self.multiLine: # just enter
                    self.append('')
                    self.marker() 
                    return None
                return None
                
        # allow all other key events
        super(PyInterp, self).keyPressEvent(event)

######################################################################
#### OpenGL renderer for viewing and rotating the mesh   #############
#### in the GUI                                          #############
######################################################################

class mesh_widget(QGLWidget):
    def __init__(self, system, parent=QWidget):
        self.parent = parent
        QGLWidget.__init__(self, system, parent)
        
        self.mesh_data = None
        self.min_size = None
        self.y_rot = 0.0
        self.last_pos = QPoint()
            
    def setMesh(self,mesh):
        self.mesh_data = mesh
        #~ self.y_rot = 0.0
        #~ self.last_pos = QPoint()
        
        if (self.mesh_data == None):            
            self.max_color = 1.0
            self.max_size = 10.0
            self.min_size = -10.0
        else:
            # if we want to visualize something other than 'teff'
            # on the star then color variables need a different
            # datatype name called
            self.max_color = self.mesh_data['teff'].max()
            self.max_size = self.mesh_data['triangle'].max()
            self.min_size = self.mesh_data['triangle'].min()
            
        self.resize(600,600)
            
    def paintGL(self):         
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.mesh_data is None or self.min_size is None:
            glColor3f(0.,0.,0.)
            return

        glLoadIdentity();
        glRotate(self.y_rot/16.0, 0.0, 1.0, 0.0)

        if (self.mesh_data != None):
            for index in xrange(self.mesh_data.size):
                glBegin(GL_TRIANGLES)
                color = self.mesh_data['teff'][index] / (self.max_color)
                glColor3f(color, color, color)
                glVertex3f(self.mesh_data['triangle'][index][0], 
                           self.mesh_data['triangle'][index][1], 
                           self.mesh_data['triangle'][index][2])
                glVertex3f(self.mesh_data['triangle'][index][3], 
                           self.mesh_data['triangle'][index][4], 
                           self.mesh_data['triangle'][index][5])
                glVertex3f(self.mesh_data['triangle'][index][6], 
                           self.mesh_data['triangle'][index][7], 
                           self.mesh_data['triangle'][index][8])
                glEnd()

    def resizeGL(self, w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.mesh_data is None or self.min_size is None:
            return

        ### Adjust the size of the aspect size of the viewing screen here
        glOrtho(self.min_size*1.5, self.max_size*1.5, 
                self.min_size*1.5, self.max_size*1.5, 
                self.min_size*1.5, self.max_size*1.5)
        #################

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

    def set_y_rotation(self, angle):
        self.normalize_angle(angle)
        if (angle != self.y_rot):
            self.y_rot = angle
            self.updateGL()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        
    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        self.set_y_rotation(self.y_rot + 8 * dx)
        self.last_pos = event.pos()

    def mouseDoubleClickEvent(self, event):
        self.set_y_rotation(0.0)

    def normalize_angle(self, angle):
        while (angle < 0):
            angle += 360 * 16
        while (angle > 360 * 16):
            angle -= 360 * 16
