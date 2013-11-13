from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

import ui_phoebe_pyqt4 as gui

### classes to allow creating the popup dialogs or attaching widgets
class CreatePopAbout(QDialog, gui.Ui_popAbout_Dialog):
    def __init__(self, parent=None):
        super(CreatePopAbout, self).__init__(parent)
        self.setupUi(self)

        #~ self.setFixedSize(self.size())

class CreatePopHelp(QDialog, gui.Ui_popHelp_Dialog):
    def __init__(self, parent=None):
        super(CreatePopHelp, self).__init__(parent)
        self.setupUi(self)

class CreatePopLock(QDialog, gui.Ui_popLock_Dialog):
    def __init__(self, parent=None, lock_info=None, server=None):
        super(CreatePopLock, self).__init__(parent)
        self.setupUi(self)
        
        self.setup(lock_info,server)
        
    def setup(self,lock_info,server):
        self.lock_info = lock_info
        self.server = server
        
        self.server_label.setText('Server: %s' % lock_info['server'])
        self.job_label.setText('Job: %s' % lock_info['command'].strip())
        
        self.connect(self.loop_button, SIGNAL("clicked()"), self.on_loop_clicked)
        self.connect(self.getresults_button, SIGNAL("clicked()"), self.on_getresults_clicked)
        self.connect(self.unlock_button, SIGNAL("clicked()"), self.on_unlock_clicked)
        
        self.server_refresh.server = self.server
        self.job_refresh.server = self.server
        self.job_refresh.job = self.lock_info['script']

        self.server_refresh.update_status()
        self.job_refresh.update_status()
        
    def on_loop_clicked(self,*args):
        self.on_refresh_server()
        self.on_refresh_job()
        
        do_command = 'bundle.server_loop()'
        undo_command = None
        description = 'wait for results from server'
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'system')
        
    def on_getresults_clicked(self,*args):
        self.on_refresh_server()
        self.on_refresh_job()
        
        do_command = 'bundle.server_get_results()'
        undo_command = None
        description = 'get results from server'
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'system')
        
    def on_unlock_clicked(self,*args):
        do_command = 'bundle.server_cancel()'
        undo_command = None
        description = 'cancel job on server'
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'system')
        
        #~ self.hide()
        
class CreatePopPrefs(QDialog, gui.Ui_popPrefs_Dialog):
    def __init__(self, parent=None, prefs=None):
        super(CreatePopPrefs, self).__init__(parent)
        self.setupUi(self)
        self.prefs = prefs
        
        self.pref_bools = self.findChildren(QCheckBox)
        self.pref_bools += self.findChildren(QRadioButton)
        
        self.connect(self.servers_returntolist, SIGNAL("clicked()"), self.on_servers_returntolist)
        self.connect(self.serverlist_edit, SIGNAL("clicked()"), self.on_edit_server_clicked)
        self.connect(self.serverlist_add, SIGNAL("clicked()"), self.on_add_server_clicked)
        self.connect(self.serverlist_recheck, SIGNAL("clicked()"), self.on_recheck_servers_clicked)
        
        self.set_gui_from_prefs(prefs,init=True)
        
    def set_gui_from_prefs(self,prefs=None,init=False):
        if prefs is None:
            prefs = self.prefs
        else:
            self.prefs = prefs
        
        p = {key: self.prefs.get_value(key) for key in self.prefs.default_preferences.keys()}
        
        bools = self.findChildren(QCheckBox)
        bools += self.findChildren(QRadioButton)

        for w in bools:
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setEnabled(False)
                    w.setChecked(p[key])
                    w.setEnabled(True)
                    w.current_value = p[key]
                if init:
                    self.connect(w, SIGNAL("toggled(bool)"), self.item_changed)
                    
        for w in self.findChildren(QPlainTextEdit):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setEnabled(False) # in case the signal is already connected
                    w.setPlainText(p[key])
                    if w.objectName() != 'p_pyinterp_startup_default':
                        w.setEnabled(True)
                    w.current_value = p[key]
                if init:
                    self.connect(w, SIGNAL("textChanged()"), self.text_changed)
                
                # now find the save button and connect signals and link
                button = self.get_button_for_textbox(w)
                if button is not None:
                    button.setVisible(False)
                    button.textbox = w
                    if init:
                        self.connect(button, SIGNAL("clicked()"), self.item_changed)
                    
        #~ for w in self.findChildren(QSpinBox):
            #~ if w.objectName().split('_')[0]=='p':
                #~ key = "_".join(str(w.objectName()).split('_')[1:])
                #~ if key in p:
                    #~ w.setValue(p[key])
                    #~ w.current_value = p[key]
                #~ self.connect(w, SIGNAL("toggled(bool)"), self.item_changed)
                
            #~ for w in self.findChildren(QComboBox):
                #~ pass
                
        for w in self.findChildren(QComboBox):
            if w.objectName().split('_')[0]=='s':
                key = "_".join(str(w.objectName()).split('_')[1:])
                names = ['None']+self.prefs.get_server().keys()
            elif w.objectName() == 'sx_serveredit_combo':
                key = None
                names = self.prefs.get_server().keys()
            elif w.objectName() == 'co_edit_combo':
                key = None
                names = self.prefs.get_compute().keys()
            elif w.objectName() == 'fo_edit_combo':
                key = None
                names = self.prefs.get_fitting().keys()
            else:
                continue
            
            orig_text = str(w.currentText())
            w.setEnabled(False)
            w.clear()
            w.addItems(names)

            if key is not None:
                w.current_value = p[key]
                if p[key] is not False:
                    if p[key] in names:
                        w.setCurrentIndex(names.index(p[key]))
            else: #then we want to try to restore to original selection
                if orig_text in names:
                    w.setCurrentIndex(names.index(orig_text))
            w.setEnabled(True)                    
            
            if init:
                if key is None: #then this doesn't control a parameter
                    if w.objectName() == 'sx_serveredit_combo':
                        self.connect(w, SIGNAL("currentIndexChanged(QString)"), self.serveredit_changed) 
                        self.serveredit_changed(names[0])
                    elif w.objectName() == 'co_edit_combo':
                        self.connect(w, SIGNAL("currentIndexChanged(QString)"), self.coedit_changed)
                        self.coedit_changed(names[0])
                    elif w.objectName() == 'fo_edit_combo':
                        self.connect(w, SIGNAL("currentIndexChanged(QString)"), self.foedit_changed)
                        self.foedit_changed(names[0])
                else:
                    self.connect(w, SIGNAL("currentIndexChanged(QString)"), self.item_changed)
                
        ### defaults stuff
        if init:
            self.connect(self.co_psedit, SIGNAL("parameterChanged"), self.on_coparam_changed)
            self.connect(self.fo_psedit, SIGNAL("parameterChanged"), self.on_foparam_changed)
            self.connect(self.co_add, SIGNAL("clicked()"), self.on_add_co_clicked)
            #~ self.connect(self.fo_add, SIGNAL("clicked()"), self.on_add_fo_clicked)
        
        ### server stuff
        if init:
            for tv in [self.sx_serveredit_psedit, self.sx_serveredit_mpipsedit]:
                self.connect(tv, SIGNAL("parameterChanged"), self.on_serverparam_changed)
                    
        ### create server list treeview
        self.serverlist_treeWidget.set_data(self.prefs.get_server())
        if init:
            self.connect(self.serverlist_treeWidget, SIGNAL("edit_server_clicked"), self.on_edit_server_clicked)
            self.connect(self.serverlist_treeWidget, SIGNAL("delete_server_clicked"), self.on_delete_server_clicked)
        
    def get_button_for_textbox(self,textedit):
        button_name = 'save_' + '_'.join(str(textedit.objectName()).split('_')[1:])
        buttons = self.findChildren(QPushButton,QString(button_name))
        if len(buttons)==1:
            return buttons[0]
        else:
            return None
          
    def text_changed(self,*args):
        button = self.get_button_for_textbox(self.sender())
        if button is not None:
            button.setVisible(True)
            
    def item_changed(self,*args):
        ## TODO - for now this is really only set to handle bools and textedits with save buttons
        
        w = self.sender()
        
        if not w.isEnabled():
            return
        
        if hasattr(w,'textbox'):
            w.setVisible(False)
            w = w.textbox
            # we need to be careful with newlines and quotes since we're
            # sending this through PyInterp
            new_value = str(w.toPlainText()).replace('\n','\\n')
        else:
            new_value = args[0]
        
        prefname = '_'.join(str(w.objectName()).split('_')[1:])
        
        old_value = w.current_value
        
        do_command = "settings.set_value('%s',%s)" % (prefname,"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.set_value('%s',%s)" % (prefname,"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change setting: %s" % prefname
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def serveredit_changed(self,servername):
        if servername == 'None' or servername == '':
            server = None
        else:
            server = self.prefs.get_server(str(servername))
        
        # update push button and check status
        self.sx_serveredit_status.server = server
        self.sx_serveredit_status.update_status()
        
        # set data for parameter tree views
        self.sx_serveredit_psedit.set_data([server.server_ps] if server is not None else [],style=['nofit','incl_label'])
        self.sx_serveredit_mpipsedit.set_data([server.mpi_ps] if server is not None and server.mpi_ps is not None else [],style=['nofit','incl_label'])
        
    def coedit_changed(self,label):
        if label == 'None' or label == '':
            co = None
        else:
            co = self.prefs.get_compute(str(label))
            
        self.co_psedit.set_data([co] if co is not None else [],style=['nofit','incl_label'])
        
    def foedit_changed(self,label):
        if label == 'None' or label == '':
            fo = None
        else:
            fo = self.prefs.get_fitting(str(label))
            
        self.fo_psedit.set_data([fo] if fo is not None else [],style=['nofit'])
        
    def on_serverparam_changed(self,treeview,label,param,old_value,new_value,oldunit=None,newunit=None,is_adjust=False,is_constraint=False):
        
        label = str(self.sx_serveredit_combo.currentText())
        
        do_command = "settings.get_server('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.get_server('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change setting: %s" % param.get_qualifier()
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_coparam_changed(self,treeview,label,param,old_value,new_value,oldunit=None,newunit=None,is_adjust=False,is_constraint=False):
        
        label = str(self.co_edit_combo.currentText())
        
        do_command = "settings.get_compute('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.get_compute('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change compute %s: %s" % (label,param.get_qualifier())
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_foparam_changed(self,treeview,label,param,old_value,new_value,oldunit=None,newunit=None,is_adjust=False,is_constraint=False):
        
        label = str(self.fo_edit_combo.currentText())
        
        do_command = "settings.get_fitting('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.get_fitting('%s').set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change fitting %s: %s" % (label,param.get_qualifier())
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_add_co_clicked(self):
        
        label = 'new compute'
        do_command = "settings.add_compute(label='%s')" % (label)
        undo_command = "settings.remove_compute(label='%s')" % (label)
        description = "add new compute options"
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_add_server_clicked(self):
        
        label = 'new server'
        do_command = "settings.add_server('%s')" % (label)
        undo_command = "settings.remove_server('%s')" % (label)
        description = "add new server"
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_delete_server_clicked(self,servername,server):
        
        do_command = "settings.remove_server('%s')" % (servername)
        undo_command = "settings.add_server('%s')" % (servername)
        description = "remove %s server" % (servername)
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_recheck_servers_clicked(self):
        for button in self.serverlist_treeWidget.statusbuttons:
            button.update_status()
        
    def on_edit_server_clicked(self,servername=None,server=None):
        if servername is not None:
            self.sx_serveredit_combo.setCurrentIndex(self.prefs.get_server().keys().index(servername))
        
        self.servers_config_stackedWidget.setCurrentIndex(1)

    def on_servers_returntolist(self,*args):
        self.servers_config_stackedWidget.setCurrentIndex(0)

class CreatePopObsOptions(QDialog, gui.Ui_popObsOptions_Dialog):
    def __init__(self, parent=None):
        super(CreatePopObsOptions, self).__init__(parent)
        self.setupUi(self)

class CreatePopPlot(QDialog, gui.Ui_popPlot_Dialog):
    def __init__(self, parent=None):
        super(CreatePopPlot, self).__init__(parent)
        self.setupUi(self)
        
class CreatePopFileEntry(QDialog, gui.Ui_popFileEntry_Dialog):
    def __init__(self, parent=None):
        super(CreatePopFileEntry, self).__init__(parent)
        self.setupUi(self)
        
class CreatePopFileEntryColWidget(QWidget, gui.Ui_popFileEntryColWidget):
    def __init__(self, parent=None):
        super(CreatePopFileEntryColWidget, self).__init__(parent)
        self.setupUi(self)
        
class CreateFileEntryWidget(QWidget, gui.Ui_fileEntryWidget):
    def __init__(self, parent=None):
        super(CreateFileEntryWidget, self).__init__(parent)
        self.setupUi(self)
        
class CreateDatasetWidget(QWidget, gui.Ui_datasetWidget):
    def __init__(self, parent=None):
        super(CreateDatasetWidget, self).__init__(parent)
        self.setupUi(self)
