from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

import ui_phoebe_pyqt4 as gui
import os
import numpy as np

from phoebe.parameters import datasets
from phoebe.atmospheres import passbands

def get_datarefs(bundle):
    """
    returns a list of datarefs for synthetic datasets
    
    note this does not include duplicates if multiple objrefs have the same dataref
    """
    
    ds_syn_all = bundle.get_syn(all=True).values()
    ds_syn_names = []
    for dss in ds_syn_all:
        if dss['ref'] not in ds_syn_names:
            ds_syn_names.append(dss['ref'])    
    return ds_syn_names


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
        self.server_refresh.server_kind = 'mount'
        self.job_refresh.server = self.server
        self.job_refresh.job = self.lock_info['script']

        self.server_refresh.update_status()
        self.job_refresh.update_status()
        
    def on_loop_clicked(self,*args):
        self.server_refresh.update_status()
        self.job_refresh.update_status()
        
        do_command = 'bundle.server_loop()'
        undo_command = None
        description = 'wait for results from server'
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'system')
        
    def on_getresults_clicked(self,*args):
        self.server_refresh.update_status()
        self.job_refresh.update_status()
        
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
    def __init__(self, parent=None, prefs=None, devel_version=False):
        super(CreatePopPrefs, self).__init__(parent)
        self.setupUi(self)
        self.prefs = prefs
        
        self.pref_bools = self.findChildren(QCheckBox)
        self.pref_bools += self.findChildren(QRadioButton)
        
        self.connect(self.servers_returntolist, SIGNAL("clicked()"), self.on_servers_returntolist)
        self.connect(self.serverlist_edit, SIGNAL("clicked()"), self.on_edit_server_clicked)
        self.connect(self.serverlist_add, SIGNAL("clicked()"), self.on_add_ps_clicked)
        self.connect(self.serverlist_recheck, SIGNAL("clicked()"), self.on_recheck_servers_clicked)
        
        self.connect(self.sx_serveredit_add, SIGNAL("clicked()"), self.on_add_ps_clicked)
        self.connect(self.sx_serveredit_delete, SIGNAL("clicked()"), self.on_delete_ps_clicked)
        self.connect(self.sx_serveredit_combo, SIGNAL("currentIndexChanged(QString)"), self.serveredit_changed) 
        self.connect(self.co_edit_combo, SIGNAL("currentIndexChanged(QString)"), self.coedit_changed)
        self.connect(self.fo_edit_combo, SIGNAL("currentIndexChanged(QString)"), self.foedit_changed)

        self.connect(self.co_psedit, SIGNAL("parameterChanged"), self.on_coparam_changed)
        self.connect(self.fo_psedit, SIGNAL("parameterChanged"), self.on_foparam_changed)
        self.connect(self.co_add, SIGNAL("clicked()"), self.on_add_ps_clicked)
        #~ self.connect(self.fo_add, SIGNAL("clicked()"), self.on_add_ps_clicked)
        self.connect(self.co_delete, SIGNAL("clicked()"), self.on_delete_ps_clicked)
        self.connect(self.fo_delete, SIGNAL("clicked()"), self.on_delete_ps_clicked)
            
        for tv in [self.sx_serveredit_psedit, self.sx_serveredit_mpipsedit]:
            self.connect(tv, SIGNAL("parameterChanged"), self.on_serverparam_changed)
            
        self.connect(self.serverlist_treeWidget, SIGNAL("edit_server_clicked"), self.on_edit_server_clicked)
        self.connect(self.serverlist_treeWidget, SIGNAL("delete_server_clicked"), self.on_delete_ps_clicked)
        
        self.connect(self.lo_psedit, SIGNAL("parameterChanged"), self.on_loparam_changed)
        
        self.set_gui_from_prefs(prefs,init=True)
        
        if not devel_version:
            # then disable certain items
            self.p_panel_fitting.setEnabled(False)
            self.p_panel_versions.setEnabled(False)
            self.p_panel_system.setEnabled(False) # maybe enable for release if ready?
            
            # fitting options
            self.label_13.setVisible(False)
            self.fo_edit_combo.setVisible(False)
            self.fo_delete.setVisible(False)
            self.fo_add.setVisible(False)
            self.fo_psedit.setVisible(False)
        
    def set_gui_from_prefs(self,prefs=None,init=False):
        if prefs is None:
            prefs = self.prefs
        else:
            self.prefs = prefs
        
        p = self.prefs.get_gui()
        
        bools = self.findChildren(QCheckBox)
        bools += self.findChildren(QRadioButton)

        for w in bools:
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setEnabled(False)
                    w.setChecked(p[key])
                    w.setEnabled(True)
                    w.current_value = p.get_value(key)
                if init:
                    self.connect(w, SIGNAL("toggled(bool)"), self.item_changed)
                    
        for w in self.findChildren(QPlainTextEdit):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setEnabled(False) # in case the signal is already connected
                    w.setPlainText(p[key])
                    w.current_value = p.get_value(key)
                    w.setEnabled(True)
                if init:
                    self.connect(w, SIGNAL("textChanged()"), self.text_changed)
                
                # now find the save button and connect signals and link
                button = self.get_button_for_textbox(w)
                if button is not None:
                    button.setVisible(False)
                    button.textbox = w
                    if init:
                        self.connect(button, SIGNAL("clicked()"), self.item_changed)
            elif w.objectName() == 'px_pyinterp_startup_default':
                # this string is hardcoded - also needs to be in phoebe_gui.__init__
                startup_default = 'import phoebe\nfrom phoebe.frontend.bundle import Bundle, load\nfrom phoebe.parameters import parameters, create, tools\nfrom phoebe.io import parsers\nfrom phoebe.utils import utils\nfrom phoebe.frontend import usersettings\nsettings = usersettings.load()'
                w.setPlainText(startup_default)
                w.setEnabled(False)
                    
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
                names = ['None']+self.prefs.get_server(all=True).keys()
            elif w.objectName() == 'sx_serveredit_combo':
                key = None
                names = self.prefs.get_server(all=True).keys()
            elif w.objectName() == 'co_edit_combo':
                key = None
                names = self.prefs.get_compute(all=True).keys()
            elif w.objectName() == 'fo_edit_combo':
                key = None
                names = self.prefs.get_fitting(all=True).keys()
            else:
                continue
            
            orig_text = str(w.currentText())
            w.setEnabled(False)
            w.clear()
            w.addItems(names)

            if key is not None:
                w.current_value = p.get_value(key)
                if p.get_value(key) is not False:
                    if p.get_value(key) in names:
                        w.setCurrentIndex(names.index(p.get_value(key)))
            else: #then we want to try to restore to original selection
                if orig_text in names:
                    w.setCurrentIndex(names.index(orig_text))
            w.setEnabled(True)                    
            
            if init:
                if key is None: #then this doesn't control a parameter
                    if w.objectName() == 'sx_serveredit_combo':
                        if len(names):
                            self.serveredit_changed(names[0])
                    elif w.objectName() == 'co_edit_combo':
                        if len(names):
                            self.coedit_changed(names[0])
                    elif w.objectName() == 'fo_edit_combo':
                        if len(names):
                            self.foedit_changed(names[0])
                else:
                    self.connect(w, SIGNAL("currentIndexChanged(QString)"), self.item_changed)
                
        ### create server list treeview
        self.serverlist_treeWidget.set_data(self.prefs.get_server(all=True))
            
        # logger stuff
        self.lo_psedit.set_data([self.prefs.get_logger()],style=['nofit'])
        
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
        
        do_command = "settings.get_gui().set_value('%s',%s)" % (prefname,"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.get_gui().set_value('%s',%s)" % (prefname,"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change setting: %s" % prefname
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def serveredit_changed(self,servername):
        if servername == 'None' or servername == '':
            server = None
        else:
            server = self.prefs.get_server(str(servername))
        
        # set data for parameter tree views
        self.sx_serveredit_psedit.set_data([server.settings['server']] if server is not None else [],style=['nofit','incl_label'])
        self.sx_serveredit_mpipsedit.set_data([server.settings['mpi']] if server is not None and server.settings['mpi'] is not None else [],style=['nofit','incl_label'])
        
    def coedit_changed(self,label):
        if label == 'None' or label == '':
            co = None
        else:
            co = self.prefs.get_compute(str(label))
            
        self.co_psedit.set_data([co] if co is not None else [],style=['nofit','incl_label'],hide_params=['time','refs','types'])
        
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
        
    def on_loparam_changed(self,treeview,label,param,old_value,new_value,oldunit=None,newunit=None,is_adjust=False,is_constraint=False):
        
        label = 'logger'

        do_command = "settings.get_%s().set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % new_value if isinstance(new_value,bool) else "\"%s\"" % new_value)
        undo_command = "settings.get_%s().set_value('%s',%s)" % (label,param.get_qualifier(),"%s" % old_value if isinstance(old_value,bool) else "\"%s\"" % old_value)
        description = "change logger %s" % (param.get_qualifier())
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_add_ps_clicked(self):
        w = self.sender()
        
        if w == self.co_add:
            typ = 'compute'
        elif w == self.fo_add:
            typ = 'fitting'
        elif w == self.sx_serveredit_add or w == self.serverlist_add:
            typ = 'server'
        
        label = 'new %s' % typ
        do_command = "settings.add_%s(label='%s')" % (typ,label)
        undo_command = "settings.remove_%s(label='%s')" % (typ,label)
        description = "add new %s" % typ
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
        
    def on_delete_ps_clicked(self):
        w = self.sender()
        
        if w == self.co_delete:
            typ = 'compute'
            label = str(self.co_edit_combo.currentText())
        elif w == self.fo_delete:
            typ = 'fitting'
            label = str(self.fo_edit_combo.currentText())
        elif w == self.sx_serveredit_delete:
            typ = 'server'
            label = str(self.sx_serveredit_combo.currentText())
            
        do_command = "settings.remove_%s('%s')" % (typ,label)
        undo_command = "settings.add_%s('%s')" % (typ,label)
        description = "remove %s %s" % (label,typ)
        self.emit(SIGNAL("parameterCommand"),do_command,undo_command,description,False,'settings')
                
    def on_recheck_servers_clicked(self):
        for button in self.serverlist_treeWidget.statusbuttons:
            button.update_status()
        #~ self.emit(SIGNAL("serverschanged"),True)
        
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

class CreatePopTimeSelect(QDialog, gui.Ui_popTimeSelect_Dialog):
    def __init__(self, parent, bundle):
        super(CreatePopTimeSelect, self).__init__(parent)
        self.setupUi(self)
        
        for w in [self.compute_min, self.compute_max, self.compute_median, self.compute_mean]:
            self.connect(w, SIGNAL("clicked()"), self.on_compute)
            
        self.ds_names = get_datarefs(bundle)
        self.datasetComboBox.addItems(['*all*']+self.ds_names)

        self.bundle = bundle
        
    def get_times(self,dataref):
        """
        returns the times (np.array) given a dataref
        this function gets the first syn that matches, not caring about objref
        as we are assuming they should all have the same times
        """
        # used to be get_syn - but that won't work until compute has been run
        return self.bundle.get_obs(dataref=dataref,all=True).values()[0].asarray()['time']
        
    def get_time(self):
        """
        this function returns the currently set time, or None if it hasn't been set
        """
        time_str = self.time.text()
        
        if time_str == '':
            return None
        else:
            return float(time_str)

    def on_compute(self):
        """
        compute time and fill time button
        user still has to accept the time before returning value
        """
        w = self.sender()
        compute = str(w.objectName()).split('_')[1] # min, max, median, mean
        
        dataref = str(self.datasetComboBox.currentText())
        
        if dataref == '*all*':
            times = np.array([])
            for dr in self.ds_names:
                times = np.append(times, self.get_times(dr))
        else:
            times = self.get_times(dataref)
            #~ print "***", dataref, len(times)
        
        if len(times):
            if compute == 'min':
                time = times.min()
            elif compute == 'max':
                time = times.max()
            elif compute == 'median':
                time = np.median(times)
            elif compute == 'mean':
                time = np.mean(times)
            
            self.time.setText(str(time))
        
        
class CreatePopFileEntry(QDialog, gui.Ui_popFileEntry_Dialog):
    def __init__(self, parent=None,devel_version=False):
        super(CreatePopFileEntry, self).__init__(parent)
        self.setupUi(self)
        self.devel_version = devel_version
        self.syn_components_checks = {}
        
        self.colwidgets = []
        
        self.pfe_fileReloadButton.setVisible(False) # TODO - will need to change this when allowing editing already created datasets
        
        self.dataTextEdit.setVisible(False)
        self.syn_timeWidget.setVisible(False)

        
        self.set_filters()
        self.on_category_changed('lc') # initialize the datatypes combo to lc
        
        self.connect(self.pfe_fileChooserButton, SIGNAL("clicked()"), self.on_file_choose)
        self.connect(self.pfe_synChooserButton, SIGNAL("clicked()"), self.on_syn_choose)
        self.connect(self.pfe_categoryComboBox, SIGNAL("currentIndexChanged(QString)"), self.on_category_changed)
        self.connect(self.times_timeorphase, SIGNAL("currentIndexChanged(QString)"), self.on_syn_set_default_times)
        
        for w in [self.timeselect_arange_min, self.timeselect_arange_max, self.timeselect_linspace_min, self.timeselect_linspace_max]:
            self.connect(w, SIGNAL("clicked()"), self.on_timeselect_clicked)
            
        if not devel_version:
            self.sigmaLabel.setVisible(False)
            self.sigmaSpinBox.setVisible(False)
            
    def set_filters(self):
        self.filtertypes = {}
        filters = passbands.list_response()
        for filt in filters:
            fset,fband = filt.split('.')[:]
            if fset not in self.filtertypes:
                self.filtertypes[fset] = []
            self.filtertypes[fset].append(fband)
            
        self.pfe_filtersetComboBox.addItems(sorted(self.filtertypes.keys()))
        self.pfe_filterbandComboBox.setEnabled(False)
        self.connect(self.pfe_filtersetComboBox, SIGNAL("currentIndexChanged(QString)"), self.on_filterset_changed)
        #~ self.pfe_filterComboBox.addItems(self.filtertypes)
        
    def on_filterset_changed(self,filterset):
        filterset = str(filterset)
        self.pfe_filterbandComboBox.clear()
        if filterset in self.filtertypes.keys():
            self.pfe_filterbandComboBox.addItems(['--Passband--']+self.filtertypes[filterset])
            self.pfe_filterbandComboBox.setEnabled(True)
        else:
            self.pfe_filterbandComboBox.addItems(['--Passband--'])
            self.pfe_filterbandComboBox.setEnabled(False)
            
    def on_syn_set_default_component_checks(self):
        for name,ps in zip(self.parent().system_names,self.parent().system_ps):
            if name in self.syn_components_checks:
                check = self.syn_components_checks[name]
        
                if self.category in ['lc','etv']:
                    if ps.context in ['orbit']:
                        check.setChecked(True)
                    else:
                        check.setChecked(False)
                elif self.category in ['rv']:
                    if ps.context not in ['orbit']:
                        check.setChecked(True)
                    else:
                        check.setChecked(False)
                else:
                    check.setChecked(False)
                    
    def on_syn_set_default_times(self, timeorphase):
        if str(timeorphase)=='time':
            try:
                period = self.parent().bundle.get_value('period')
            except:
                # maybe there was more than one period
                pass
            else:
                self.arange_max.setValue(period)
                self.linspace_max.setValue(period)
        
            self.timeselect_arange_min.setEnabled(True)
            self.timeselect_arange_max.setEnabled(True)
            self.timeselect_linspace_min.setEnabled(True)
            self.timeselect_linspace_max.setEnabled(True)
            #~ self.times_match.setEnabled(True)
        
        else:
            self.arange_max.setValue(1.)
            self.linspace_max.setValue(1.)

            self.timeselect_arange_min.setEnabled(False)
            self.timeselect_arange_max.setEnabled(False)
            self.timeselect_linspace_min.setEnabled(False)
            self.timeselect_linspace_max.setEnabled(False)
            #~ if self.times_match.isChecked():
                #~ self.times_arange.setChecked(True)
            #~ self.times_match.setEnabled(False)
            
    def on_syn_choose(self):
        # show the time chooser widget
        
        # set datarefs
        datarefs = get_datarefs(self.parent().bundle)
        self.datasetComboBox.addItems(datarefs)
        
        # set components
        vbox = QVBoxLayout(self.syn_componentsWidget)
        for name in self.parent().system_names:
            check = QCheckBox(name)
            self.syn_components_checks[name] = check
            vbox.addWidget(check)
            
        self.on_syn_set_default_component_checks()
        self.on_syn_set_default_times(self.times_timeorphase.currentText())
                
            
        # show panel
        self.syn_timeWidget.setVisible(True)
        
        # this is now a synthetic dataset - hide the file chooser button
        self.dataTextEdit.setVisible(False)
        self.pfe_synChooserButton.setEnabled(False)
        self.pfe_fileChooserButton.setVisible(False)
        
    def on_timeselect_clicked(self):
        pop = CreatePopTimeSelect(self,self.parent().bundle)
        result = pop.exec_()
        
        name = str(self.sender().objectName())
        
        if result:
            # get the necessary widget
            w = self.findChildren(QDoubleSpinBox,'_'.join(name.split('_')[1:]))[0]
            w.setValue(pop.get_time())
        
    def on_file_choose(self):
        """
        this is called when the file selector button is clicked
        we need to create a filechooser popup and then set the gui based on the selected file
        """

        f = QFileDialog.getOpenFileName(self, 'Import Data...', self.parent().latest_dir if self.parent().latest_dir is not None else '.', **self.parent()._fileDialog_kwargs)
        if len(f)==0: #then no file selected
            return
 
        # store the directory used (to the main gui class)
        self.parent().latest_dir = os.path.dirname(str(f))
        
        # change the text of the button to say the filename
        self.pfe_fileChooserButton.setText(f)

        # make the widget that will show the contents visible
        self.dataTextEdit.clear()
        self.dataTextEdit.setVisible(True)

        # open and read the file
        fdata = open(f, 'r')
        data = fdata.readlines()

        # parse header and try to predict settings
        (columns,components,datatypes,units,ncols),(pbdep,dataset) = datasets.parse_header(str(f))
        if components is None:
            components = ['None']*ncols
        
        filterset,filterband = pbdep['passband'].split('.')[:]
        
        if filterset in self.filtertypes.keys():
            self.pfe_filtersetComboBox.setCurrentIndex(self.filtertypes.keys().index(filterset)+1)
            if filterband in self.filtertypes[filterset]:
                self.pfe_filterbandComboBox.setCurrentIndex(self.filtertypes[filterset].index(filterband)+1)
        
        # if no name has been provided, try to guess
        if self.name.text()=='':
            self.name.setText(pbdep['ref'])

        # remove any existing colwidgets
        if hasattr(self, 'colwidgets'): 
            # then we have existing columns from a previous file and need to remove them
            for colwidget in self.colwidgets:
                self.horizontalLayout_ColWidgets.removeWidget(colwidget)
                
        self.colwidgets = []
        for col in range(ncols):
            colwidget = CreatePopFileEntryColWidget(devel_version=self.devel_version)
            self.colwidgets.append(colwidget) # so we can iterate through these later
            self.horizontalLayout_ColWidgets.addWidget(colwidget)

            colwidget.col_comboBox.setVisible(False)

            colwidget.type_comboBox.addItems(self.datatypes)

            colwidget.comp_comboBox.addItems(self.parent().system_names)
            #~ colwidget.comp_comboBox.addItem('provided in another column')
            
            QObject.connect(colwidget.type_comboBox, SIGNAL("currentIndexChanged(QString)"), self.on_pfe_typeComboChanged)
            
            # try to guess the column types and settings
            if columns is not None and columns[col] in self.datatypes:
                colwidget.type_comboBox.setCurrentIndex(self.datatypes.index(columns[col]))
            if components is not None and components[col] in self.parent().system_names:
                colwidget.comp_comboBox.setCurrentIndex(self.parent().system_names.index(components[col])+1) #+1 because of header item
            

            colwidget.col_comboBox.setEnabled(False)
            colwidget.col_comboBox.clear()
            for col in range(ncols):
                colwidget.col_comboBox.addItem('Column %d' % (col+1))
            colwidget.col_comboBox.setEnabled(True)

        # show the first 100 lines of the data
        for i,line in enumerate(data):
            if i > 100: continue    
            if i==100: self.dataTextEdit.appendPlainText('...')
            if i<100:
                self.dataTextEdit.appendPlainText(line.strip())
                
        # show the reload button, hide the syn only button
        self.pfe_fileReloadButton.setVisible(True)
        self.pfe_synChooserButton.setVisible(False)
                
    def on_pfe_typeComboChanged(self):
        """
        when data column type is changed, this changes the options for the remaining combo boxes
        """
        combo = self.sender()
        if not combo.isEnabled():
            return
        selection = combo.currentText()
        colwidget = combo.parent()
        unitscombo = colwidget.units_comboBox
        compcombo = colwidget.comp_comboBox
        colcombo = colwidget.col_comboBox

        unitscombo.setEnabled(False)
        unitscombo.clear()
        unitscombo.setEnabled(True)

        if selection=='--Data Type--' or selection=='ignore' or selection=='component':
            unitscombo.addItems(['--Units--'])
            unitscombo.setEnabled(False)
        if selection=='time' or selection=='eclipse time':
            unitscombo.addItems(['BJD', 'HJD', 'Phase'])
        if selection=='wavelength':
            unitscombo.addItems(['Angstroms'])
        if selection=='flux':
            unitscombo.addItems(['Flux','Magnitude'])
        if selection=='rv':
            unitscombo.addItems(['m/s', 'km/s'])
        if selection=='o-c':
            unitscombo.addItems(['days', 'mins'])
        if selection=='sigma':
            unitscombo.addItems(['Standard Weight', 'Standard Deviation'])
            
        unitscombo.setEnabled(False) #TODO unit support

        if selection=='component':
            compcombo.setVisible(False)
            colcombo.setVisible(True)
        else:
            compcombo.setVisible(True)
            colcombo.setVisible(False)

        if selection != 'time' and selection != 'wavelength' and selection!="--Data Type--" and selection!='ignore': # x-axis types
            compcombo.setEnabled(True)
        else:
            compcombo.setCurrentIndex(0)
            compcombo.setEnabled(False)
        
    def on_category_changed(self,category):
        """
        this is called whenever the category combo is changed
        
        we need to change the column datatype combos to match the category
        and also need to set self.category so when ok is clicked we can call the correct command
        """
        
        #determine what type of data we're loading, and set options accordingly
        if category=='lc':
            self.datatypes = ['--Data Type--', 'time', 'flux', 'sigma']
            #~ self.datatypes = ['--Data Type--', 'time', 'flux', 'sigma', 'component', 'ignore']
        elif category=='rv':
            self.datatypes = ['--Data Type--', 'time', 'rv', 'sigma']
            #~ self.datatypes = ['--Data Type--', 'time', 'rv', 'sigma', 'component', 'ignore']
        elif category=='etv':
            self.datatypes = ['--Data Type--', 'time', 'o-c', 'sigma']
            #~ self.datatypes = ['--Data Type--', 'time', 'o-c', 'sigma', 'component', 'ignore']
        elif category=='sp':
            self.datatypes = ['--Data Type--', 'wavelength', 'flux', 'sigma']
            #~ self.datatypes = ['--Data Type--', 'wavelength', 'flux', 'sigma', 'component', 'ignore']
        else:
            return
            
        self.category = category
            
        for colwidget in self.colwidgets:
            colwidget.type_comboBox.clear()
            colwidget.type_comboBox.addItems(self.datatypes)
            
        self.on_syn_set_default_component_checks()

class CreatePopFileEntryColWidget(QWidget, gui.Ui_popFileEntryColWidget):
    def __init__(self, parent=None,devel_version=False):
        super(CreatePopFileEntryColWidget, self).__init__(parent)
        self.setupUi(self)
        
        if not devel_version:
            self.units_comboBox.setVisible(False)
        
class CreateFileEntryWidget(QWidget, gui.Ui_fileEntryWidget):
    def __init__(self, parent=None):
        super(CreateFileEntryWidget, self).__init__(parent)
        self.setupUi(self)
        
class CreateDatasetWidget(QWidget, gui.Ui_datasetWidget):
    def __init__(self, parent=None):
        super(CreateDatasetWidget, self).__init__(parent)
        self.setupUi(self)
        self.myparent=parent
