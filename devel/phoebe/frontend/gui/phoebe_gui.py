#!/usr/bin/python

import numpy as np
from glob import glob
from copy import deepcopy
from collections import OrderedDict
import os
import sys, json, imp, inspect, functools

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg

from phoebe.frontend.gui import ui_phoebe_pyqt4 as gui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

#phoebe modules
from phoebe.frontend import usersettings
from phoebe.frontend.gui import phoebe_plotting, phoebe_widgets

from phoebe.parameters import parameters, datasets
from phoebe.backend import universe, observatory
from phoebe.utils import callbacks, utils

### unity launcher
global _unityImport
try:
    from gi.repository import Unity
except ImportError:
    _unityImport = False
else:
    _unityImport = True

### global options
global _fileDialog_kwargs
_fileDialog_kwargs = {}
if 'kde' in os.environ.get('DESKTOP_SESSION',''): 
    # kde systems have terrible lag with the native dialog, so override
    _fileDialog_kwargs['options'] = QFileDialog.DontUseNativeDialog


global _alpha_test
_alpha_test = True


### functools
def PyInterp_selfdebug(fctn):
    @functools.wraps(fctn)
    def send(guiself,*args,**kwargs):
        """
        if the debug flag was set when running phoebe
        self will be available to the python interface through the class 'gui'
        
        any function that makes changes to values in self (not just gui state)
        should include this decorator
        """
        fctn(guiself,*args,**kwargs)
        if guiself.debug:
            guiself.PyInterp_send('gui',guiself)
        return
    return send


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
        
class CreatePopPrefs(QDialog, gui.Ui_popPrefs_Dialog):
    def __init__(self, parent=None):
        super(CreatePopPrefs, self).__init__(parent)
        self.setupUi(self)

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
        
### MAIN WINDOW
class PhoebeGUI(QMainWindow, gui.Ui_PHOEBE_MainWindow):
    def __init__(self, DEBUG=(False, False), selfdebug=False, font=None, palette=None, parent=None):
        """
        setup the gui
        connect signals
        initialize variables that need to be initialized at startup (most variables should be initialized in on_new_bundle)        
        """
        # DEBUG - any errors will be stolen by PyInterp (which might not be visible in the case of a crash)
        # when running from the command line add stderr and/or stdout in sys.argv to send to terminal instead of PyInterp

        super(PhoebeGUI, self).__init__(parent)
        self.setupUi(self)
        
        self.gui_dir = os.path.abspath(os.path.dirname(gui.__file__)) 
        
        self.debug = selfdebug
        self.font = font
        self.palette = palette
        
        self.resize(1220, 768)  # have to set size manually because the hidden widgets won't allow setting the vertical size this small
        self.mp_stackedWidget.setCurrentIndex(0) # just in case was changed in ui file
        
        if _unityImport:
            self.unity_launcher = Unity.LauncherEntry.get_for_desktop_id("phoebe_gui.desktop")
        else:
            self.unity_launcher = None
        
        # hide float and close buttons on dock widgets
        #~ self.bp_pyDockWidget.setTitleBarWidget(QWidget())
        self.bp_datasetsDockWidget.setTitleBarWidget(QWidget())

        # set default visibility for docks/frames
        self.mp_addPlotButtonsWidget.setVisible(False)
        self.lp_DockWidget.setVisible(False)
        self.rp_fittingDockWidget.setVisible(False)
        self.bp_pyDockWidget.setVisible(False)
        self.bp_datasetsDockWidget.setVisible(False)
        self.lp_systemDockWidget.setVisible(False)
        self.sb_StatusBar.setVisible(False)
        self.rp_fitoptionsWidget.setVisible(False)
        self.lp_observeoptionsWidget.setVisible(False)
        self.sys_orbitOptionsWidget.setVisible(False)
        self.sys_meshOptionsWidget.setVisible(False)
        self.mp_systemSelectWebViewWidget.setVisible(False) #hierarchy view
        self.rp_savedFeedbackTreeView.setVisible(False)
        self.rp_savedFeedbackAutoSaveCheck.setVisible(False)
        self.rp_stackedWidget.setCurrentIndex(0) #fit input
        
        # tabify left dock widgets
        self.tabifyDockWidget(self.lp_systemDockWidget, self.lp_DockWidget)
        
        # tabify right dock widgets
        self.tabifyDockWidget(self.rp_fittingDockWidget, self.rp_versionsDockWidget)
        
        # tabify bottom dock widgets
        self.tabifyDockWidget(self.bp_pyDockWidget, self.bp_datasetsDockWidget)
        
        # set inital states for widgets
        self.tb_file_saveasAction.setEnabled(False)
        self.lp_meshPushButton.setChecked(False)

        # create PyInterp in bottom panel tab
        self.PythonEdit = phoebe_widgets.PyInterp(self.bp_pyDockWidget, DEBUG)
        self.bp_pyLayout.addWidget(self.PythonEdit)
        self.PythonEdit.initInterpreter(locals())
               
        #~ ########### TEST PLUGINS ################
        #~ from plugins import keplereb
        #~ self.keplerebplugin = keplereb.GUIPlugin()
        #~ keplereb.GUIPlugin.enablePlugin(self.keplerebplugin, self, gui.Ui_PHOEBE_MainWindow)
        #~ keplereb.GUIPlugin.disablePlugin(self.keplerebplugin)
        #~ 
        #~ from plugins import example
        #~ self.exampleplugin = example.GUIPlugin()
        #~ example.GUIPlugin.enablePlugin(self.exampleplugin, self, gui.Ui_PHOEBE_MainWindow)
        #################################

        # add master dataset tree view to bottom panel 
        self.datasetswidget_main = CreateDatasetWidget()
        self.bp_datasetsDockWidget.setWidget(self.datasetswidget_main)
        
        # Add the OpenGL mesh_widget to gui
        self.mesh_widget = phoebe_widgets.mesh_widget(self, None)
        self.mp_glLayout.addWidget(self.mesh_widget,0,0)
        self.mesh_widget.setMinimumWidth(300)
        self.mesh_widget.setMinimumHeight(140)
        
        # Load webviews
        self.mp_sysSelWebView = phoebe_widgets.SysWebView() # for some ridiculous reason, if we create the webview in designer we can't send it jsmessenger
        vbox = QVBoxLayout()
        vbox.addWidget(self.mp_sysSelWebView)
        self.mp_systemSelectWebViewWidget.setLayout(vbox)
        
        self.mp_sysEditWebView = phoebe_widgets.SysWebView()
        vbox = QVBoxLayout() 
        vbox.addWidget(self.mp_sysEditWebView)
        self.webviewwidget.setLayout(vbox)
        
        self.jsmessenger = phoebe_widgets.JavaScriptMessenger() 
        self.mp_sysSelWebView.page().mainFrame().addToJavaScriptWindowObject("messenger", self.jsmessenger)
        self.mp_sysEditWebView.page().mainFrame().addToJavaScriptWindowObject("messenger", self.jsmessenger)
        # set font to match qt
        #~ self.mp_sysSelWebView.page().mainFrame().evaluateJavaScript("$('li').css('font-family',%s)" % str(font.family()))
        
        self.mp_sysSelWebView.load(QUrl(os.path.join(self.gui_dir, "html/phoebe_sysselect.html")))
        self.mp_sysEditWebView.load(QUrl(os.path.join(self.gui_dir, "html/phoebe_sysedit.html")))
        
        # Create canvas for expanded plot
        self.expanded_plot_widget, self.expanded_canvas = self.create_plot_widget()
        expanded_plot = CreatePopPlot()
        expanded_plot.xaxisCombo.axes_i = 'expanded'
        expanded_plot.yaxisCombo.axes_i = 'expanded'
        expanded_plot.titleLinkButton.axes_i = 'expanded'
        expanded_plot.title_cancelButton.axes_i = 'expanded'
        expanded_plot.title_saveButton.axes_i = 'expanded'
        self.mp_expandLayout.addWidget(expanded_plot.plot_Widget,0,0)
        expanded_plot.plot_gridLayout.addWidget(self.expanded_plot_widget,0,0)
        self.expanded_plot = expanded_plot
      
        # Create undo stack
        self.undoStack = QUndoStack(self)
        
        # Connect signals
        # menu bar signals
        QObject.connect(self.tb_file_newAction, SIGNAL("activated()"), self.on_new_clicked)
        QObject.connect(self.tb_fileImport_libraryAction, SIGNAL("activated()"), self.on_open_clicked)
        QObject.connect(self.actionLegacy_PHOEBE, SIGNAL("activated()"), self.on_open_clicked)
        QObject.connect(self.tb_file_openAction, SIGNAL("activated()"), self.on_open_clicked)
        QObject.connect(self.mp_splash_openPushButton, SIGNAL("clicked()"), self.on_open_clicked)
        QObject.connect(self.tb_file_saveAction, SIGNAL("activated()"), self.on_save_clicked)        
        QObject.connect(self.tb_file_saveasAction, SIGNAL("activated()"), self.on_save_clicked) 
        QObject.connect(self.tb_view_pythonAction, SIGNAL("toggled(bool)"), self.on_view_python_toggled)       
        QObject.connect(self.tb_view_datasetsAction, SIGNAL("toggled(bool)"), self.on_view_datasets_toggled)       
        #~ QObject.connect(self.tb_view_versionsAction, SIGNAL("toggled(bool)"), self.on_view_versions_toggled)       
        QObject.connect(self.tb_edit_prefsAction, SIGNAL("activated()"), self.on_prefsShow)
        QObject.connect(self.tb_help_aboutAction, SIGNAL("activated()"), self.on_aboutShow)
        QObject.connect(self.tb_help_helpAction, SIGNAL("activated()"), self.on_helpShow)
        QObject.connect(self.tb_edit_undoAction, SIGNAL("activated()"), self.on_undo_clicked)
        QObject.connect(self.tb_edit_redoAction, SIGNAL("activated()"), self.on_redo_clicked)
        QObject.connect(self.tb_tools_scriptAction, SIGNAL("activated()"), self.on_load_script)
        
        # left panel signals
        QObject.connect(self.lp_previewPushButton, SIGNAL("clicked()"), self.on_observe_clicked)
        QObject.connect(self.lp_computePushButton, SIGNAL("clicked()"), self.on_observe_clicked)
        QObject.connect(self.lp_progressQuit, SIGNAL("clicked()"), self.cancel_thread)
        
        QObject.connect(self.sys_orbitPushButton, SIGNAL("clicked()"), self.on_orbit_update_clicked)
        QObject.connect(self.sys_meshPushButton, SIGNAL("clicked()"), self.on_mesh_update_clicked)
        QObject.connect(self.sys_meshAutoUpdate, SIGNAL("toggled(bool)"), self.on_mesh_update_auto_toggled)

        # middle panel signals
        QObject.connect(self.mp_sysSelWebView, SIGNAL("ctrl_pressed"), self.on_sysSel_ctrl)
        QObject.connect(self.mp_sysSelWebView, SIGNAL("ctrl_released"), self.on_sysSel_ctrlReleased)
        if not _alpha_test:
            QObject.connect(self.jsmessenger, SIGNAL("editClicked"), self.on_systemEdit_clicked)
        QObject.connect(self.jsmessenger, SIGNAL("selectionUpdate"), self.on_sysSel_selectionChanged)
        QObject.connect(self.mpsys_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle)
        QObject.connect(self.mpgl_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle)
        QObject.connect(self.mp_addPlotLabelWidget, SIGNAL("hover_enter"), self.on_plot_add_hover_enter)
        QObject.connect(self.mp_addPlotLabelWidget, SIGNAL("hover_leave"), self.on_plot_add_hover_leave)
        QObject.connect(self.mp_addLCPlotPushButton, SIGNAL("clicked()"), self.on_plot_add)
        QObject.connect(self.mp_addRVPlotPushButton, SIGNAL("clicked()"), self.on_plot_add)
        QObject.connect(self.mp_addSPPlotPushButton, SIGNAL("clicked()"), self.on_plot_add)
        QObject.connect(self.datasetswidget_main.datasetTreeView, SIGNAL("axes_add"), self.on_axes_add)
        QObject.connect(self.datasetswidget_main.datasetTreeView, SIGNAL("axes_goto"), self.on_axes_goto)
        QObject.connect(self.mp_splash_binaryPushButton, SIGNAL("clicked()"), self.splash_binary)
        QObject.connect(self.mp_splash_triplePushButton, SIGNAL("clicked()"), self.splash_triple)
        QObject.connect(expanded_plot.popPlot_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle) 
        QObject.connect(expanded_plot.popPlot_popPushButton, SIGNAL("clicked()"), self.on_plot_pop)
        QObject.connect(expanded_plot.popPlot_delPushButton, SIGNAL("clicked()"), self.on_plot_del)
        QObject.connect(expanded_plot.xaxisCombo, SIGNAL("currentIndexChanged(QString)"), self.on_plot_xaxis_changed)
        QObject.connect(expanded_plot.yaxisCombo, SIGNAL("currentIndexChanged(QString)"), self.on_plot_yaxis_changed)
        QObject.connect(expanded_plot.titleLinkButton, SIGNAL("clicked()"), self.on_plot_title_clicked)
        QObject.connect(expanded_plot.title_cancelButton, SIGNAL("clicked()"), self.on_plot_title_cancel)
        QObject.connect(expanded_plot.title_saveButton, SIGNAL("clicked()"), self.on_plot_title_save)
        
        # right panel signals     
        QObject.connect(self.rp_methodComboBox, SIGNAL("currentIndexChanged(QString)"), self.on_fittingOption_changed)
        QObject.connect(self.rp_fitPushButton, SIGNAL("clicked()"), self.on_fit_clicked)
        QObject.connect(self.rp_rejectPushButton, SIGNAL("clicked()"), self.on_feedback_reject_clicked)
        QObject.connect(self.rp_acceptPushButton, SIGNAL("clicked()"), self.on_feedback_accept_clicked)
        
        QObject.connect(self.rp_savedFeedbackAutoSaveCheck, SIGNAL("toggled(bool)"), self.on_feedbacksaveauto_toggled)
        
        QObject.connect(self.versions_oncompute, SIGNAL("toggled(bool)"), self.on_versionsauto_toggled)
        QObject.connect(self.versions_addnow, SIGNAL("clicked()"), self.on_versionsadd_clicked)
                
        # bottom panel signals
        QObject.connect(self.datasetswidget_main.ds_typeComboBox, SIGNAL("currentIndexChanged(QString)"), self.update_datasets)
        QObject.connect(self.datasetswidget_main.ds_plotComboBox, SIGNAL("currentIndexChanged(QString)"), self.update_datasets)
        QObject.connect(self.datasetswidget_main.addLCButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        QObject.connect(self.datasetswidget_main.addRVButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        QObject.connect(self.datasetswidget_main.addSPButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        QObject.connect(self.datasetswidget_main.addETVButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        self.datasetswidget_main.addETVButton.setEnabled(False)
        
        # tree view signals
        self.paramTreeViews = [self.lp_compTreeView,self.lp_orbitTreeView, self.lp_meshTreeView, self.rp_fitinTreeView, self.rp_fitoutTreeView, self.rp_fitoptionsTreeView, self.lp_observeoptionsTreeView, self.datasetswidget_main.datasetTreeView, self.versions_treeView, self.rp_savedFeedbackTreeView, self.sys_orbitOptionsTreeView, self.sys_meshOptionsTreeView]
        for tv in self.paramTreeViews:
            QObject.connect(tv, SIGNAL("parameterChanged"), self.on_param_changed)
            QObject.connect(tv, SIGNAL("parameterCommand"), self.on_param_command)
            QObject.connect(tv, SIGNAL("focusIn"), self.on_paramfocus_changed)
        QObject.connect(self.rp_fitinTreeView, SIGNAL("priorChanged"), self.on_prior_changed)  
        QObject.connect(self.rp_savedFeedbackTreeView, SIGNAL("feedbackExamine"), self.on_feedback_changed) 

        # pyinterp signals
        QObject.connect(self.PythonEdit, SIGNAL("command_run"), self.on_PyInterp_commandrun)
        QObject.connect(self.PythonEdit, SIGNAL("GUILock"), self.gui_lock)
        QObject.connect(self.PythonEdit, SIGNAL("GUIUnlock"), self.gui_unlock) 
        QObject.connect(self.PythonEdit, SIGNAL("set_time"), self.on_set_time)
        #~ QObject.connect(self.PythonEdit, SIGNAL("set_select_time"), self.on_select_time_changed)
        QObject.connect(self.PythonEdit, SIGNAL("undo"), self.on_undo_clicked) 
        QObject.connect(self.PythonEdit, SIGNAL("redo"), self.on_redo_clicked) 
        QObject.connect(self.PythonEdit, SIGNAL("plots_changed"), self.on_plots_changed) 

        # load settings
        self.pluginsloaded = []
        self.plugins = []
        self.guiplugins = []
        self.on_prefsLoad() #load to self.prefs
        self.on_prefsUpdate() #apply to gui
        self.latest_dir = None
        
        # send startup commands to interpreter
        for line in self.prefs.get_setting('pyinterp_startup_default').split('\n'):
            self.PyInterp_run(line, write=True, thread=False)
        for line in self.prefs.get_setting('pyinterp_startup_custom').split('\n'):
            self.PyInterp_run(line, write=True, thread=False)
            
        # disable items for alpha version
        if _alpha_test:
            #~ self.rp_fitPushButton.setEnabled(False)
            self.mp_splash_triplePushButton.setEnabled(False)

        # Set system to None - this will then result in a call to on_new_bundle
        # any additional setup should be done there
        self.PyInterp_run('bundle = Bundle()',kind='sys',thread=False)
        
    def str_includes(self,string,lst):
        for item in lst:
            if item in string:
                return True
        return False
    
    def gui_lock(self,command,thread=None):
        """
        lock the gui while commands are being run from the python console
        """
        self.gui_locked=True
        self.current_thread=thread
        self.lp_mainWidget.setEnabled(False) #main instead of dock so progressbar is enabled
        self.lp_observeoptionsWidget.setEnabled(False) #outside of mainWidget
        self.rp_fittingDockWidget.setEnabled(False)
        self.bp_pyDockWidget.setEnabled(False)
        self.bp_datasetsDockWidget.setEnabled(False)
        self.rp_versionsDockWidget.setEnabled(False)
        self.centralwidget.setEnabled(False)
        
        self.set_time_i = 0
        # should be deciding based on the type of call made
        #~ self.mp_progressStackedWidget.setCurrentIndex(1) #progress bar
        if self.str_includes(command, ['set_time','observe','run_compute']):
            self.lp_progressStackedWidget.setCurrentIndex(1) #progress bar
        if self.str_includes(command, ['run_fitting']):
            #~ self.lp_progressStackedWidget.setCurrentIndex(1) #progress bar
            self.rp_progressStackedWidget.setCurrentIndex(1) #progress bar
    
    @PyInterp_selfdebug
    def gui_unlock(self):
        """
        unlock to gui to full responsive state
        this should undo anything done by gui_lock
        """
        self.gui_locked=False
        self.current_thread = None
        self.lp_mainWidget.setEnabled(True)
        self.lp_observeoptionsWidget.setEnabled(True)
        self.rp_fittingDockWidget.setEnabled(True)
        self.bp_pyDockWidget.setEnabled(True)
        self.bp_datasetsDockWidget.setEnabled(True)
        self.rp_versionsDockWidget.setEnabled(True)
        self.centralwidget.setEnabled(True)
        
        # can't hurt to reset all
        self.mp_progressStackedWidget.setCurrentIndex(0) #plot buttons
        self.lp_progressStackedWidget.setCurrentIndex(0) #plot buttons
        self.rp_progressStackedWidget.setCurrentIndex(0) #plot buttons
        
        self.mp_progressBar.setValue(0)
        self.lp_progressBar.setValue(0)
        self.rp_progressBar.setValue(0)
        
        if self.unity_launcher is not None:
            self.unity_launcher.set_property("progress_visible", False)
            #~ if self.set_time_is is not None: #then we were running a computation
                #~ self.unity_launcher.set_property("urgent", True)
        
        self.set_time_i, self.set_time_is = None, None
             
        self.PythonEdit.setFocus()
        
    def cancel_thread(self):
        if self.current_thread is not None:
            # clean up signals
            QObject.disconnect(self.current_thread, SIGNAL("threadComplete"), self.PythonEdit.update_from_thread)
            QObject.disconnect(self.current_thread, SIGNAL("threadFailed"), self.PythonEdit.update_from_threadfailed)
            QObject.disconnect(self.current_thread.buff, SIGNAL("newtext"), self.PythonEdit.update_textfromthread)
            QObject.disconnect(self.current_thread, SIGNAL("set_time"), self.PythonEdit.on_set_time)

            # kill thread
            self.current_thread.terminate()
            
            # force new line in python console
            #~ self.PythonEdit.write('\n')
            
            # clear any partial synthetic datasets to avoid plotting issues and inconsistencies
            self.bundle.system.clear_synthetic()
           
            # and force redraw plots to clear all models
            self.on_plots_changed()
            
            # restore gui
            self.gui_unlock()
        
    def on_undo_clicked(self):
        self.undoStack.undo()
        
    def on_redo_clicked(self):
        self.undoStack.redo()

    def on_new_clicked(self):
        self.PyInterp_run('bundle = Bundle()',kind='sys',thread=False) # this will call on_new_bundle and reset to home/splash screen

    @PyInterp_selfdebug
    def on_new_bundle(self):
        """
        reset any variables, widgets, plots etc when a new bundle is loaded
        this is also run on startup (although is the last thing done)        
        """
        #will be run whenever bundle = found in command line
        #which should also be triggered when on_new_clicked
        
        # reset state
        self.undoStack.clear()
        try:
            self.on_plot_clear_all() # this needs to be called before resetting lists
        except AttributeError: #will fail on startup because the lists haven't been initialized yet and there are no plots
            pass

        self.plotEntry_axes_i = [None]
        
        
        self.datasetswidget_main.datasetTreeView.clear()
        self.datasetswidget_main.ds_plotComboBox.clear()
        
        self.gui_locked=False
        self.plot_widgets = []
        self.plot_canvases = []
        self.plotEntry_widgets = [self.datasetswidget_main.datasetTreeView]
        self.plotEntry_axes_i = [None]
        self.attached_plot_signals = [] # pairs of parameter, canvas
        self.meshmpl_widget, self.meshmpl_canvas = None, None
        self.orbitmpl_widget, self.orbitmpl_canvas = None, None
        self.on_plot_clear_all()
        self.on_plot_add(mesh=True)
        self.on_plot_add(orbit=True)
        self.set_time_i, self.set_time_is = None, None
        self.current_feedback_name = None
        
        self.pop_i = None

        if self.bundle.system is None: #then we have no bundle and should be on the splash
            self.filename = None
            self.tb_file_saveAction.setEnabled(False)
            self.tb_file_saveasAction.setEnabled(False)
            
            self.mp_stackedWidget.setCurrentIndex(0)
        
            self.lp_DockWidget.setVisible(False)
            self.rp_fittingDockWidget.setVisible(False)
        
        else:        
            self.bundle.attach_signal(self.bundle,'add_axes',self.on_plots_add)
            self.bundle.attach_signal(self.bundle,'remove_axes',self.on_plots_remove)
            self.bundle.attach_signal(self.bundle,'load_data',self.update_datasets)
            self.bundle.attach_signal(self.bundle,'add_fitting',self.update_fittingOptions)
            self.bundle.attach_signal(self.bundle,'remove_fitting',self.update_fittingOptions)
            #~ self.bundle.attach_signal(self.bundle,'set_select_time',self.on_select_time_changed)
            #~ print "*** attaching set_time signal"
            #~ self.bundle.attach_signal(self.bundle.system,'set_time',self.on_set_time)

        self.update_system()
        self.on_plots_changed()
        self.update_fittingOptions()
        
    def on_load_script(self):
        """
        parse a file and send it through the python console
        TODO - does not work yet, menu item is currently disabled until working
        """
        filename = QFileDialog.getOpenFileName(self, 'Load Python Script', self.latest_dir if self.latest_dir is not None else './', "*.py", **_fileDialog_kwargs)
        self.PyInterp_run('print "not implemented yet"', write=False)
        return
        
        if len(filename) > 0:
            self.latest_dir = os.path.dirname(str(filename))
            f = open(filename, "r")
            #~ self.PyInterp_run(''.join(f.readlines()))
        
            from time import sleep
            for line in f:
                while self.gui_locked:
                    print "gui locked"
                    sleep(1)
                    pass
                self.PythonEdit.write(line.strip('\n'))
                self.PythonEdit.keyPressEvent(event='run')
                
    def on_open_clicked(self):
        """
        handle opening phoebe or par files and loading the bundle
        if successful, this will result in calling on_new_bundle
        """
        if self.sender() == self.tb_fileImport_libraryAction:
            filename = QFileDialog.getOpenFileName(self, 'From Library', '../../parameters/library/', ".par(*.par)", **_fileDialog_kwargs)  #need to choose this directory more carefully
        else:
            filename = QFileDialog.getOpenFileName(self, 'Open File', self.latest_dir if self.latest_dir is not None else './', "phoebe(*.phoebe *.par)", **_fileDialog_kwargs)
            if len(filename)>0: self.latest_dir = os.path.dirname(str(filename))
        if len(filename) > 0:
            self.tb_file_saveAction.setEnabled(True)
            self.tb_file_saveasAction.setEnabled(True)
            if self.sender()==self.tb_fileImport_libraryAction or filename.split('.')[-1]=='par':
                #~ path = os.path.basename(str(filename)).strip('.par') if 'parameters/library' in str(filename) else str(filename)
                path = os.path.basename(".".join(str(filename).split('.')[:-1])) if 'parameters/library' in str(filename) else str(filename)
                self.PyInterp_run('bundle = Bundle(system=create.from_library(\'%s\', create_body=True))' % path, kind='sys', thread=False)
                self.filename = None
                return
            if self.sender()==self.actionLegacy_PHOEBE:
                self.PyInterp_run('bundle = Bundle(system=parsers.legacy_to_phoebe(\'%s\', create_body=True))' % filename, kind='sys', thread=False)
                return
            self.PyInterp_run('bundle = load(\'%s\')' % filename, kind='sys', thread=False)
            self.filename = filename
                
    def on_save_clicked(self):
        """
        save a bundle to a .phoebe file
        this function handles both save and save as
        """
        if self.sender() == self.tb_file_saveAction and self.filename is not None:
            filename = self.filename
        else:
            filename = QFileDialog.getSaveFileName(self, 'Save File', self.latest_dir if self.latest_dir is not None else './', ".phoebe(*.phoebe)", **_fileDialog_kwargs)
        if len(filename) > 0:
            self.latest_dir = os.path.dirname(str(filename))
            self.PyInterp_run('bundle.save(\'%s\')' % filename, kind='sys', thread=False)
            self.filename = filename
            #~ self.tb_file_saveAction.setEnabled(True)
            
        # signals (including those to redraw plots) will have been purged during saving
        # for now we'll get around this by forcing the signals to be reconnected by redrawing all plots
        self.on_plots_changed()
            
    def on_view_python_toggled(self, truth):
        self.bp_pyDockWidget.setVisible(truth)
        
    def on_view_datasets_toggled(self, truth):
        self.bp_datasetsDockWidget.setVisible(truth)
        
    #~ def on_view_versions_toggled(self, truth):
        #~ self.rp_versionsDockWidget.setVisible(truth)

    def splash_binary(self):
        """
        load the default binary
        this will result in a call to on_new_bundle        
        """
        #this will obviously be reworked to be more general
        self.PyInterp_run('bundle = Bundle(system = create.from_library(\'HV2241\', create_body=True))', kind='sys', thread=False) 
    
    def splash_triple(self):
        """
        load the default triple
        this will result in a call to on_new_bundle
        """
        #this will obviously be reworked to be more general
        self.PyInterp_run('bundle = Bundle(system=create.KOI126())', write=True)

    def get_system_structure(self, struclabel=None, strucsel=None):
        """
        get the hierarchical structure of the system while maintaining the selected items
        this gets passed on to the html system views
        """
        curr_struclabel = list(utils.traverse(struclabel)) #flattened lists of the incoming values before updating
        curr_strucsel = list(utils.traverse(strucsel))
        
        if self.bundle.system is not None:
            strucnames, strucnchild, strucsel, strucps = self.bundle.get_system_structure(return_type=['label','nchild','mask','ps'],old_mask=(curr_struclabel, curr_strucsel))
        else:
            strucnames, strucnchild, strucsel, strucps = [[]],[[]],[[]],[[]]
            
        return strucnames, strucnchild, strucsel, strucps
             
    @PyInterp_selfdebug
    def update_system(self):
        """
        this should be called whenever a change to the system structure is detected or possible
        this will recreate the html system views (maintaining previous selection)
        """
        #~ print "*** update_system"
        self.dataview_refs = [] # will hold all refs in all dataviews
        self.dataview_treeitems = []
        self.dataview_treeitemnames = []
        
        strucnames, strucnchild, strucsel, strucps = self.get_system_structure(struclabel=self.jsmessenger.sysitems_flat, strucsel=self.jsmessenger.sysitems_sel_flat)
        self.system_names = list(utils.traverse(strucnames))
        self.system_ps = list(utils.traverse(strucps))

        self.jsmessenger.sysitems = strucnames
        self.jsmessenger.sysitems_flat = list(utils.traverse(strucnames))
        self.jsmessenger.sysitems_ps = strucps
        self.jsmessenger.sysitems_nchild = strucnchild
        self.jsmessenger.sysitems_sel = strucsel
        self.jsmessenger.sysitems_sel_flat = list(utils.traverse(strucsel))
        
        self.mp_sysSelWebView.page().mainFrame().evaluateJavaScript("PHOEBE.reset('select','from_messenger','None','%s','%s','%s')" % (self.font.family(),self.palette.color(QPalette.Window).name(),self.palette.color(QPalette.Highlight).name()))
        self.mp_sysEditWebView.page().mainFrame().evaluateJavaScript("PHOEBE.reset('edit','from_messenger','None','%s','%s','%s')" % (self.font.family(),self.palette.color(QPalette.Window).name(),self.palette.color(QPalette.Highlight).name()))

        self.on_sysSel_selectionChanged(skip_collapse=True)
        
    @PyInterp_selfdebug
    def update_datasets(self,*args):
        """
        this updates the bottom panel datasets treeview
        this should be called whenever there is a change to the system, datasets, axes, or plotoptions
        currently this is called after any command is sent through the python console
        """
        if self.bundle.system is None:
            return
        # get obs and syn
        ds_obs_all = self.bundle.get_obs()
        ds_syn_all = self.bundle.get_syn()
        
        
        # remove duplicates
        ds_obs = []
        ds_syn = []
        
        for dso in ds_obs_all:
            if dso['ref'] not in [dsor['ref'] for dsor in ds_obs]:
                ds_obs.append(dso)
        for dss in ds_syn_all:
            if dss['ref'] not in [dssr['ref'] for dssr in ds_syn]:
                ds_syn.append(dss)
        
        # will eventually want to loop over all treeviews (main and in plot pops)
        trees = [self.datasetswidget_main.datasetTreeView]
        
        for tree in self.plotEntry_widgets:
            # filter which items to show
            if tree == self.datasetswidget_main.datasetTreeView:
                #then filter based off combos
                types = str(self.datasetswidget_main.ds_typeComboBox.currentText())
                plots = str(self.datasetswidget_main.ds_plotComboBox.currentText())
                if plots != 'all plots':
                    self.datasetswidget_main.ds_typeComboBox.setEnabled(False)
                else:
                    self.datasetswidget_main.ds_typeComboBox.setEnabled(True)
            else:
                types = 'all types'
                plots = tree.plotindex
            
            tree.set_data(ds_obs,ds_syn,types,plots,self.bundle,self.system_ps,self.system_names)

    @PyInterp_selfdebug
    def on_sysSel_selectionChanged(self,skip_collapse=False):
        """
        this function is called when the selection in the html view is changed
        this then sends the selected objects to the left panel treeviews and rebuilds them
        
        if skip_collapse is False, the treeviews will intelligently be hidden or shown
        if skip_collapse is True, no changes will be made to visibility (data will still be refreshed)
        """
        # flatten all parametersets for comparison
        ps_all = list(utils.traverse(self.jsmessenger.sysitems_ps))
        # apply mask to get selected parametersets
        ps_sel = [ps_all[i] for i in range(len(ps_all)) if self.jsmessenger.sysitems_sel_flat[i]]
        
        # now split into orbits and components
        sel_orbits = [s for s in ps_sel if s.context=='orbit']
        sel_comps = [s for s in ps_sel if s.context in ['component','star']]
        sel_meshes = [self.bundle.get_mesh(s.get_value('label')) for s in sel_comps] # this is kind of ugly

        #~ self.lp_compTreeView.setColumnCount(len(sel_comps)+1)
        #~ self.lp_compTreeView.resizeColumnToContents(0)

        #~ self.lp_orbitTreeView.setColumnCount(len(sel_orbits)+1)
        #~ self.lp_orbitTreeView.resizeColumnToContents(0)
        
        # Collapse unused tree views
        if len(sel_comps) == 0:
            if not skip_collapse:
                self.lp_compPushButton.setChecked(False)
            self.lp_compTreeView.clear() #if not, the cleared first column might still be populated
            self.lp_meshTreeView.clear()
        else:
            if not skip_collapse:
                self.lp_compPushButton.setChecked(True)

            #update component tree view
            self.lp_compTreeView.set_data(sel_comps)
            self.lp_meshTreeView.set_data(sel_meshes,style=['nofit'])

        if len(sel_orbits) == 0:
            if not skip_collapse:
                self.lp_orbitPushButton.setChecked(False)
            self.lp_orbitTreeView.clear()
        else:
            if not skip_collapse:
                self.lp_orbitPushButton.setChecked(True)

            #update orbit tree view
            self.lp_orbitTreeView.set_data(sel_orbits)

        for i,item in enumerate(sel_orbits):
            self.lp_orbitTreeView.headerItem().setText(i+1, item['label'])
            self.lp_orbitTreeView.resizeColumnToContents(i+1)

        for i,item in enumerate(sel_comps):
            self.lp_compTreeView.headerItem().setText(i+1, item['label'])
            self.lp_compTreeView.resizeColumnToContents(i+1)
            self.lp_meshTreeView.headerItem().setText(i+1, item['label'])
            self.lp_meshTreeView.resizeColumnToContents(i+1)
            
        if self.bundle.system is not None:
            #~ print "*** updating fitting treeviews", len(self.bundle.system.get_adjustable_parameters())
            self.rp_fitinTreeView.set_data(self.bundle.system.get_adjustable_parameters(),self.system_ps,self.system_names)
            self.rp_fitoutTreeView.set_data(self.bundle.get_feedback(-1),self.system_ps,self.system_names)
        else:
            self.rp_fitinTreeView.set_data([],self.system_ps,self.system_names)
            self.rp_fitoutTreeView.set_data({},self.system_ps,self.system_names)
        
    def on_sysSel_ctrl(self):
        """
        remembers that ctrl is pressed for multiple selection in the html system views
        """
        self.jsmessenger.ctrl=True
        
    def on_sysSel_ctrlReleased(self):
        """
        remembers that ctrl was released for single selection in the html system views
        """
        self.jsmessenger.ctrl=False
            
    def on_plot_add_hover_enter(self):
        self.mp_addPlotButtonsWidget.setVisible(True)
        self.mp_addPlotLabel.setVisible(False)

    def on_plot_add_hover_leave(self):
        self.mp_addPlotLabel.setVisible(True)
        self.mp_addPlotButtonsWidget.setVisible(False)
        
    def attach_plot_signals(self, axes, i=0, canvas=None, skip_axes_attach=False):
        #~ print "*** attach_plot_signals", i
        if canvas is None:
            canvas = self.plot_canvases[i]
           
        for po in axes.plots:
            if (po, canvas) not in self.attached_plot_signals:
                for paramname in po.keys():
                    param = po.get_parameter(paramname)
                    #~ self.bundle.purge_signals([param]) # what do we want to purge is the question
                    self.bundle.attach_signal(param, 'set_value', self.plot_redraw, i, canvas)
                self.attached_plot_signals.append((po, canvas))
        #~ if (axes, canvas) not in self.attached_plot_signals:
        if not skip_axes_attach:
            self.bundle.attach_signal(self.bundle, 'set_select_time', self.on_select_time_changed, i, canvas)
            self.bundle.attach_signal(self.bundle, 'set_system', self.plot_redraw, i, canvas) #will also be called for get_version(set_system=True)
            self.bundle.attach_signal(axes.axesoptions, 'set_value', self.plot_redraw, i, canvas)
            #~ self.bundle.attach_signal(axes, 'set_value', self.plot_redraw, i, canvas)
            self.bundle.attach_signal(axes, 'add_plot', self.attach_plot_signals, i, canvas, True) # so that we can add the new parameter options
            self.bundle.attach_signal(axes, 'add_plot', self.plot_redraw, i, canvas)
            self.bundle.attach_signal(axes, 'remove_plot', self.plot_redraw, i, canvas)
            #~ self.attached_plot_signals.append((axes, canvas))
        else: # then we're coming from add_plot
            self.plot_redraw(None, i, canvas)
            
    def update_plot_widgets(self, i, canvas):
        if not hasattr(canvas,'info'):
            #~ print '* update_plot_widgets exiting'
            return
        canvas.info['xaxisCombo'].setEnabled(False)
        canvas.info['xaxisCombo'].clear()
        items = ['time']
        for name in self.bundle.get_system_structure(flat=True):
            ps = self.bundle.get_ps(name)
            if ps.context=='orbit':
                items.append('phase:%s' % name)
        canvas.info['xaxisCombo'].addItems(items)
        canvas.info['xaxisCombo'].setCurrentIndex(items.index(self.bundle.axes[i].get_value('xaxis')) if self.bundle.axes[i].get_value('xaxis') in items else 0)
        canvas.info['xaxisCombo'].setEnabled(True)
        
        #~ canvas.info['yaxisCombo'].setCurrentIndex()

        canvas.info['titleLinkButton'].setText(self.bundle.axes[i].get_value('title'))
            
    def plot_redraw(self, param=None, i=0, canvas=None):
        #~ print "*** redraw plot", i
        if param is not None and len(self.plot_canvases) != len(self.bundle.axes):
            #then this is being called from a signal, but the number of canvases isn't right
            #so redraw all to make sure we're in sync
            self.on_plots_changed()
        else:
            if canvas is None:
                canvas = self.plot_canvases[i]
            else: # not a thumbnail
                # update widgets corresponding to this canvas
                self.update_plot_widgets(i, canvas)
            
            canvas.cla()
            canvas.plot(self.bundle, self.bundle.axes[i])
            canvas.draw()
            
            if i==self.pop_i: # then this plot is in the expanded plot and we should also draw that
                self.expanded_canvas.cla()
                self.expanded_canvas.plot(self.bundle, self.bundle.axes[i])
                self.expanded_canvas.draw()

    def on_plot_add(self, mesh=False, orbit=False, plotoptions=None):
        # add button clicked from gui
        if mesh==True and self.meshmpl_widget is None:
            self.meshmpl_widget, self.meshmpl_canvas = self.create_plot_widget(thumb=True)
            #~ self.meshmpl_widget.setMaximumWidth(self.meshmpl_widget.height())
            self.mp_sysmplGridLayout.addWidget(self.meshmpl_widget, 0,0)
        elif orbit==True and self.orbitmpl_widget is None:
            self.orbitmpl_widget, self.orbitmpl_canvas = self.create_plot_widget(thumb=True)
            self.sys_orbitmplGridLayout.addWidget(self.orbitmpl_widget, 0,0)
            
        else:
            if plotoptions is None:
                plottype='lcobs'
                if self.sender()==self.mp_addLCPlotPushButton:
                    plottype='lcobs'
                if self.sender()==self.mp_addRVPlotPushButton:
                    plottype='rvobs'
                if self.sender()==self.mp_addSPPlotPushButton:
                    plottype='spobs'
            add_command = 'bundle.add_axes(category=\'%s\', title=\'Plot %d\')' % (plottype[:-3],len(self.bundle.axes)+1)
            remove_command = 'bundle.remove_axes(%d)' % (len(self.bundle.axes))
            command = phoebe_widgets.CommandRun(self.PythonEdit,add_command,remove_command,kind='plots',thread=False,description='add new plot')
            
            self.undoStack.push(command) 
            
    def create_plot_widget(self, canvas=None, thumb=False):
        new_plot_widget = QWidget()
        if canvas is None:
            canvas = phoebe_plotting.MyMplCanvas(new_plot_widget, width=5, height=4, dpi=100, thumb=thumb, bg=str(self.palette.color(QPalette.Window).name()), hl=str(self.palette.color(QPalette.Highlight).name()))
        if thumb:
            QObject.connect(canvas, SIGNAL("plot_clicked"), self.on_plot_expand_toggle)
        else:
            QObject.connect(canvas, SIGNAL("plot_clicked"), self.on_expanded_plot_clicked)
        vbox = QVBoxLayout()
        vbox.addWidget(canvas)
        if not thumb:
            pass
            # create a navigation toolbar but don't show it 
            # we'll create and connect our own buttons
            #~ navigation = NavigationToolbar2QTAgg(canvas, None)
            #~ new_plot_widget.navigation = navigation
        new_plot_widget.setLayout(vbox)
        return new_plot_widget, canvas

    def on_plot_clear_all(self):    
        # delete all canvases and widgets from the gui to prepare to redraw
        for i,widget in enumerate(self.plot_widgets):
            widget.close()
        self.plot_widgets = []
        self.plot_canvases = []   
        
    def on_plot_xaxis_changed(self, value, *args):
        if self.sender().isEnabled(): #so we don't call while changing combo contents
            self.on_plot_axis_changed(value, 'x')
        
    def on_plot_yaxis_changed(self, value, *args):
        if self.sender().isEnabled():
            self.on_plot_axis_changed(value, 'y')
        
    def on_plot_axis_changed(self, value, axis='x', sender=None):
        axes_i = self.pop_i if self.sender().axes_i is 'expanded' else self.sender().axes_i
        
        #~ print "* on_plot_axis_changed", axes_i, self.pop_i
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        
        do_command = "bundle.get_axes('%s').set_value('%saxis', '%s')" % (axesname, axis, value)
        undo_command = "bundle.get_axes('%s').set_value('%saxis', '%s')" % (axesname, axis, self.bundle.get_axes(self.pop_i).get_value('%saxis' % axis))
        description = "change %saxis to %s" % (axis, value)
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,kind='plot',description=description)
        self.undoStack.push(command)
        
    def on_plot_title_clicked(self, *args):
        pop = self.expanded_plot if self.sender().axes_i is 'expanded' else self.sender().pop
        axes_i = self.pop_i if self.sender().axes_i is 'expanded' else self.sender().axes_i
        title_old = self.bundle.get_axes(axes_i).get_value('title')
                
        pop.titleLineEdit.setText(title_old)
        
        pop.titleStackedWidget.setCurrentIndex(1)
        
    def on_plot_title_cancel(self, *args):
        pop = self.expanded_plot if self.sender().axes_i is 'expanded' else self.sender().pop
        
        pop.titleStackedWidget.setCurrentIndex(0)
        
    def on_plot_title_save(self, *args):
        pop = self.expanded_plot if self.sender().axes_i is 'expanded' else self.sender().pop
        axes_i = self.pop_i if self.sender().axes_i is 'expanded' else self.sender().axes_i
        title_old = self.bundle.get_axes(axes_i).get_value('title')
        
        title_new = pop.titleLineEdit.text()
        
        do_command = "bundle.get_axes('%s').set_value('title','%s')" % (title_old, title_new)
        undo_command = "bundle.get_axes('%s').set_value('title','%s')" % (title_new, title_old)
        description = "change axis title to %s" % title_new

        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,kind='plot',description=description)
        self.undoStack.push(command)
        
        pop.titleLinkButton.setText(title_new)  #TODO this should really be handled on the plot creation so they're all in sync
        pop.titleStackedWidget.setCurrentIndex(0)
        
        self.on_plots_changed() #to force updating title in treeviews and plots
        

    @PyInterp_selfdebug
    def on_plot_expand_toggle(self, plot=None, *args):
        #~ print "* on_plot_expand_toggle"
        #plot expanded from gui
        if self.sender() == self.meshmpl_canvas:
            self.mp_stackedWidget.setCurrentIndex(3)
            #~ self.on_plots_changed() # just to make sure up to date - REMOVE?
            return
        elif self.sender() == self.orbitmpl_canvas:
            return
        if self.mp_stackedWidget.currentIndex()==1: #then we need to expand            
            # this should intelligently raise if on data tab (and then return on collapse)
            i = self.plot_canvases.index(self.sender())
            
            # make title and axes options available from canvas
            self.expanded_canvas.info = {'xaxisCombo': self.expanded_plot.xaxisCombo, 'yaxisCombo': self.expanded_plot.yaxisCombo, 'titleLinkButton': self.expanded_plot.titleLinkButton}
            self.update_plot_widgets(i,self.expanded_canvas)
            
            if i!= self.pop_i: # then we've changed plots and need to force a redraw
                self.plot_redraw(None, i, self.expanded_canvas)
                self.pop_i = i #to track this in case pop is clicked

            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(i+1)
            
            self.mp_stackedWidget.setCurrentIndex(2)
            
            #~ new_plot_widget, canvas = self.create_plot_widget()
            #~ pop = CreatePopPlot()
            #~ self.mp_expandLayout.addWidget(pop.plot_Widget,0,0)
            #~ QObject.connect(pop.popPlot_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle) 
            #~ QObject.connect(pop.popPlot_popPushButton, SIGNAL("clicked()"), self.on_plot_pop)
            #~ QObject.connect(pop.popPlot_delPushButton, SIGNAL("clicked()"), self.on_plot_del)
            #~ pop.plot_gridLayout.addWidget(new_plot_widget,0,1)

            self.expanded_canvas.cla()
            self.expanded_canvas.plot(self.bundle, self.bundle.axes[i])
            

            # instead of dealing with attaching and destroying signals for the expanded plot
            # let's always check to see if the expanded plot is the same plot as a plot that is being asked to be redrawn
            # we can access this info in plot_redraw() through self.pop_i
            
            #~ self.attach_plot_signals(self.bundle.axes[i], i, self.expanded_canvas)
            #~ self.bundle.attach_signal(self.bundle.axes[i], 'add_plot', self.plot_redraw, i, canvas)
            #~ self.bundle.attach_signal(self.bundle.axes[i], 'remove_plot', self.plot_redraw, i, canvas)
            #~ for po in self.bundle.axes[i].plots:
                #~ self.bundle.attach_signal(po, 'set_value', self.plot_redraw, i, canvas)

        else: #then we need to return to grid
            self.mp_stackedWidget.setCurrentIndex(1)
            
            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(0)
            
        #~ self.update_datasets()
        
    def on_expanded_plot_clicked(self,canvas,event):
        t = event.xdata
        
        if 'phase' in canvas.xaxis:
            if self.bundle.select_time is None:
                return
            period = canvas.period
            phase_new = t
            phase_old = (self.bundle.select_time % period) / period
            dt = (phase_new - phase_old) * period
            t = self.bundle.select_time + dt
            
        do_command = 'bundle.set_select_time(%f)' % t
        undo_command = 'bundle.set_select_time(%s)' % self.bundle.select_time if self.bundle.select_time is not None else 'None'
        description = 'change select time to %f' % t
        self.on_param_command(do_command,undo_command,description='',thread=False)
        
        if self.bundle.get_setting('update_mesh_on_select_time'):
            self.on_mesh_update_clicked()

    @PyInterp_selfdebug
    def on_plot_pop(self):
        #plot popped from gui
        i = self.pop_i
        
        self.on_plot_expand_toggle() #to reset to grid view

        new_plot_widget, canvas = self.create_plot_widget()
       
        #~ self.bundle.attach_signal(self.bundle.axes[i], 'add_plot', self.plot_redraw, i, canvas)
        #~ self.bundle.attach_signal(self.bundle.axes[i], 'remove_plot', self.plot_redraw, i, canvas)       
        #~ for po in self.bundle.axes[i].plots:
            #~ self.bundle.attach_signal(po, 'set_value', self.plot_redraw, i, canvas)
        
        pop = CreatePopPlot(self)
        pop.popPlot_gridPushButton.setVisible(False)
        pop.popPlot_popPushButton.setVisible(False)
        pop.popPlot_delPushButton.setVisible(False)
        pop.plot_gridLayout.addWidget(new_plot_widget,0,0)
        pop.xaxisCombo.axes_i = i
        pop.xaxisCombo.pop = pop
        pop.yaxisCombo.axes_i = i
        pop.yaxisCombo.pop = pop
        pop.titleLinkButton.axes_i = i
        pop.titleLinkButton.pop = pop
        pop.title_cancelButton.axes_i = i
        pop.title_cancelButton.pop = pop
        pop.title_saveButton.axes_i = i
        pop.title_saveButton.pop = pop
        QObject.connect(pop.xaxisCombo, SIGNAL("currentIndexChanged(QString)"), self.on_plot_xaxis_changed)
        QObject.connect(pop.yaxisCombo, SIGNAL("currentIndexChanged(QString)"), self.on_plot_yaxis_changed)    
        QObject.connect(pop.titleLinkButton, SIGNAL("clicked()"), self.on_plot_title_clicked)  
        QObject.connect(pop.title_cancelButton, SIGNAL("clicked()"), self.on_plot_title_cancel)
        QObject.connect(pop.title_saveButton, SIGNAL("clicked()"), self.on_plot_title_save)  
        
        # make title and axes options available from canvas
        canvas.info = {'xaxisCombo': pop.xaxisCombo, 'yaxisCombo': pop.yaxisCombo, 'titleLinkButton': pop.titleLinkButton}
        
        plotEntryWidget = CreateDatasetWidget()
        plotEntryWidget.datasetTreeView.plotindex = i
        plotEntryWidget.selectorWidget.setVisible(False)
        plotEntryWidget.addDataWidget.setVisible(False)
        QObject.connect(plotEntryWidget.datasetTreeView, SIGNAL("parameterCommand"), self.on_param_command)
        QObject.connect(plotEntryWidget.datasetTreeView, SIGNAL("focusIn"), self.on_paramfocus_changed)
        
        pop.treeviewLayout.addWidget(plotEntryWidget)        
        
        self.plot_redraw(None, i, canvas)
 
        self.attach_plot_signals(self.bundle.axes[i], i, canvas)
        
        pop.show()
        
        self.plotEntry_widgets.append(plotEntryWidget.datasetTreeView) # add to the list of treeviews to be updated when data or plots change
        self.paramTreeViews.append(plotEntryWidget.datasetTreeView) # add to the list of treeviews that only allow one item to be focused
        self.plotEntry_axes_i.append(i)
        
        #~ self.update_plotTreeViews(wa=zip([plotEntryWidget.bp_plotsTreeView],[i])) #just draw the new one, not all
        self.update_datasets()
        
    def on_plot_del(self):
        #plot deleted from gui
        plottype = self.bundle.axes[self.pop_i].get_value('category')
        axesname = self.bundle.axes[self.pop_i].get_value('title')
        command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.remove_axes('%s')" % axesname,"bundle.add_axes(category='%s', title='%s')" % (plottype, axesname),kind='plots',thread=True,description='remove axes %s' % axesname)
        self.undoStack.push(command)  
        
        self.on_plot_expand_toggle()
        
    def on_systemEdit_clicked(self):
        self.mp_stackedWidget.setCurrentIndex(4)
                
    def on_observe_clicked(self):
        if self.sender()==self.lp_previewPushButton:
            kind = 'Preview'
        else: 
            kind = 'Compute'

        if kind not in self.bundle.compute:
            # then we need to create a compute parameterSet with default options
            #~ p = {key.replace(kind+"_",""): value for (key, value) in self.prefs.iteritems() if key.split('_')[0]==kind}
            #~ self.on_observeOptions_createnew(kind, p)
            self.observe_create(kind)
            
        self.bundle.purge_signals(self.bundle.attached_signals_system)
        params = self.bundle.get_compute(kind).copy()
        observatory.extract_times_and_refs(self.bundle.get_system(),params)
        self.set_time_is = len(params.get_value('time'))
        self.PyInterp_run('bundle.run_compute(\'%s\')' % (kind), kind='sys', thread=True)

    def on_observeOptions_createnew(self, kind, options):
        """
        this function will be run before changing observeoptions if the default parameterset has not already been created
        """
        # kind = 'compute' or 'preview'
        # options is a dictionary of all options needed to be sent to the parameterSet
        
        args = ",".join(["%s=\'%s\'" % (key, options[key]) for key in options if key!='mpi'])
        command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.add_compute(parameters.ParameterSet(context=\'compute\',%s),\'%s\')' % (args,kind),'bundle.remove_compute(\'%s\')' % (kind),kind='sys',thread=False,description='save %s options' % kind)
        self.undoStack.push(command) 
        
    def on_orbit_update_clicked(self,*args):
        """
        """
        self.orbitmpl_canvas.plot_orbit(self.bundle)
        
    def on_mesh_update_clicked(self,*args):
        """
        """
        #~ mplfig = self.meshmpl_canvas
        #~ 
        #~ self.bundle.plot_mesh(mplfig)
        self.meshmpl_canvas.plot_mesh(self.bundle)
        self.mesh_widget.setMesh(self.bundle.system.get_mesh()) # 3D view
        
    def on_mesh_update_auto_toggled(self,state):
        if self.versions_oncompute.isEnabled(): #so we can change its state while disabled
            do_command = "bundle.set_setting('update_mesh_on_select_time',%s)" % state
            undo_command = "bundle.set_setting('update_mesh_on_select_time',%s)" % (not state)
            description = 'auto update mesh set to %s' % state
            
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description=description)
            self.undoStack.push(command)
        
    def on_paramfocus_changed(self,*args):
        """
        only allow one item across all treeviews in self.paramTreeViews to have focus
        selecting an item in another tree view will *accept* any changes
        to *cancel* changes you must press escape first
        """
        #~ print "*** on_paramfocus_changed"
        for tv in self.paramTreeViews:
            if tv != self.sender():
                tv.item_changed()

    def on_param_changed(self,treeview,label,param,oldvalue,newvalue,oldunit=None,newunit=None,is_adjust=False,is_constraint=False):
        """
        this function is called from the signals of the parameter treeviews
        this will determine the call that needs to be made through the console to update the values
        if any command is necessary, it will be run, which will then result in the treeview being rebuilt with the updated value
        """
        #~ print "*** on_param_changed", label, param, oldvalue, newvalue
        
        parname = param.get_qualifier()
        if treeview==self.lp_meshTreeView:
            kind = 'mesh'
        elif treeview==self.lp_compTreeView:
            kind = 'component'
        elif treeview==self.lp_orbitTreeView:
            kind = 'orbit'
        elif treeview==self.lp_observeoptionsTreeView:
            kind = 'compute'
            if label not in self.bundle.compute:
                # need to first create item
                self.observe_create(label)
        elif treeview==self.rp_fitoptionsTreeView:
            kind = 'fitting'
            if label not in self.bundle.fitting:
                # need to first creat item
                self.fitting_create(label)
        elif treeview==self.sys_orbitOptionsTreeView:
            kind = 'orbitview'
            label = 'orbitview'
        elif treeview==self.sys_meshOptionsTreeView:
            kind = 'meshview'
            label = 'meshview'
        else:
            #probably coming from rp_fitinTreeView and need to determine type
            i = self.system_names.index(label)
            nchild = list(utils.traverse(self.jsmessenger.sysitems_nchild))[i]
            kind = 'component' if nchild=='0' else 'orbit'
        
        labelstr = '\'%s\'' % label if label not in ['orbitview','meshview'] else ''
        
        # add prior if necessary    
        if is_adjust and newvalue == True and not param.has_prior(): #then we need to create an initial prior
            lims = param.get_limits()
            command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).get_parameter(\'%s\').set_prior(distribution=\'uniform\',lower=%s,upper=%s)' % (kind,labelstr,parname,lims[0],lims[1]),'bundle.get_%s(%s).get_parameter(\'%s\').remove_prior()' % (kind,labelstr,parname),thread=False,description='add default prior for %s:%s' % (label,parname))
            self.undoStack.push(command)
        
        # change adjust/value if necessary
        if is_adjust:
            command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).set_adjust(\'%s\', %s)' % (kind,labelstr,parname,newvalue), 'bundle.get_%s(%s).set_adjust(\'%s\', %s)' % (kind,labelstr,parname,oldvalue),thread=False,description='change adjust of %s to %s' % (kind,newvalue))
        else:
            command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).set_value(\'%s\',%s)' % (kind,labelstr,parname,'%s' % newvalue if isinstance(newvalue,str) and 'np.' in newvalue else '\'%s\'' % newvalue),'bundle.get_%s(%s).set_value(\'%s\',%s)' % (kind,labelstr,parname,'%s' % oldvalue if isinstance(oldvalue,str) and 'np.' in oldvalue else '\'%s\'' % oldvalue),thread=False,description='change value of %s:%s' % (kind,parname))
        
        # change/add constraint
        if is_constraint:
            if newvalue.strip() == '':
                do_command = 'bundle.get_%s(%s).remove_constraint(\'%s\')' % (kind,labelstr,parname)
            else:
                do_command = 'bundle.get_%s(%s).add_constraint(\'{%s} = %s\')' % (kind,labelstr,parname,newvalue)
            if oldvalue.strip() == '':
                undo_command = 'bundle.get_%s(%s).remove_constraint(\'%s\')' % (kind,labelstr,parname)
            else:
                undo_command = 'bundle.get_%s(%s).add_constraint(\'{%s} = %s\')' % (kind,labelstr,parname,oldvalue)
        
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description="change constraint on %s:%s" % (label,parname))
            
        self.undoStack.push(command)
        
        # change units
        if oldunit is not None and newunit is not None:
            command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_unit(\'%s\')' % (kind,label,parname,newunit),'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_unit(\'%s\')' % (kind,label,parname,oldunit),thread=False,description='change value of %s:%s' % (kind,parname))
            self.undoStack.push(command)
            
    def on_param_command(self,do_command,undo_command,description='',thread=False):
        """
        allows more flexible/customized commands to be sent from parameter treeviews
        """
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=thread,description=description)
        self.undoStack.push(command)
            
        
    def on_plotoption_changed(self,*args):
        info = self.sender().info
        axes_i = info['axes_i']
        plot_i = info['plot_i']
        param = info['param']
        
        old_value = self.bundle.get_axes(axes_i).get_plot(plot_i).get_value(param)
        if param=='marker' or param=='linestyle':
            new_value = self.sender().currentText()
        if param=='active':
            new_value = True if self.sender().checkState()==2 else False
        if param=='color':
            button = self.sender()
            initialcolor = QColor()
            initialcolor.setNamedColor(old_value)
            dialog = QColorDialog()
            color = dialog.getColor(initialcolor)
            new_value = color.name()
            
        if new_value == old_value:
            return
        
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        command_do = "bundle.get_axes('%s').get_plot(%d).set_value('%s','%s')" % (axes_i, plot_i, param, new_value)
        command_undo = "bundle.get_axes('%s').get_plot(%d).set_value('%s','%s')" % (axes_i, plot_i, param, old_value)
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,command_do, command_undo, thread=False, description='change plotting option', kind='plots')
        
        self.undoStack.push(command)
        
    def on_plotoption_new(self,*args):
        info = self.sender().info
        axes_i = info['axes_i']
        plot_i = info['plot_i']       
        dataref = info['dataref']
        objref = info['objref']
        type = info['type']
        # we know its active and a check
        
        if objref==dataref: #then we're dealing with the toplevel
            objref = self.system_names[0]
        
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        command_do = "bundle.get_axes('%s').add_plot(type='%s',dataref='%s',objref='%s')" % (axesname, type, dataref, objref)
        command_undo = "bundle.get_axes('%s).remove_plot(%d)" % (axesname, len(self.bundle.get_axes(axes_i).get_plot()))

        command = phoebe_widgets.CommandRun(self.PythonEdit, command_do, command_undo, thread=False, description='add plot', kind='plots')
        
        self.undoStack.push(command)
        
    def on_PyInterp_commandrun(self):
        """
        this will be called after any command is sent through the python interface
        anything that cannot be caught through signals belongs here
        """
        try:
            self.bundle = self.PyInterp_get('bundle')
            #~ self.prefs = self.PyInterp_get('settings')
        except KeyError:
            return
            
        if self.bundle is not None and self.bundle.system is not None:
            self.update_system()
            self.on_fittingOption_changed()
            self.update_observeoptions()
            if self.mp_stackedWidget.currentIndex()==0:
                self.mp_stackedWidget.setCurrentIndex(1)
            self.tb_file_saveasAction.setEnabled(True)
            self.lp_DockWidget.setVisible(self.tb_view_lpAction.isChecked())
            self.rp_fittingDockWidget.setVisible(self.tb_view_rpAction.isChecked())
            self.bp_datasetsDockWidget.setVisible(self.tb_view_datasetsAction.isChecked())
            self.lp_systemDockWidget.setVisible(self.tb_view_systemAction.isChecked())
            self.rp_versionsDockWidget.setVisible(self.tb_view_versionsAction.isChecked())
            self.bp_pyDockWidget.setVisible(self.tb_view_pythonAction.isChecked())
            
            # check whether system is uptodate
            uptodate = self.bundle.get_uptodate()
            self.lp_previewPushButton.setEnabled(uptodate!='Preview')
            self.lp_computePushButton.setEnabled(uptodate!='Compute')
            
            # update misc gui items
            self.versions_oncompute.setEnabled(False)
            self.versions_oncompute.setChecked(self.bundle.get_setting('add_version_on_compute'))
            self.versions_oncompute.setEnabled(True)
            
            self.rp_savedFeedbackAutoSaveCheck.setEnabled(False)
            self.rp_savedFeedbackAutoSaveCheck.setChecked(self.bundle.get_setting('add_feedback_on_fitting'))
            self.rp_savedFeedbackAutoSaveCheck.setEnabled(True)
            
            self.sys_meshAutoUpdate.setEnabled(False)
            self.sys_meshAutoUpdate.setChecked(self.bundle.get_setting('update_mesh_on_select_time'))
            self.sys_meshAutoUpdate.setEnabled(True)
            
            # update version - should probably move this
            self.versions_treeView.set_data(self.bundle.versions)
            self.rp_savedFeedbackTreeView.set_data(self.bundle.feedbacks)
            
            # update plot mesh options - should probably move this
            self.sys_meshOptionsTreeView.set_data([self.bundle.plot_meshviewoptions],style=['nofit'])
            self.sys_orbitOptionsTreeView.set_data([self.bundle.plot_orbitviewoptions],style=['nofit'])
        
        else:
            if self.mp_stackedWidget.currentIndex()!=0:
                self.mp_stackedWidget.setCurrentIndex(0)
            self.tb_file_saveasAction.setEnabled(False)
            self.lp_DockWidget.setVisible(False)
            self.rp_fittingDockWidget.setVisible(False)
            self.bp_datasetsDockWidget.setVisible(False)
            self.lp_systemDockWidget.setVisible(False)
            self.rp_versionsDockWidget.setVisible(False)
            self.bp_pyDockWidget.setVisible(False)

        command = self.PythonEdit.prevcommand.replace(' ','')
        
        if 'bundle=' in command in command:
            #then new bundle
            self.on_new_bundle()
            
        if 'run_compute' in command:
            #~ self.meshmpl_canvas.plot_mesh(self.bundle.system)
        
            #should we only draw this if its visible?
            #~ self.mesh_widget.setMesh(self.bundle.system.get_mesh())
            self.on_plots_changed()
            
        if 'run_fitting' in command:
            #~ self.lp_stackedWidget.setCurrentIndex(1) #compute
            self.rp_stackedWidget.setCurrentIndex(1) #feedback
            
        #### TESTING ###
        ## EVENTUALLY RUN ONLY WHEN NEEDED THROUGH SIGNAL OR OTHER LOGIC
        self.update_datasets()
        
    def on_axes_add(self,category,objref,dataref):
        # signal received from dataset treeview with info to create new plot
        title = 'Plot %d' % (len(self.bundle.axes)+1)
        do_command = 'bundle.add_axes(category=\'%s\', title=\'%s\')' % (category,title)
        undo_command = 'bundle.remove_axes(%d)' % (len(self.bundle.axes))
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='plots',thread=False,description='add new plot')
        self.undoStack.push(command)
        
        self.on_axes_goto()
                    
    def on_axes_goto(self,plotname=None,ind=None):
        # signal receive from dataset treeview to goto a plot (axes) by name
        self.datasetswidget_main.ds_plotComboBox.setCurrentIndex([ax.get_value('title') for ax in self.bundle.axes].index(plotname)+1)        
        # expand plot? or leave as is?

    def on_plots_add(self,*args):
        #signal received from backend that a plot was added to bundle.axes
        self.on_plots_changed()
        
    def on_plots_remove(self,*args):
        #signal received from backend that a plot was removed from bundle.axes
        self.on_plots_changed()
        
    def on_plots_rename(self,*args):
        #signal received from backend that a plot was renamed in bundle.axes
        #~ self.on_plots_changed()
        # eventually this may change a label or something but probably won't
        # have to do much of anything
        #clear plot selection box
        
        currentText = self.datasetswidget_main.ds_plotComboBox.currentText()
        self.datasetswidget_main.ds_plotComboBox.setEnabled(False) # so we can ignore the signal
        self.datasetswidget_main.ds_plotComboBox.clear()
        items = ['all plots']+[ax.get_value('title') for ax in self.bundle.axes]
        self.datasetswidget_main.ds_plotComboBox.addItems(items)
        if currentText in items:
            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(items.index(currentText))
        self.datasetswidget_main.ds_plotComboBox.setEnabled(True)
        
        
    def on_plots_changed(self,*args):
        #~ print "*** on_plots_changed", len(self.bundle.axes)

        #bundle.axes changed - have to handle adding/deleting/reordering
        #for now we'll just completely recreate all thumbnail widgets and redraw
        
        #clear all canvases
        self.on_plot_clear_all() 
        
        self.on_plots_rename() # to update selection combo
        
        #redraw all plots
        for i,axes in enumerate(self.bundle.axes):
                
            new_plot_widget, canvas = self.create_plot_widget(thumb=True)
            self.plot_widgets.append(new_plot_widget)
            self.plot_canvases.append(canvas)
            #TODO change to horizontals in verticals so we don't have empty space for odd numbers
            num = len(self.plot_widgets)
            rows = 2 if num % 2 == 0 else 3
            for j,widget in enumerate(self.plot_widgets):
                row = j % rows
                col = j - row
                self.mp_plotGridLayout.addWidget(widget, row, col)

            if num >= 9:
                self.mp_addPlotLabelWidget.setVisible(False)
            else:
                self.mp_addPlotLabelWidget.setVisible(True)

            # create hooks
            self.attach_plot_signals(axes, i, canvas)
            #~ self.bundle.attach_signal(axes.axesoptions, 'set_value', self.plot_redraw, i)
            #~ self.bundle.attach_signal(axes, 'add_plot', self.plot_redraw, i)
            #~ self.bundle.attach_signal(axes, 'remove_plot', self.plot_redraw, i)
            #~ for po in axes.plots:
                #~ for paramname in po.keys():
                    #~ param = po.get_parameter(paramname)
                    #~ self.bundle.purge_signals([param])
                    #~ self.bundle.attach_signal(param, 'set_value', self.plot_redraw, i)
        
        for i in range(len(self.bundle.axes)):    
            self.plot_redraw(None,i)
            
    def on_select_time_changed(self,param=None,i=None,canvas=None):
        #~ print "*** on_select_time_changed",i,canvas
        #~ print "*** len(self.plot_canvases)", len(self.plot_canvases+[self.expanded_canvas])
            
        self.plot_redraw(param,i,canvas)
        #~ canvas.update_select_time(self.bundle)
        #~ self.on_plots_changed()
            
            
    def PyInterp_run(self, command, write=True, thread=False, kind=None):
        phoebe_widgets.PyInterp.run_from_gui(self.PythonEdit, command, write, thread, kind)

    def PyInterp_get(self, obj):
        '''get copy of object from interpreter'''
        return self.PythonEdit.comm[str(obj)]

    def PyInterp_send(self, objname, obj):
        '''send copy of object to be available to the interpreter'''
        self.PythonEdit.comm[str(objname)]=obj
        phoebe_widgets.PyInterp.update_from_comm(self.PythonEdit)
                
    def on_set_time(self,*args):
        #~ print "*** on_set_time", self.set_time_i, self.set_time_is
        if self.set_time_i is not None and self.set_time_is is not None:
            if self.unity_launcher is not None:
                self.unity_launcher.set_property("progress", float(self.set_time_i)/self.set_time_is)
                self.unity_launcher.set_property("progress_visible", True)
                
            #~ self.mp_progressBar.setValue(int(float(self.set_time_i+1)/self.set_time_is*100))
            self.lp_progressBar.setValue(int(float(self.set_time_i+1)/self.set_time_is*100))
            #~ self.rp_progressBar.setValue(int(float(self.set_time_i+1)/self.set_time_is*100))
            self.set_time_i += 1
            

        
        #~ self.meshmpl_canvas.plot_mesh(self.bundle.system)
        
        #should we only draw this if its visible?
        #~ self.mesh_widget.setMesh(self.bundle.system.get_mesh())
            
    def update_observeoptions(self):
        default = {}
        
        # apply any default from user settings
        default['Preview'] = parameters.ParameterSet(context='compute')
        default['Compute'] = parameters.ParameterSet(context='compute')
        
        computes = []
        for key in ['Preview','Compute']:
            if key in self.bundle.compute:
                computes.append(self.bundle.compute[key])
            else:
                computes.append(default[key])
                
        self.lp_observeoptionsTreeView.set_data(computes,style=['nofit'])
        self.lp_observeoptionsTreeView.headerItem().setText(1, 'Preview')
        self.lp_observeoptionsTreeView.headerItem().setText(2, 'Compute')
        self.lp_observeoptionsTreeView.resizeColumnToContents(0) 
        
    def observe_create(self, kind):
        command = phoebe_widgets.CommandRun(self.PythonEdit, 'bundle.add_compute(label=\'%s\')' % (kind), 'bundle.remove_compute(\'%s\')' % kind,thread=False,description='add compute options %s' % kind)
        self.undoStack.push(command)
        
    def fitting_create(self, kind):
        command = phoebe_widgets.CommandRun(self.PythonEdit, 'bundle.add_fitting(parameters.ParameterSet(context=\'fitting:%s\'), \'%s\')' % (kind,kind), 'bundle.remove_fitting(\'%s\')' % kind,thread=False,description='add fitting options %s' % kind)
        self.undoStack.push(command)
    
    def update_fittingOptions(self, *args):
        #~ print "*** update_fittingOptions"
        default = OrderedDict()
        
        default['grid'] = parameters.ParameterSet(context='fitting:grid')
        default['minuit'] = parameters.ParameterSet(context='fitting:minuit')
        default['lmfit'] = parameters.ParameterSet(context='fitting:lmfit')
        default['lmfit:leastsq'] = parameters.ParameterSet(context='fitting:lmfit:leastsq')
        default['lmfit:nelder'] = parameters.ParameterSet(context='fitting:lmfit:nelder')
        default['emcee'] = parameters.ParameterSet(context='fitting:emcee')
        default['pymc'] = parameters.ParameterSet(context='fitting:pymc')
        
        #loop through fitting options in bundle first
        currenttext = self.rp_methodComboBox.currentText()
        fittingkeys = []
        self.rp_methodComboBox.clear()
        for k,v in self.bundle.fitting.iteritems():
            fittingkeys.append(k)
            self.rp_methodComboBox.addItem(k)
        for k,v in default.iteritems():
            if k not in fittingkeys:
                fittingkeys.append(k)
                self.rp_methodComboBox.addItem(k)
        # return to original selection
        if len(currenttext) > 0: #ignore empty case
            self.rp_methodComboBox.setCurrentIndex(self.rp_methodComboBox.findText(currenttext))
        
        
    def on_fittingOption_changed(self, *args):
        # get the correct parameter set and send to treeview
        combo = self.rp_methodComboBox
        key = str(combo.currentText())
        if len(key)==0: return
        #~ print "*** on_fittingOption_changed", key
        
        if key in self.bundle.fitting.keys():
            fitting = self.bundle.get_fitting(key)
        else:
            fitting = parameters.ParameterSet(context="fitting:%s" % key)
            
        self.rp_fitoptionsTreeView.set_data([fitting],style=['nofit'])
        self.rp_fitoptionsTreeView.headerItem().setText(1, key)
        
    def on_fitoptions_param_changed(self,treeview,label,param,oldvalue,newvalue,oldunit=None,newunit=None,is_adjust=False,is_prior=False):
        #~ print "*** on_fitoptions_param_changed", label, param, oldvalue, newvalue
        #override label
        #~ label = str(self.rp_methodComboBox.currentText())
        paramname = param.get_qualifier()
        
        #~ if label not in self.bundle.fitting:
            # need to first create item
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit, 'bundle.add_fitting(parameters.ParameterSet(context=\'fitting:%s\'), \'%s\')' % (label, label), 'bundle.remove_fitting(\'%s\')' % label,thread=False,description='add fitting options %s' % label)
            #~ self.undoStack.push(command)
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_fitting(\'%s\').set_value(\'%s\',\'%s\')' % (label,paramname,newvalue),'bundle.get_fitting(\'%s\').set_value(\'%s\',\'%s\')' % (label,paramname,oldvalue),thread=False,description='change value of fitting:%s:%s' % (label,param))
            
        self.undoStack.push(command)
        
    def on_prior_changed(self,label,param,distribution):
        #~ print "*** on_fitoptions_prior_changed"
        
        paramname = param.get_qualifier()
        new_dist = distribution
        old_dist = param.get_prior().get_distribution()
        kind = param.context[0]
        
        new_prior = 'distribution=\'%s\'' % new_dist[0]
        old_prior = 'distribution=\'%s\'' % old_dist[0]
    
        for key in new_dist[1].keys():
            new_prior +=',%s=%f' % (key,new_dist[1][key])
        for key in old_dist[1].keys():
            old_prior +=',%s=%f' % (key,old_dist[1][key])
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_prior(%s)' % (kind,label,paramname,new_prior),'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_prior(%s)' % (kind,label,paramname,old_prior),thread=False,description='change prior of %s:%s' % (label,paramname))
        self.undoStack.push(command)
        
    def on_fit_clicked(self):
        label = str(self.rp_methodComboBox.currentText())

        if label not in self.bundle.fitting:
            # need to first create item
            self.fitting_create(label)
        if 'Compute' not in self.bundle.compute:
            self.observe_create('Compute')

        self.PyInterp_run('bundle.run_fitting(\'Compute\', \'%s\')' % (label),thread=True,kind='sys')
        
        #~ self.rp_stackedWidget.setCurrentIndex(1) #feedback
        
    def on_feedback_changed(self,feedback_name):
        
        self.current_feedback_name = feedback_name
        self.rp_fitoutTreeView.set_data(self.bundle.get_feedback(feedback_name),self.system_ps,self.system_names)
        self.rp_stackedWidget.setCurrentIndex(1) # feedback
        
    def on_feedback_reject_clicked(self):
        #~ self.rp_fittingWidget.setVisible(True)
        #~ self.rp_feedbackWidget.setVisible(False)
      
        self.rp_stackedWidget.setCurrentIndex(0) #fitting input
        
    def on_feedback_accept_clicked(self):
        
        if self.current_feedback_name is not None:
            do_command = "bundle.accept_feedback('%s')" % self.current_feedback_name
        else:
            do_command = "bundle.accept_feedback(-1)"
        undo_command = "undo not available"
        description = "accept feedback"
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description=description)
        self.undoStack.push(command)
        
        self.rp_stackedWidget.setCurrentIndex(0) # TODO call this automatically?
        
    def on_feedbacksaveauto_toggled(self,state):
        if self.rp_savedFeedbackAutoSaveCheck.isEnabled(): #so we can change its state while disabled
            do_command = "bundle.set_setting('add_feedback_on_fitting',%s)" % state
            undo_command = "bundle.set_setting('add_feedback_on_fitting',%s)" % (not state)
            description = 'auto save feedback set to %s' % state
            
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description=description)
            self.undoStack.push(command)
        
    def on_versionsauto_toggled(self,state):
        if self.versions_oncompute.isEnabled(): #so we can change its state while disabled
            do_command = "bundle.set_setting('add_version_on_compute',%s)" % state
            undo_command = "bundle.set_setting('add_version_on_compute',%s)" % (not state)
            description = 'auto version set to %s' % state
            
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description=description)
            self.undoStack.push(command)
            
    def on_versionsadd_clicked(self):
        do_command = "bundle.add_version()"
        undo_command = "bundle.remove_version(0)"
        description = 'add version'
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description=description)
        self.undoStack.push(command)
        
    @PyInterp_selfdebug
    def on_prefsLoad(self):
        '''
        load settings from file and save to self.prefs
        '''       
        
        #~ try:
            #~ self.prefs = self.PyInterp_get('settings')
        #~ except:
        self.prefs = usersettings.load()


    def on_prefsShow(self):
        '''
        open settings window
        if changes are applied, update self.prefs and save to file
        and then run on_prefsUpdate to update gui
        '''
        pop = CreatePopPrefs(self)
        p = {key: self.prefs.get_setting(key) for key in self.prefs.default_preferences.keys()}
        
        self.setpopguifromprefs(pop, p)
                          
        response = pop.exec_()
        
        if response:
            p = self.getprefsfrompopgui(pop, p)
 
            self.on_prefsSave(p)
            self.on_prefsUpdate()
    
    def on_prefsSave(self,p):
        self.prefs.preferences = p
        self.PyInterp_send('settings',self.prefs)
        # TODO we should really send system calls for each of the changed settings
        
        command_do = "settings.save()"
        command_undo = ""
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,command_do, command_undo, thread=False, description='save settings', kind='settings')
        
        self.undoStack.push(command)        
        
    @PyInterp_selfdebug
    def on_prefsUpdate(self):
        p = {key: self.prefs.get_setting(key) for key in self.prefs.default_preferences.keys()}
        
        self.tb_view_lpAction.setChecked(p['panel_params'])
        self.tb_view_rpAction.setChecked(p['panel_fitting'])
        self.tb_view_versionsAction.setChecked(p['panel_versions'])
        self.tb_view_systemAction.setChecked(p['panel_system'])
        self.tb_view_datasetsAction.setChecked(p['panel_datasets'])
        self.tb_view_pythonAction.setChecked(p['panel_python'])
        #~ self.tb_view_sysAction.setChecked(p['sys'])
        #~ self.tb_view_glAction.setChecked(p['gl'])

        self.tb_view_pythonAction.setChecked(p['pyinterp_enabled'])
        self.PythonEdit.thread_enabled = p['pyinterp_thread_on']
        self.PythonEdit.write_sys = p['pyinterp_tutsys']
        self.PythonEdit.write_plots = p['pyinterp_tutplots']
        self.PythonEdit.write_settings = p['pyinterp_tutsettings']
        
        # THIS PROBABLY DOESN'T WORK
        for pluginpath in p['plugins'].keys():
            if pluginpath not in self.pluginsloaded and p['plugins'][pluginpath]:
                # need to load and enable
                self.pluginsloaded.append(pluginpath)
                self.plugins.append(imp.load_source(os.path.basename(pluginpath).strip('.py'), pluginpath))
                self.guiplugins.append(self.plugins[-1].GUIPlugin())
                self.plugins[i].enablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)
            if pluginpath in self.pluginsloaded:
                i = self.pluginsloaded.index(pluginpath)
                if pluginpath in p['plugins'][pluginpath]:
                    # then already loaded, but need to enable
                    self.plugins[i].enablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)
                else:
                    #then already loaded, but need to disable
                    self.plugins[i].disablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)

        #~ from pyphoebe.plugins import keplereb
        #~ self.keplerebplugin = keplereb.GUIPlugin()
        #~ keplereb.GUIPlugin.enablePlugin(self.keplerebplugin, self, gui.Ui_PHOEBE_MainWindow)
        #~ keplereb.GUIPlugin.disablePlugin(self.keplerebplugin)

    def setpopguifromprefs(self, pop, p, pdef=None):
        if pdef is None:
            #~ pdef = self.prefsdefault
            pdef = self.prefs
        bools = pop.findChildren(QCheckBox)
        bools += pop.findChildren(QRadioButton)

        for w in bools:
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setChecked(p[key])
                else:
                    w.setChecked(pdef[key])
                    
        for w in pop.findChildren(QPlainTextEdit):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setPlainText(p[key])
                else:
                    w.setPlainText(pdef[key])
                    
        for w in pop.findChildren(QSpinBox):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    w.setValue(p[key])
                else:
                    w.setValue(pdef[key])
                    
        #~ for w in pop.findChildren(QComboBox):
            #~ if w.objectName().split('_')[0]=='p':
                #~ key = "_".join(str(w.objectName()).split('_')[1:])
                #~ if key in p:
                    #~ w.setCurrentText(p[key])
                #~ else:
                    #~ w.setCurrentText(self.prefsdefault[key])
    
    def getprefsfrompopgui(self, pop, p, pdef=None):
        if pdef is None:
            #~ pdef = self.prefsdefault
            pdef = self.prefs
        bools = pop.findChildren(QCheckBox)
        bools += pop.findChildren(QRadioButton)
        
        for w in bools:
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    p[key] = w.isChecked()
                else:
                    p[key] = pdef[key]

        for w in pop.findChildren(QPlainTextEdit):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    p[key] = str(w.toPlainText())
                else:
                    p[key] = pdef[key]
                    
        for w in pop.findChildren(QSpinBox):
             if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    p[key] = w.value()
                else:
                    p[key] = pdef[key]
        
        for w in pop.findChildren(QComboBox):
            if w.objectName().split('_')[0]=='p':
                key = "_".join(str(w.objectName()).split('_')[1:])
                if key in p:
                    p[key] = str(w.currentText())
                else:
                    p[key] = pdef[key]
        return p
            
    def on_aboutShow(self):
        pop = CreatePopAbout(self)
        pop.show()

    def on_helpShow(self):
        pop = CreatePopHelp(self)
        pop.show()
        
    def on_fileEntryShow(self):
        pop = CreatePopFileEntry(self)
        pop.show()

        #determine what type of data we're loading, and set options accordingly
        sender = self.sender()
        if sender==self.datasetswidget_main.addLCButton:
            if _alpha_test:
                pop.datatypes = ['--Data Type--', 'time', 'flux', 'sigma']
            else:
                pop.datatypes = ['--Data Type--', 'time', 'flux', 'sigma', 'component', 'ignore']
            pop.context = 'lcobs'
            pop.setWindowTitle('PHOEBE - Import LC Data')
        elif sender==self.datasetswidget_main.addRVButton:
            if _alpha_test:
                pop.datatypes = ['--Data Type--', 'time', 'rv', 'sigma']
            else:
                pop.datatypes = ['--Data Type--', 'time', 'rv', 'sigma', 'component', 'ignore']
            pop.context = 'rvobs'
            pop.setWindowTitle('PHOEBE - Import RV Data')
        elif sender==self.datasetswidget_main.addETVButton:
            if _alpha_test:
                pop.datatypes = ['--Data Type--', 'time', 'o-c', 'sigma']
            else:
                pop.datatypes = ['--Data Type--', 'time', 'o-c', 'sigma', 'component', 'ignore']
            pop.context = 'etvobs'
            pop.setWindowTitle('PHOEBE - Import ETV Data')
        elif sender==self.datasetswidget_main.addSPButton:
            if _alpha_test:
                pop.datatypes = ['--Data Type--', 'wavelength', 'flux', 'sigma']
            else:
                pop.datatypes = ['--Data Type--', 'wavelength', 'flux', 'sigma', 'component', 'ignore']
            pop.context = 'spobs'
            pop.setWindowTitle('PHOEBE - Import Spectral Data')
        pop.filtertypes = ['JOHNSON.V','JOHNSON.R']
            
        QObject.connect(pop.pfe_fileChooserButton, SIGNAL("clicked()"), self.on_pfe_fileChoose)
        QObject.connect(pop.buttonBox.buttons()[1], SIGNAL("clicked()"), pop.close)
        QObject.connect(pop.buttonBox.buttons()[0], SIGNAL("clicked()"), self.on_pfe_okClicked)
                
        if True:  #change this to load current filename if already set
            pop.pfe_fileChooserButton.setText('')
            #~ pop.dataTreeView.setVisible(False)
            pop.dataTextEdit.setVisible(False)
        else:
            self.on_pfe_fileChoose(pop.pfe_fileChooserButton)

    def on_pfe_fileChoose(self, button=None):
        from_button = True if button is None else False
        if button is None:
            button = self.sender() #this will be the button from the correct dialog
        pop = button.topLevelWidget()
        f = QFileDialog.getOpenFileName(self, 'Import Data...', self.latest_dir if self.latest_dir is not None else '.', **_fileDialog_kwargs)
        if len(f)==0: #then no file selected
            return
        self.latest_dir = os.path.dirname(str(f))
        
        button.setText(f)
        #~ pop.dataTreeView.setVisible(True)
        pop.dataTextEdit.setVisible(True)

        fdata = open(f, 'r')
        data = fdata.readlines()

        pop.dataTextEdit.clear()
        
        pop.pfe_filterComboBox.addItems(pop.filtertypes)

        # parse header and try to predict settings
        (columns,components,datatypes,units,ncols),(pbdep,dataset) = datasets.parse_header(str(f))
        if components is None:
            components = ['None']*ncols
        
        if pbdep['passband'] in pop.filtertypes:
            passband = pop.pfe_filterComboBox.setCurrentIndex(pop.filtertypes.index(pbdep['passband'])+1)
        pop.name.setText(pbdep['ref'])

        # remove any existing colwidgets
        if hasattr(pop, 'colwidgets'): # then we have existing columns from a previous file and need to remove them
            for colwidget in pop.colwidgets:
                pop.horizontalLayout_ColWidgets.removeWidget(colwidget)
        pop.colwidgets = []
        for col in range(ncols):
            colwidget = CreatePopFileEntryColWidget()
            pop.colwidgets.append(colwidget) # so we can iterate through these later
            pop.horizontalLayout_ColWidgets.addWidget(colwidget)

            colwidget.col_comboBox.setVisible(False)

            colwidget.type_comboBox.addItems(pop.datatypes)

            try:
                colwidget.comp_comboBox.addItems(self.system_names) #self.system_names needs to be [] at startup
                if not _alpha_test:
                    colwidget.comp_comboBox.addItem('provided in another column')
            except: pass
            
            QObject.connect(colwidget.type_comboBox, SIGNAL("currentIndexChanged(QString)"), self.on_pfe_typeComboChanged)
            
            if columns is not None and columns[col] in pop.datatypes:
                colwidget.type_comboBox.setCurrentIndex(pop.datatypes.index(columns[col]))
            if components is not None and components[col] in self.system_names:
                colwidget.comp_comboBox.setCurrentIndex(self.system_names.index(components[col])+1) #+1 because of header item
            
            #~ pop.dataTreeView.headerItem().setText(col, str(col+1))

            colwidget.col_comboBox.setEnabled(False)
            colwidget.col_comboBox.clear()
            for col in range(ncols):
                colwidget.col_comboBox.addItem('Column %d' % (col+1))
            colwidget.col_comboBox.setEnabled(True)

        for i,line in enumerate(data):
            if i > 100: continue    
            if i==100: pop.dataTextEdit.appendPlainText('...')
            if i<100:
                pop.dataTextEdit.appendPlainText(line.strip())
            
    def on_pfe_typeComboChanged(self):
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
            
    def on_pfe_okClicked(self):
        pop = self.sender().topLevelWidget()
        filename=pop.pfe_fileChooserButton.text()
        passband = pop.pfe_filterComboBox.currentText()
        name = pop.name.text() if len(pop.name.text()) > 0 else None
        
        if filename is None: #then just create the synthetic (pbdeps)
            #TODO: make this undoable
            self.PyInterp_run('', thread=False, kind='sys')
            return
        
        columns = [str(colwidget.type_comboBox.currentText()) if '--' not in str(colwidget.type_comboBox.currentText()) else None for colwidget in pop.colwidgets]
        units = [str(colwidget.units_comboBox.currentText()) if '--' not in str(colwidget.units_comboBox.currentText()) else None for colwidget in pop.colwidgets]
        components = [str(colwidget.comp_comboBox.currentText()) if '--' not in str(colwidget.comp_comboBox.currentText()) else None for colwidget in pop.colwidgets]

        
        #TODO make this more intelligent so values that weren't changed by the user aren't sent
        #TODO name currently doesn't do anything in bundle
        
        if passband == '--Passband--':
            QMessageBox.information(None, "Warning", "Cannot load data: no passband provided")  
            return
        
        for i,colwidget in enumerate(pop.colwidgets):
            typecombo = colwidget.type_comboBox
            unitscombo = colwidget.units_comboBox
            compcombo = colwidget.comp_comboBox
            colcombo = colwidget.col_comboBox
            
            if typecombo.currentIndex()!=0:
                if compcombo.isEnabled() and compcombo.currentIndex()==0:
                    QMessageBox.information(None, "Warning", "Cannot load data: no component for column %d" % (i+1))  
                    return

        do_command = "bundle.load_data(context='%s',filename='%s',passband='%s',columns=%s,components=%s,ref='%s')" % (pop.context, filename, passband, columns, components, name)
        undo_command = "bundle.remove_data(ref='%s')" % name
        description = "load %s dataset" % name
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='sys',thread=False,description=description)
        self.undoStack.push(command)

        pop.close()
            
    def on_pfe_refreshClicked(self):
        pop = self.sender().topLevelWidget()
        # reload clicked inside popup - just refresh the data in the window but don't apply until ok clicked
        # reload clicked from main window tree view - immediately refresh data object 

    def main(self):
        self.show()

### startup the main window and pass any command-line arguments
if __name__=='__main__':
    STDOUT, STDERR, DEBUG = False, False, False
    #~ STDOUT, STDERR = True, True
    if "stdout" in sys.argv:
        STDOUT = True
    if "stderr" in sys.argv:
        STDERR = True
    if "debug" in sys.argv:
        DEBUG = True
    app = QApplication(sys.argv)
    font = app.font()
    palette = app.palette()
    phoebegui = PhoebeGUI((STDOUT, STDERR),DEBUG,font,palette)
    phoebegui.main()
    app.exec_()
