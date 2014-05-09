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
from phoebe.frontend.usersettings import Settings
from phoebe.frontend.gui import phoebe_plotting, phoebe_widgets, phoebe_dialogs

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


global _devel_version
# _devel_version = True for non-released features
# general public version should be _devel_version = False
_devel_version = False


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
        
### MAIN WINDOW
class PhoebeGUI(QMainWindow, gui.Ui_PHOEBE_MainWindow):
    def __init__(self, DEBUG=(False, False), selfdebug=False, font=None, palette=None, parent=None, file_to_open=None):
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
            
        # initialize needed stuff
        self.plotEntry_widgets = []
        
        # hide float and close buttons on dock widgets
        #~ self.bp_pyDockWidget.setTitleBarWidget(QWidget())
        self.bp_datasetsDockWidget.setTitleBarWidget(QWidget())

        # set default visibility for docks/frames
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
        self.datasetswidget_main = phoebe_dialogs.CreateDatasetWidget(parent=self)
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
        self.expanded_canvas.info['axes_i'] = 'expanded'
        expanded_plot = phoebe_dialogs.CreatePopPlot()
        expanded_plot.xaxisCombo.axes_i = 'expanded'
        expanded_plot.yaxisCombo.axes_i = 'expanded'
        expanded_plot.titleLinkButton.axes_i = 'expanded'
        expanded_plot.title_cancelButton.axes_i = 'expanded'
        expanded_plot.title_saveButton.axes_i = 'expanded'
        self.mp_expandLayout.addWidget(expanded_plot.plot_Widget,0,0)
        expanded_plot.plot_gridLayout.addWidget(self.expanded_plot_widget,0,0)
        expanded_plot.zoom_in.info = {'axes_i': 'expanded', 'canvas': self.expanded_canvas}
        expanded_plot.zoom_out.info = {'axes_i': 'expanded', 'canvas': self.expanded_canvas}
        expanded_plot.pop_mpl.info = {'axes_i': 'expanded'}
        expanded_plot.save.info = {'axes_i': 'expanded'}
        self.expanded_plot = expanded_plot
        
        QObject.connect(expanded_plot.zoom_in, SIGNAL("toggled(bool)"), self.on_plot_zoom_in_toggled)
        QObject.connect(self.expanded_canvas, SIGNAL("plot_zoom"), self.on_plot_zoom_in)
        QObject.connect(expanded_plot.zoom_out, SIGNAL("clicked()"), self.on_plot_zoom_out_clicked)
        QObject.connect(expanded_plot.pop_mpl, SIGNAL("clicked()"), self.on_plot_pop_mpl_clicked)
        QObject.connect(expanded_plot.save, SIGNAL("clicked()"), self.on_plot_save_clicked)
      
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
        QObject.connect(self.lp_methodComboBox, SIGNAL("currentIndexChanged(QString)"), self.on_observeOption_changed)
        QObject.connect(self.lp_observeoptionsReset, SIGNAL("clicked()"), self.on_observeOption_delete)
        QObject.connect(self.lp_observeoptionsDelete, SIGNAL("clicked()"), self.on_observeOption_delete)
        QObject.connect(self.lp_observeoptionsAdd, SIGNAL("clicked()"), self.on_observeOption_add)
        QObject.connect(self.lp_computePushButton, SIGNAL("clicked()"), self.on_observe_clicked)
        QObject.connect(self.lp_progressQuit, SIGNAL("clicked()"), self.cancel_thread)
        
        QObject.connect(self.sys_orbitPushButton, SIGNAL("clicked()"), self.on_orbit_update_clicked)
        QObject.connect(self.sys_meshPushButton, SIGNAL("clicked()"), self.on_mesh_update_clicked)
        QObject.connect(self.sys_meshAutoUpdate, SIGNAL("toggled(bool)"), self.on_mesh_update_auto_toggled)

        # middle panel signals
        QObject.connect(self.mp_sysSelWebView, SIGNAL("ctrl_pressed"), self.on_sysSel_ctrl)
        QObject.connect(self.mp_sysSelWebView, SIGNAL("ctrl_released"), self.on_sysSel_ctrlReleased)
        if _devel_version:
            QObject.connect(self.jsmessenger, SIGNAL("editClicked"), self.on_systemEdit_clicked)
        QObject.connect(self.jsmessenger, SIGNAL("selectionUpdate"), self.on_sysSel_selectionChanged)
        QObject.connect(self.mpsys_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle)
        QObject.connect(self.mpgl_gridPushButton, SIGNAL("clicked()"), self.on_plot_expand_toggle)
        QObject.connect(self.datasetswidget_main.datasetTreeView, SIGNAL("axes_add"), self.on_axes_add)
        QObject.connect(self.datasetswidget_main.datasetTreeView, SIGNAL("axes_goto"), self.on_axes_goto)
        #~ QObject.connect(self.datasetswidget_main.datasetTreeView, SIGNAL("parameterCommand"), self.on_param_command) # done in loop below
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
        QObject.connect(self.rp_fitoptionsReset, SIGNAL("clicked()"), self.on_fittingOption_delete)
        QObject.connect(self.rp_fitoptionsDelete, SIGNAL("clicked()"), self.on_fittingOption_delete)
        QObject.connect(self.rp_fitPushButton, SIGNAL("clicked()"), self.on_fit_clicked)
        QObject.connect(self.rp_rejectPushButton, SIGNAL("clicked()"), self.on_feedback_reject_clicked)
        QObject.connect(self.rp_acceptPushButton, SIGNAL("clicked()"), self.on_feedback_accept_clicked)
        
        QObject.connect(self.rp_savedFeedbackAutoSaveCheck, SIGNAL("toggled(bool)"), self.on_feedbacksaveauto_toggled)
        
        QObject.connect(self.versions_oncompute, SIGNAL("toggled(bool)"), self.on_versionsauto_toggled)
        QObject.connect(self.versions_addnow, SIGNAL("clicked()"), self.on_versionsadd_clicked)
                
        # bottom panel signals
        QObject.connect(self.datasetswidget_main.ds_typeComboBox, SIGNAL("currentIndexChanged(QString)"), self.update_datasets)
        QObject.connect(self.datasetswidget_main.ds_plotComboBox, SIGNAL("currentIndexChanged(QString)"), self.update_datasets)
        QObject.connect(self.datasetswidget_main.addDataButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        #~ QObject.connect(self.datasetswidget_main.addLCButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        #~ QObject.connect(self.datasetswidget_main.addRVButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        #~ QObject.connect(self.datasetswidget_main.addSPButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        #~ QObject.connect(self.datasetswidget_main.addETVButton, SIGNAL("clicked()"), self.on_fileEntryShow) 
        #~ self.datasetswidget_main.addETVButton.setEnabled(False)
        
        # tree view signals
        self.paramTreeViews = [self.lp_compTreeView,self.lp_orbitTreeView, self.lp_meshTreeView, self.rp_fitinTreeView, self.rp_fitoutTreeView, self.rp_fitoptionsTreeView, self.lp_observeoptionsTreeView, self.datasetswidget_main.datasetTreeView, self.versions_treeView, self.rp_savedFeedbackTreeView, self.sys_orbitOptionsTreeView, self.sys_meshOptionsTreeView]
        for tv in self.paramTreeViews:
            QObject.connect(tv, SIGNAL("parameterChanged"), self.on_param_changed)
            QObject.connect(tv, SIGNAL("focusIn"), self.on_paramfocus_changed)
        QObject.connect(self.rp_fitinTreeView, SIGNAL("priorChanged"), self.on_prior_changed)  
        QObject.connect(self.rp_savedFeedbackTreeView, SIGNAL("feedbackExamine"), self.on_feedback_changed) 

        # pyinterp signals
        QObject.connect(self.PythonEdit, SIGNAL("command_run"), self.on_PyInterp_commandrun)
        QObject.connect(self.PythonEdit, SIGNAL("GUILock"), self.gui_lock)
        QObject.connect(self.PythonEdit, SIGNAL("GUIUnlock"), self.gui_unlock)
        QObject.connect(self.PythonEdit, SIGNAL("GUIthrowerror"), self.thread_failed) 
        QObject.connect(self.PythonEdit, SIGNAL("set_time"), self.on_set_time)
        QObject.connect(self.PythonEdit, SIGNAL("undo"), self.on_undo_clicked) 
        QObject.connect(self.PythonEdit, SIGNAL("redo"), self.on_redo_clicked) 
        QObject.connect(self.PythonEdit, SIGNAL("plots_changed"), self.on_plots_changed) 

        # load settings
        self.pluginsloaded = []
        self.plugins = []
        self.guiplugins = []
        #~ self.on_prefsLoad() #load to self.prefs
        self.prefs_pop = None
        self.lock_pop = None
        self.on_prefsUpdate(startup=True) #apply to gui
        self.latest_dir = None
        
        # send startup commands to interpreter
        startup_default = 'import phoebe\nimport matplotlib.pyplot as plt\nfrom phoebe.frontend.bundle import Bundle, load\nfrom phoebe.parameters import parameters, create, tools\nfrom phoebe.io import parsers\nfrom phoebe.utils import utils\nfrom phoebe.frontend.usersettings import Settings\nsettings = Settings()'
        # this string is hardcoded - also needs to be in phoebe_dialogs.CreatePopPres.set_gui_from_prefs
        for line in startup_default.split('\n'):
            self.PyInterp_run(line, write=True, thread=False)
        for line in self.prefs.get_gui().get_value('pyinterp_startup_custom').split('\n'):
            self.PyInterp_run(line, write=True, thread=False)
            
        # disable items for alpha version
        if not _devel_version:
            #~ self.rp_fitPushButton.setEnabled(False)
            self.mp_splash_triplePushButton.setEnabled(False)
            self.tb_view_rpAction.setEnabled(False)
            self.tb_view_versionsAction.setEnabled(False)
            self.tb_view_systemAction.setEnabled(False) # maybe enable for release if ready?

        # Set system to None - this will then result in a call to on_new_bundle
        # any additional setup should be done there
        self.PyInterp_run("bundle = Bundle(False)",kind='sys',write=False,thread=False)
        #~ self.on_new_bundle()
        #~ self.PyInterp_run("bundle = Bundle()",kind='sys',thread=False)
        self.on_new_clicked(file_to_open)
        
    def bundle_get_system_structure(self,bundle,return_type='label',flat=False,**kwargs):
        """
        Get the structure of the system below any bodybag in a variety of formats
        
        @param return_type: list of types to return including label,obj,ps,nchild,mask
        @type return_type: str or list of strings
        @param flat: whether to flatten to a 1d list
        @type flat: bool
        @return: the system structure
        @rtype: list or list of lists        
        """
        all_types = ['obj','ps','nchild','mask','label']
        
        # create empty list for all types, later we'll decide which to return
        struc = {}
        for typ in all_types:
            struc[typ]=[]
        
        if 'old_mask' in kwargs.keys() and 'mask' in return_type:
            # old_mask should be passed as a tuple of two flattened lists
            # the first list should be parametersets
            # the second list should be the old mask (booleans)
            # if 'mask' is in return_types, and this info is given
            # then any matching ps from the original ps will retain its old bool
            # any new items will have True in the mask
            
            # to find any new items added to the system structure
            # pass the old flattened ps output and a list of False of the same length
            
            old_mask = kwargs['old_mask']
            old_struclabel = old_mask[0]
            old_strucmask = old_mask[1]
            
        else:
            old_struclabel = [] # new mask will always be True
                
        if 'top_level' in kwargs.keys():
            item = kwargs.pop('top_level') # we don't want it in kwargs for the recursive call
        else:
            item = bundle.get_system()
            
        struc['obj'].append(item)
        itemlabel = self.bundle_get_label(item)
        struc['label'].append(itemlabel)
        struc['ps'].append(bundle.get_ps(item))
        
        # label,ps,nchild are different whether item is body or bodybag
        if hasattr(item, 'bodies'):
            struc['nchild'].append('2') # should not be so strict
        else:
            struc['nchild'].append('0')
            
        if itemlabel in old_struclabel: #then apply previous bool from mask
            struc['mask'].append(old_strucmask[old_struclabel.index(itemlabel)])
        else:
            struc['mask'].append(True)

        # recursively loop to get hierarchical structure
        children = bundle.get_children(item)
        if len(children) > 1:
            for typ in all_types:
                struc[typ].append([])
        for child in children:
            new = self.bundle_get_system_structure(bundle, return_type=all_types,flat=flat,top_level=child,**kwargs)
            for i,typ in enumerate(all_types):
                struc[typ][-1]+=new[i]

        if isinstance(return_type, list):
            return [list(utils.traverse(struc[rtype])) if flat else struc[rtype] for rtype in return_type]
        else: #then just one passed, so return a single list
            rtype = return_type
            return list(utils.traverse(struc[rtype])) if flat else struc[rtype]
        
    def str_includes(self,string,lst):
        for item in lst:
            if item in string:
                return True
        return False
    
    def gui_lock(self,command='',thread=None):
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
        if self.bundle.lock['locked']:
            return
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
        self.lp_progressStackedWidget.setCurrentIndex(0) #plot buttons
        self.rp_progressStackedWidget.setCurrentIndex(0) #plot buttons
        
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
            self.bundle.get_system().clear_synthetic()
           
            # and force redraw plots to clear all models
            self.on_plots_changed()
            
            # restore gui
            self.gui_unlock()
            
    def thread_failed(self, message):
        QMessageBox.information(None, "Error", str(message))
        
    def on_undo_clicked(self):
        self.undoStack.undo()
        
    def on_redo_clicked(self):
        self.undoStack.redo()

    def on_new_clicked(self,file_to_open=None):
        self.PyInterp_run("bundle = Bundle('%s')" % file_to_open if file_to_open is not None else "bundle = Bundle()",kind='sys',thread=False) # this will call on_new_bundle and reset to home/splash screen

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

        if self.bundle.get_system() is None: #then we have no bundle and should be on the splash
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
            #~ print "*** attaching set_time signal"
            #~ self.bundle.attach_signal(self.bundle.get_system(),'set_time',self.on_set_time)

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
                    print("gui locked")
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
            filename = QFileDialog.getOpenFileName(self, 'From Library', os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../parameters/library/'), ".par(*.par)", **_fileDialog_kwargs)  #need to choose this directory more carefully
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
            elif self.sender()==self.actionLegacy_PHOEBE:
                self.PyInterp_run('bundle = Bundle(system=parsers.legacy_to_phoebe(\'%s\', create_body=True))' % filename, kind='sys', thread=False)
                return
            else:
                self.PyInterp_run('bundle = load(\'%s\')' % filename, kind='sys', thread=False)
                self.filename = filename
            self.PyInterp_run('bundle.set_usersettings(settings)', kind='settings', thread=False)
                
    def on_save_clicked(self):
        """
        save a bundle to a .phoebe file
        this function handles both save and save as
        """
        if (self.sender() == self.tb_file_saveAction or (self.lock_pop is not None and self.sender() == self.lock_pop.save_button)) and self.filename is not None:
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
        self.PyInterp_run("bundle = Bundle('defaults.phoebe')", kind='sys', thread=False) 
    
    def splash_triple(self):
        """
        load the default triple
        this will result in a call to on_new_bundle
        """
        #this will obviously be reworked to be more general
        self.PyInterp_run('bundle = Bundle(system=create.KOI126())', write=True)
        
    def bundle_get_label(self,obj):
        """
        Get the label/name for any object (Body or BodyBag)
        
        @param obj: the object
        @type obj: Body or BodyBag
        @return: the label/name
        @rtype: str        
        """
        if isinstance(obj,str): #then probably already name, and return
            return obj
        
        objectname = None
        if hasattr(obj,'bodies'): #then bodybag
            #search for orbit in the children bodies
            for item in obj.bodies: # should be the same for all of them, but we'll search all anyways
                #NOTE: this may fail if you have different orbits for each component
                if 'orbit' in item.params.keys():
                    objectname = item.params['orbit']['label']
            return objectname
        else: #then hopefully body
            return obj.get_label()
        
    def bundle_get_system_structure(self,bundle,return_type='label',flat=False,**kwargs):
        """
        Get the structure of the system below any bodybag in a variety of formats
        
        @param return_type: list of types to return including label,obj,ps,nchild,mask
        @type return_type: str or list of strings
        @param flat: whether to flatten to a 1d list
        @type flat: bool
        @return: the system structure
        @rtype: list or list of lists        
        """
        all_types = ['obj','ps','nchild','mask','label']
        
        # create empty list for all types, later we'll decide which to return
        struc = {}
        for typ in all_types:
            struc[typ]=[]
        
        if 'old_mask' in kwargs.keys() and 'mask' in return_type:
            # old_mask should be passed as a tuple of two flattened lists
            # the first list should be parametersets
            # the second list should be the old mask (booleans)
            # if 'mask' is in return_types, and this info is given
            # then any matching ps from the original ps will retain its old bool
            # any new items will have True in the mask
            
            # to find any new items added to the system structure
            # pass the old flattened ps output and a list of False of the same length
            
            old_mask = kwargs['old_mask']
            old_struclabel = old_mask[0]
            old_strucmask = old_mask[1]
            
        else:
            old_struclabel = [] # new mask will always be True
                
        if 'top_level' in kwargs.keys():
            item = kwargs.pop('top_level') # we don't want it in kwargs for the recursive call
        else:
            item = bundle.get_system()
            
        struc['obj'].append(item)
        itemlabel = self.bundle_get_label(item)
        struc['label'].append(itemlabel)
        struc['ps'].append(bundle.get_ps(itemlabel))
        
        # label,ps,nchild are different whether item is body or bodybag
        if hasattr(item, 'bodies'):
            struc['nchild'].append('2') # should not be so strict
        else:
            struc['nchild'].append('0')
            
        if itemlabel in old_struclabel: #then apply previous bool from mask
            struc['mask'].append(old_strucmask[old_struclabel.index(itemlabel)])
        else:
            struc['mask'].append(True)

        # recursively loop to get hierarchical structure
        #~ print "*** searching for child with label", itemlabel
        children = bundle.get_children(itemlabel)
        if len(children) > 1:
            for typ in all_types:
                struc[typ].append([])
        for child in children:
            new = self.bundle_get_system_structure(bundle, return_type=all_types,flat=flat,top_level=child,**kwargs)
            for i,typ in enumerate(all_types):
                struc[typ][-1]+=new[i]

        if isinstance(return_type, list):
            return [list(utils.traverse(struc[rtype])) if flat else struc[rtype] for rtype in return_type]
        else: #then just one passed, so return a single list
            rtype = return_type
            return list(utils.traverse(struc[rtype])) if flat else struc[rtype]

    def get_system_structure(self, struclabel=None, strucsel=None):
        """
        get the hierarchical structure of the system while maintaining the selected items
        this gets passed on to the html system views
        """
        curr_struclabel = list(utils.traverse(struclabel)) #flattened lists of the incoming values before updating
        curr_strucsel = list(utils.traverse(strucsel))
        
        if self.bundle.get_system() is not None:
            strucnames, strucnchild, strucsel, strucps = self.bundle_get_system_structure(self.bundle, return_type=['label','nchild','mask','ps'],old_mask=(curr_struclabel, curr_strucsel))
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
        if self.bundle.get_system() is None:
            return
        # get obs and syn
        ds_obs_all = self.bundle.get_obs(all=True).values()
        ds_syn_all = self.bundle.get_syn(all=True).values()
        
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
        
        #~ print "*** update_datasets", len(trees), len(ds_obs), len(ds_syn)
        
        for tree in self.plotEntry_widgets:
            # filter which items to show (actual filtering is done by tree.set_data)
            if tree == self.datasetswidget_main.datasetTreeView:
                #then filter based off combos
                types = str(self.datasetswidget_main.ds_typeComboBox.currentText())
                plots = str(self.datasetswidget_main.ds_plotComboBox.currentText())
                if plots != 'all plots':
                    self.datasetswidget_main.ds_typeComboBox.setEnabled(False)
                else:
                    self.datasetswidget_main.ds_typeComboBox.setEnabled(True)
            else:
                types = 'all categories'
                plots = tree.plotindex
            
            tree.set_data(ds_obs,ds_syn,types,plots,self.bundle,self.system_ps,self.system_names)
            
        self.mp_stackedWidget_to_grid(force_grid=len(self.bundle._get_dict_of_section('axes'))!=0)

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
        sel_comps = [s for s in ps_sel if s.context != 'orbit']
        #~ sel_meshes = [self.bundle.get_mesh(s.get_value('label')) for s in sel_comps] # this is kind of ugly
        sel_meshes = [self.bundle.get_object(s['label']).params['mesh'] for s in sel_comps] # still ugly

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
            self.lp_compTreeView.set_data(sel_comps,style=[] if _devel_version else ['nofit'])
            self.lp_meshTreeView.set_data(sel_meshes,style=['nofit'])

        if len(sel_orbits) == 0:
            if not skip_collapse:
                self.lp_orbitPushButton.setChecked(False)
            self.lp_orbitTreeView.clear()
        else:
            if not skip_collapse:
                self.lp_orbitPushButton.setChecked(True)

            #update orbit tree view
            self.lp_orbitTreeView.set_data(sel_orbits,style=[] if _devel_version else ['nofit'])

        for i,item in enumerate(sel_orbits):
            self.lp_orbitTreeView.headerItem().setText(i+1, item['label'])
            self.lp_orbitTreeView.resizeColumnToContents(i+1)

        for i,item in enumerate(sel_comps):
            self.lp_compTreeView.headerItem().setText(i+1, item['label'])
            self.lp_compTreeView.resizeColumnToContents(i+1)
            self.lp_meshTreeView.headerItem().setText(i+1, item['label'])
            self.lp_meshTreeView.resizeColumnToContents(i+1)
            
        if self.bundle.get_system() is not None:
            #~ print "*** updating fitting treeviews", len(self.bundle.get_system().get_adjustable_parameters())
            self.rp_fitinTreeView.set_data(self.bundle.get_system().get_adjustable_parameters(),self.system_ps,self.system_names)
            self.rp_fitoutTreeView.set_data(self.bundle._get_dict_of_section('feedback'),self.system_ps,self.system_names)
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
            
    def attach_plot_signals(self, axes, i=0, canvas=None, skip_axes_attach=False):
        #~ print "*** attach_plot_signals", i
        if canvas is None:
            canvas = self.plot_canvases[i]
           
        for po in axes._get_dict_of_section('plot').values():
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
            self.bundle.attach_signal(self.bundle, 'reload_obs', self.plot_redraw, i, canvas)
            self.bundle.attach_signal(axes.sections['axes'][0], 'set_value', self.plot_redraw, i, canvas)
            #~ self.bundle.attach_signal(axes, 'set_value', self.plot_redraw, i, canvas)
            self.bundle.attach_signal(axes, 'add_plot', self.attach_plot_signals, i, canvas, True) # so that we can add the new parameter options
            self.bundle.attach_signal(axes, 'add_plot', self.plot_redraw, i, canvas)
            self.bundle.attach_signal(axes, 'remove_plot', self.plot_redraw, i, canvas)

            #~ self.attached_plot_signals.append((axes, canvas))
        else: # then we're coming from add_plot
            self.plot_redraw(None, i, canvas)
            
    def update_plot_widgets(self, i, canvas):
        #~ print "*** update_plot_widgets", i
        if not hasattr(canvas,'info') or 'xaxisCombo' not in canvas.info.keys():
            #~ print '* update_plot_widgets exiting'
            return
        canvas.info['xaxisCombo'].setEnabled(False)
        canvas.info['xaxisCombo'].clear()
        items = ['time']
        for name in self.bundle_get_system_structure(self.bundle,flat=True):
            ps = self.bundle.get_ps(name)
            if ps.context=='orbit':
                items.append('phase:%s' % name)
        canvas.info['xaxisCombo'].addItems(items)
        canvas.info['xaxisCombo'].setCurrentIndex(items.index(self.bundle.get_axes(i).get_value('xaxis')) if self.bundle.get_axes(i).get_value('xaxis') in items else 0)
        canvas.info['xaxisCombo'].setEnabled(True)
        
        #~ canvas.info['yaxisCombo'].setCurrentIndex()

        canvas.info['titleLinkButton'].setText(self.bundle.get_axes(i).get_value('title'))
            
    def plot_redraw(self, param=None, i=0, canvas=None):
        #~ print "*** redraw plot", i
        if param is not None and len(self.plot_canvases) != len(self.bundle._get_dict_of_section('axes')):
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
            canvas.plot(self.bundle, self.bundle.get_axes(i))
            #~ canvas.draw()
            
            if i==self.pop_i: # then this plot is in the expanded plot and we should also draw that
                self.expanded_canvas.cla()
                self.expanded_canvas.plot(self.bundle, self.bundle.get_axes(i))
                #~ self.expanded_canvas.draw()

    def on_plot_add(self, mesh=False, orbit=False, plotoptions=None):
        # add button clicked from gui
        if mesh==True and self.meshmpl_widget is None:
            self.meshmpl_widget, self.meshmpl_canvas = self.create_plot_widget(thumb=True,closable=False)
            #~ self.meshmpl_widget.setMaximumWidth(self.meshmpl_widget.height())
            self.mp_sysmplGridLayout.addWidget(self.meshmpl_widget, 0,0)
        elif orbit==True and self.orbitmpl_widget is None:
            self.orbitmpl_widget, self.orbitmpl_canvas = self.create_plot_widget(thumb=True,closable=False)
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
            title = 'Plot %d' % len(self.bundle._get_dict_of_section('axes'))+1
            add_command = "bundle.add_axes(category='%s', title='Plot %d')" % (plottype[:-3],title)
            remove_command = "bundle.remove_axes('%s')" % (title)
            command = phoebe_widgets.CommandRun(self.PythonEdit,add_command,remove_command,kind='plots',thread=False,description='add new plot')
            
            self.undoStack.push(command) 
            
    def create_plot_widget(self, canvas=None, thumb=False, closable=True):
        new_plot_widget = QWidget()
        if canvas is None:
            canvas = phoebe_plotting.MyMplCanvas(new_plot_widget, width=5, height=4, dpi=100, thumb=thumb, bg=str(self.palette.color(QPalette.Window).name()), hl=str(self.palette.color(QPalette.Highlight).name()))
        if thumb:
            QObject.connect(canvas, SIGNAL("plot_clicked"), self.on_plot_clicked)
            QObject.connect(canvas, SIGNAL("expand_clicked"), self.on_plot_expand_toggle)
            QObject.connect(canvas, SIGNAL("plot_delete"), self.on_plot_del)
            QObject.connect(canvas, SIGNAL("plot_pop"), self.on_plot_pop)
            overlay = phoebe_plotting.PlotOverlay(canvas,closable)
        else:
            QObject.connect(canvas, SIGNAL("plot_clicked"), self.on_plot_clicked)
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
            return
        elif self.sender() == self.orbitmpl_canvas:
            return
        if self.mp_stackedWidget.currentIndex()==1: #then we need to expand            
            # this should intelligently raise if on data tab (and then return on collapse)
            i = self.plot_canvases.index(self.sender())
            
            # make title and axes options available from canvas
            self.expanded_canvas.info = {'axes_i': i, 'xaxisCombo': self.expanded_plot.xaxisCombo, 'yaxisCombo': self.expanded_plot.yaxisCombo, 'titleLinkButton': self.expanded_plot.titleLinkButton, 'zoominButton': self.expanded_plot.zoom_in}
            self.update_plot_widgets(i,self.expanded_canvas)
            
            if i!= self.pop_i: # then we've changed plots and need to force a redraw
                self.plot_redraw(None, i, self.expanded_canvas)
                self.pop_i = i #to track this in case pop is clicked

            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(i+1)
            
            self.mp_stackedWidget.setCurrentIndex(2)
            
            self.expanded_canvas.cla()
            self.expanded_canvas.plot(self.bundle, self.bundle.get_axes(i))
            
            # instead of dealing with attaching and destroying signals for the expanded plot
            # let's always check to see if the expanded plot is the same plot as a plot that is being asked to be redrawn
            # we can access this info in plot_redraw() and on_select_time_changed() through self.pop_i
            
        else: #then we need to return to grid
            self.mp_stackedWidget_to_grid(force_grid=True)
            
            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(0)
            
        #~ self.update_datasets()
        
    def mp_stackedWidget_to_grid(self,force_grid=False):
        if len(self.bundle.get_obs(all=True))==0:
            # load data tutorial
            self.mp_stackedWidget.setCurrentIndex(5)
        elif len(self.bundle._get_dict_of_section('axes'))==0:
            # create plot tutorial
            self.mp_stackedWidget.setCurrentIndex(6)
        elif force_grid:
            self.mp_stackedWidget.setCurrentIndex(1)
        
    def on_plot_clicked(self,canvas,event):
        if not hasattr(canvas,'xaxis'):
            return
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
            
    def on_plot_zoom_in_toggled(self,state):
        # this function is called whenever the toggle button changes state
        # and just controls whether the recselect is active or not
        # after selecting a rectangle and releasing, the canvas will emit
        # a signal 'plot_zoom' which will then call the function on_plot_zoom_in
        
        button = self.sender()
        canvas = button.info['canvas']
        canvas.recselect.set_active(state)
            
    def on_plot_zoom_in(self,axes_i,xlim,ylim):
        # this function should be entered from the 'plot_zoom' signal
        # sent from the canvas when the recselect has been set
        
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        current_zoom = self.bundle.get_axes(axes_i).get_zoom()
        
        canvas = self.sender()
        canvas.info['zoominButton'].setChecked(False)
        
        do_command = "bundle.get_axes('%s').set_zoom(%s,%s)" % (axesname, xlim, ylim)
        undo_command = "bundle.get_axes('%s').set_zoom(%s)" % (axesname, current_zoom)
        description = "%s zoom in" % axesname
        self.on_param_command(do_command,undo_command,description='',kind='plots',thread=False)

            
    def on_plot_zoom_out_clicked(self,*args):
        # reset zoom to automatic values and make sure the zoom in button is disabled
        
        button = self.sender()
        button.info['canvas'].info['zoominButton'].setChecked(False)        
        axes_i = self.pop_i if button.info['axes_i'] == 'expanded' else button.info['axes_i']
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        current_zoom = self.bundle.get_axes(axes_i).get_zoom()
        
        do_command = "bundle.get_axes('%s').set_zoom((None,None),(None,None))" % axesname
        undo_command = "bundle.get_axes('%s').set_zoom(%s)" % (axesname, current_zoom)
        description = "%s zoom out" % axesname
        self.on_param_command(do_command,undo_command,description='',kind='plots',thread=False)
        
    def on_plot_pop_mpl_clicked(self, *args):
        button = self.sender()
        axes_i = self.pop_i if button.info['axes_i'] == 'expanded' else button.info['axes_i']
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        
        do_command = "bundle.plot_axes('%s')" % (axesname)
        undo_command = "print 'cannot undo plot axes'"
        description = "plot axes"
        self.on_param_command(do_command,undo_command,description='',kind='plots',thread=False)        
        
        do_command = "plt.show()"
        undo_command = "print 'cannot undo plt.show()'"
        description = "plot axes"
        self.on_param_command(do_command,undo_command,description='',kind='plots',thread=False)        
        
    def on_plot_save_clicked(self,*args):
        button = self.sender()
        axes_i = self.pop_i if button.info['axes_i'] == 'expanded' else button.info['axes_i']
        axesname = self.bundle.get_axes(axes_i).get_value('title')
        
        filename = QFileDialog.getSaveFileName(self, 'Export Axes Image', self.latest_dir if self.latest_dir is not None else './', **_fileDialog_kwargs)

        do_command = "bundle.save_axes('%s','%s')" % (axesname, filename)
        undo_command = "print 'cannot undo save axes'"
        description = "save axes"
        self.on_param_command(do_command,undo_command,description='',kind='plots',thread=False)

    @PyInterp_selfdebug
    def on_plot_pop(self,i=None):
        #plot popped from gui
        if i is None:
            i = self.pop_i
            self.on_plot_expand_toggle() #to reset to grid view

        new_plot_widget, canvas = self.create_plot_widget()
       
        #~ self.bundle.attach_signal(self.bundle.get_axes(i), 'add_plot', self.plot_redraw, i, canvas)
        #~ self.bundle.attach_signal(self.bundle.get_axes(i), 'remove_plot', self.plot_redraw, i, canvas)       
        #~ for po in self.bundle.get_axes(i).get_plot():
            #~ self.bundle.attach_signal(po, 'set_value', self.plot_redraw, i, canvas)
        
        pop = phoebe_dialogs.CreatePopPlot(self)
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
        canvas.info = {'axes_i': i, 'xaxisCombo': pop.xaxisCombo, 'yaxisCombo': pop.yaxisCombo, 'titleLinkButton': pop.titleLinkButton, 'zoominButton': pop.zoom_in}
        
        # zoom and save buttons
        pop.zoom_in.info = {'axes_i': i, 'canvas': canvas}
        pop.zoom_out.info = {'axes_i': i, 'canvas': canvas}
        pop.save.info = {'axes_i': i}
        
        QObject.connect(pop.zoom_in, SIGNAL("toggled(bool)"), self.on_plot_zoom_in_toggled)
        QObject.connect(canvas, SIGNAL("plot_zoom"), self.on_plot_zoom_in)
        QObject.connect(pop.zoom_out, SIGNAL("clicked()"), self.on_plot_zoom_out_clicked)
        QObject.connect(pop.save, SIGNAL("clicked()"), self.on_plot_save_clicked)
        
        plotEntryWidget = phoebe_dialogs.CreateDatasetWidget(parent=self)
        plotEntryWidget.datasetTreeView.plotindex = i
        plotEntryWidget.selectorWidget.setVisible(False)
        plotEntryWidget.addDataWidget.setVisible(False)
        QObject.connect(plotEntryWidget.datasetTreeView, SIGNAL("parameterCommand"), self.on_param_command)
        QObject.connect(plotEntryWidget.datasetTreeView, SIGNAL("focusIn"), self.on_paramfocus_changed)
        
        pop.treeviewLayout.addWidget(plotEntryWidget)        
        
        self.plot_redraw(None, i, canvas)
 
        self.attach_plot_signals(self.bundle.get_axes(i), i, canvas)
        
        pop.show()
        
        self.plotEntry_widgets.append(plotEntryWidget.datasetTreeView) # add to the list of treeviews to be updated when data or plots change
        self.paramTreeViews.append(plotEntryWidget.datasetTreeView) # add to the list of treeviews that only allow one item to be focused
        self.plotEntry_axes_i.append(i)
        
        #~ self.update_plotTreeViews(wa=zip([plotEntryWidget.bp_plotsTreeView],[i])) #just draw the new one, not all
        self.update_datasets()
        
    def on_plot_del(self,i=None):
        #plot deleted from gui
        if i is None:
            i = self.pop_i
            self.on_plot_expand_toggle()
        
        plottype = self.bundle.get_axes(i).get_value('category')
        axesname = self.bundle.get_axes(i).get_value('title')
        command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.remove_axes('%s')" % axesname,"bundle.add_axes(category='%s', title='%s')" % (plottype, axesname),kind='plots',thread=True,description='remove axes %s' % axesname)
        self.undoStack.push(command)  
        
    def on_systemEdit_clicked(self):
        self.mp_stackedWidget.setCurrentIndex(4)
        
    def update_servers_avail(self,update_prefs=False):
        if update_prefs:
            self.prefs = self.PyInterp_get('settings')
        servers_on = ['None']+[s.get_value('label') for s in self.prefs._get_dict_of_section('server').values() if s.last_known_status['status']]        
        for w in [self.lp_serverComboBox, self.rp_serverComboBox]:
            orig_text = str(w.currentText())
            w.clear()
            w.addItems(servers_on)
            if orig_text in servers_on:
                w.setCurrentIndex(servers_on.index(orig_text))
            w.setVisible(len(servers_on)>1)
        
    def update_observeoptions(self):
        currenttext = self.lp_methodComboBox.currentText()
        self.lp_methodComboBox.clear()
        for k,v in self.bundle._get_dict_of_section(section='compute').iteritems():
            self.lp_methodComboBox.addItem(k.split('@')[0])

        # return to original selection
        if len(currenttext) > 0: #ignore empty case
            self.lp_methodComboBox.setCurrentIndex(self.lp_methodComboBox.findText(currenttext))
        
    def on_observeOption_changed(self, *args):
        # get the correct parameter set and send to treeview
        combo = self.lp_methodComboBox
        key = str(combo.currentText())
        if len(key)==0: return
        #~ print "*** on_fittingOption_changed", key
        
        co = self.bundle.get_compute(key)
            
        self.lp_observeoptionsTreeView.set_data([co] if co is not None else [],style=['nofit','incl_label'],hide_params=['time','refs','types'])
        self.lp_observeoptionsTreeView.headerItem().setText(1, key)
        
        # set visibility of reset/delete buttons
        in_settings = key.split('@')[0] in self.prefs._get_dict_of_section('compute')
        self.lp_observeoptionsReset.setVisible(in_settings)
        self.lp_observeoptionsDelete.setVisible(in_settings==False)

    def on_observeOption_delete(self, *args):
        key = str(self.lp_methodComboBox.currentText())
        
        do_command = "bundle.remove_compute('%s')" % key
        undo_command = "bundle.add_compute(label='%s')" % key
        description = "remove %s compute options" % key
        
        self.on_param_command(do_command,undo_command,description='',thread=False,kind='system')
                
    def on_observeOption_add(self, *args):
        key = 'new compute'
        
        do_command = "bundle.add_compute(label='%s')" % key
        undo_command = "bundle.add_compute(label='%s')" % key
        description = "add new compute options"
        
        self.on_param_command(do_command,undo_command,description='',thread=False,kind='system')
                
    def on_observe_clicked(self):
        kind = str(self.lp_methodComboBox.currentText())
        server = str(self.lp_serverComboBox.currentText())

        self.bundle.purge_signals(self.bundle.attached_signals_system)
        params = self.bundle.get_compute(kind).copy()
        observatory.extract_times_and_refs(self.bundle.get_system(),params)
        self.set_time_is = len(params.get_value('time'))
        if server == 'None':
            self.PyInterp_run("bundle.run_compute('%s')" % (kind), kind='sys', thread=True)
        else:
            self.PyInterp_run("bundle.run_compute('%s',server='%s')" % (kind,server), kind='sys', thread=True)

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
        self.mesh_widget.setMesh(self.bundle.get_system().get_mesh()) # 3D view
        
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
        elif treeview==self.rp_fitoptionsTreeView:
            kind = 'fitting'

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
        
        # add prior if necessary   - no handled by bundle  
        #~ if is_adjust and newvalue == True and not param.has_prior(): #then we need to create an initial prior
            #~ lims = param.get_limits()
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.set_prior('%s@%s', distribution='uniform', lower=%s, upper=%s)" % (parname,label,lims[0],lims[1]),"bundle.remove_prior('%s@%s')" % (parname,label),thread=False,description='add default prior for %s@%s' % (label,parname))
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).get_parameter(\'%s\').set_prior(distribution=\'uniform\',lower=%s,upper=%s)' % (kind,labelstr,parname,lims[0],lims[1]),'bundle.get_%s(%s).get_parameter(\'%s\').remove_prior()' % (kind,labelstr,parname),thread=False,description='add default prior for %s:%s' % (label,parname))
            #~ self.undoStack.push(command)
        
        qualifier = '%s@%s' % (parname,label)
        if kind in ['compute','fitting']:
            # not meshview or orbitview because they are required to only have 1 item (no labels)
            qualifier += '@%s' % kind
        
        # change adjust/value if necessary
        if is_adjust:
            command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.set_adjust('%s', %s)" % (qualifier,newvalue), "bundle.set_adjust('%s', %s)" % (qualifier,oldvalue),thread=False,description='change adjust of %s to %s' % (qualifier,newvalue))
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).set_adjust(\'%s\', %s)' % (kind,labelstr,parname,newvalue), 'bundle.get_%s(%s).set_adjust(\'%s\', %s)' % (kind,labelstr,parname,oldvalue),thread=False,description='change adjust of %s to %s' % (kind,newvalue))
        else:
            command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.set_value('%s', %s)" % (qualifier,"%s" % newvalue if isinstance(newvalue,str) and "np." in newvalue else "'%s'" % newvalue),"bundle.set_value('%s', %s)" % (qualifier,"%s" % oldvalue if isinstance(oldvalue,str) and "np." in oldvalue else "'%s'" % oldvalue),thread=False,description='change value of %s' % (qualifier))
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(%s).set_value(\'%s\',%s)' % (kind,labelstr,parname,'%s' % newvalue if isinstance(newvalue,str) and 'np.' in newvalue else '\'%s\'' % newvalue),'bundle.get_%s(%s).set_value(\'%s\',%s)' % (kind,labelstr,parname,'%s' % oldvalue if isinstance(oldvalue,str) and 'np.' in oldvalue else '\'%s\'' % oldvalue),thread=False,description='change value of %s:%s' % (kind,parname))
        
        # change/add constraint
        if is_constraint:
            if newvalue.strip() == '':
                do_command = "bundle.get_ps(%s).remove_constraint('%s')" % (labelstr,parname)
            else:
                do_command = "bundle.get_ps(%s).add_constraint('{%s} = %s')" % (labelstr,parname,newvalue)
            if oldvalue.strip() == '':
                undo_command = "bundle.get_ps(%s).remove_constraint('%s')" % (labelstr,parname)
            else:
                undo_command = "bundle.get_ps(%s).add_constraint('{%s} = %s')" % (labelstr,parname,oldvalue)
        
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=False,description="change constraint on %s:%s" % (label,parname))
        
        self.undoStack.push(command)
        
        # change units
        if oldunit is not None and newunit is not None:
            command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.get_parameter('%s@%s').set_unit('%s')" % (parname,label,newunit),"bundle.get_parameter('%s@%s').set_unit('%s')" % (parname,label,oldunit),thread=False,description='change units of %s@%s' % (parname,label))
            #~ command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_unit(\'%s\')' % (kind,label,parname,newunit),'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_unit(\'%s\')' % (kind,label,parname,oldunit),thread=False,description='change value of %s:%s' % (kind,parname))
            self.undoStack.push(command)
            
    def on_param_command(self,do_command,undo_command,description='',thread=False,kind=None):
        """
        allows more flexible/customized commands to be sent from parameter treeviews
        """
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,thread=thread,description=description,kind=kind)
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
        command_do = "bundle.get_axes('%s').get_plot(%d).set_value('%s','%s')" % (axesname, plot_i, param, new_value)
        command_undo = "bundle.get_axes('%s').get_plot(%d).set_value('%s','%s')" % (axesname, plot_i, param, old_value)
        
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
        #~ import time
        #~ start_time = time.time()
        
        try:
            self.bundle = self.PyInterp_get('bundle')
            #~ self.prefs = self.PyInterp_get('settings')
        except KeyError:
            return
            
        if self.bundle is not None and self.bundle.get_system() is not None:
            self.update_system()  # TRY TO OPTIMIZE OR USE SIGNALS (costs ~0.15 s)
            self.on_fittingOption_changed()
            self.update_observeoptions()
            if self.mp_stackedWidget.currentIndex()==0:
                self.mp_stackedWidget_to_grid(force_grid=False)
                #~ self.mp_stackedWidget.setCurrentIndex(1)
            self.tb_file_saveasAction.setEnabled(True)
            self.lp_DockWidget.setVisible(self.tb_view_lpAction.isChecked())
            self.rp_fittingDockWidget.setVisible(self.tb_view_rpAction.isChecked())
            self.bp_datasetsDockWidget.setVisible(self.tb_view_datasetsAction.isChecked())
            self.lp_systemDockWidget.setVisible(self.tb_view_systemAction.isChecked())
            self.rp_versionsDockWidget.setVisible(self.tb_view_versionsAction.isChecked())
            self.bp_pyDockWidget.setVisible(self.tb_view_pythonAction.isChecked())
            
            # check whether system is uptodate
            #~ uptodate = self.bundle.get_uptodate()
            #~ self.lp_computePushButton.setEnabled(uptodate==True)
            
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
            self.versions_treeView.set_data(self.bundle._get_dict_of_section('version').values())
            self.rp_savedFeedbackTreeView.set_data(self.bundle._get_dict_of_section('feedback').values())
            
            # update plot mesh options - should probably move this
            self.sys_meshOptionsTreeView.set_data(self.bundle._get_dict_of_section('meshview').values(),style=['nofit'])
            self.sys_orbitOptionsTreeView.set_data(self.bundle._get_dict_of_section('meshview').values(),style=['nofit'])

            # bundle lock
            if self.bundle.lock['locked']:
                if self.lock_pop is None:
                    self.lock_pop = phoebe_dialogs.CreatePopLock(self,self.bundle.lock,self.bundle.get_server(self.bundle.lock['server']))
                    QObject.connect(self.lock_pop, SIGNAL("parameterCommand"), self.on_param_command)   
                    QObject.connect(self.lock_pop.save_button, SIGNAL("clicked()"), self.on_save_clicked)         
                    QObject.connect(self.lock_pop.saveas_button, SIGNAL("clicked()"), self.on_save_clicked)         
                    QObject.connect(self.lock_pop.new_button, SIGNAL("clicked()"), self.on_new_clicked)      
                    # all other signals handled in phoebe_dialogs.CreatePopLock subclass   
                else:
                    self.lock_pop.setup(self.bundle.lock,self.bundle.get_server(self.bundle.lock['server']))
                self.lock_pop.show()
                self.lock_pop.shown = True
                self.gui_lock()
            elif self.lock_pop is not None and self.lock_pop.shown:
                self.lock_pop.hide()
                self.lock_pop.shown = False
                self.gui_unlock()

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
        
        if 'settings' in command and self.prefs_pop is not None:
            self.prefs = self.PyInterp_get('settings')
            if self.prefs_pop is not None:
                self.prefs_pop.set_gui_from_prefs(self.prefs)
            
        # probably don't always have to do this (maybe only if settings or first call?)
        self.update_servers_avail()    
        
        if 'bundle=' in command in command:
            #then new bundle
            self.on_new_bundle()
            
        if 'run_compute' in command or 'server_get_results' in command or 'server_loop' in command:
            #~ self.meshmpl_canvas.plot_mesh(self.bundle.get_system())
        
            #should we only draw this if its visible?
            #~ self.mesh_widget.setMesh(self.bundle.get_system().get_mesh())
            self.on_plots_changed()
            
        if 'run_fitting' in command:
            #~ self.lp_stackedWidget.setCurrentIndex(1) #compute
            self.rp_stackedWidget.setCurrentIndex(1) #feedback
            

            
        #### TESTING ###
        ## EVENTUALLY RUN ONLY WHEN NEEDED THROUGH SIGNAL OR OTHER LOGIC
        ## OR NEED TO SIGNIFICANTLY OPTIMIZE
        ## this is causing the majority of lag (0.4 s)
        self.update_datasets()
        
        #~ end_time = time.time()
        #~ print "!!!", end_time - start_time
        
    def on_axes_add(self,category,objref,dataref):
        # signal received from dataset treeview with info to create new plot
        title = 'Plot %d' % (len(self.bundle._get_dict_of_section('axes'))+1)
        
        do_command = "bundle.add_axes(category='%s', title='%s')" % (category,title)
        undo_command = "bundle.remove_axes('%s')" % (title)
        command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='plots',thread=False,description='add new axes')
        self.undoStack.push(command)
        
        for context_kind in ['obs','syn']:
            if (context_kind=='obs' and len(self.bundle.get_obs(objref=objref,dataref=dataref,all=True)) > 0 and phoebe_widgets.has_ydata(self.bundle.get_obs(objref=objref,dataref=dataref,all=True).values()[0])) or (context_kind=='syn' and len(self.bundle.get_syn(objref=objref,dataref=dataref,all=True).values()) > 0):
                do_command = "bundle.get_axes('%s').add_plot(type='%s%s',objref='%s',dataref='%s')" % (title,category,context_kind,objref,dataref)
                undo_command = "bundle.get_axes('%s').remove_plot(0)" % (title)
                command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='plots',thread=False,description='add new plot')
                self.undoStack.push(command)
        
        self.on_axes_goto(title)
                    
    def on_axes_goto(self,plotname=None):
        # signal receive from dataset treeview to goto a plot (axes) by name
        self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(self.bundle._get_dict_of_section('axes', kind='Container').keys().index(plotname)+1)        
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
        items = ['all plots']+self.bundle._get_dict_of_section('axes').keys()
        self.datasetswidget_main.ds_plotComboBox.addItems(items)
        if currentText in items:
            self.datasetswidget_main.ds_plotComboBox.setCurrentIndex(items.index(currentText))
        self.datasetswidget_main.ds_plotComboBox.setEnabled(True)
        
        
    def on_plots_changed(self,*args):
        #~ print "*** on_plots_changed", len(self.bundle.get_axes(return_type='list'))

        #bundle.axes changed - have to handle adding/deleting/reordering
        #for now we'll just completely recreate all thumbnail widgets and redraw
        
        #clear all canvases
        self.on_plot_clear_all() 
        
        self.on_plots_rename() # to update selection combo
        
        #redraw all plots
        for i,axes in enumerate(self.bundle._get_dict_of_section('axes', kind='Container').values()):
                
            new_plot_widget, canvas = self.create_plot_widget(thumb=True)
            canvas.info['axes_i'] = i
            self.plot_widgets.append(new_plot_widget)
            self.plot_canvases.append(canvas)
            #TODO change to horizontals in verticals so we don't have empty space for odd numbers
            num = len(self.plot_widgets)
            rows = 2 if num % 2 == 0 else 3
            for j,widget in enumerate(self.plot_widgets):
                row = j % rows
                col = j - row
                self.mp_plotGridLayout.addWidget(widget, row, col)

            # create hooks
            self.attach_plot_signals(axes, i, canvas)
        
        for i in range(len(self.bundle._get_dict_of_section('axes').values())):    
            self.plot_redraw(None,i)
            
    def on_select_time_changed(self,param=None,i=None,canvas=None):
        canvas.update_select_time(self.bundle)

        # check to see if this is the same plot as the expanded plot
        # and if so, also update that canvas
        if i==self.pop_i:
            self.expanded_canvas.update_select_time(self.bundle)
            
            
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
                
            self.lp_progressBar.setValue(int(float(self.set_time_i+1)/self.set_time_is*100))
            #~ self.rp_progressBar.setValue(int(float(self.set_time_i+1)/self.set_time_is*100))
            self.set_time_i += 1
            
            
        #~ if True:
            #~ self.on_plots_changed()
            #~ self.meshmpl_canvas.plot_mesh(self.bundle.get_system())
        
            #should we only draw this if its visible?
            #~ self.mesh_widget.setMesh(self.bundle.get_system().get_mesh())
            

    def update_fittingOptions(self, *args):
        currenttext = self.rp_methodComboBox.currentText()
        self.rp_methodComboBox.clear()
        for k,v in self.bundle._get_dict_of_section('fitting').iteritems():
            self.rp_methodComboBox.addItem(k.split('@')[0])

        # return to original selection
        if len(currenttext) > 0: #ignore empty case
            self.rp_methodComboBox.setCurrentIndex(self.rp_methodComboBox.findText(currenttext))
        
    def on_fittingOption_changed(self, *args):
        # get the correct parameter set and send to treeview
        combo = self.rp_methodComboBox
        key = str(combo.currentText())
        if len(key)==0: return
        #~ print "*** on_fittingOption_changed", key

        fitting = self.bundle.get_fitting(key)
            
        self.rp_fitoptionsTreeView.set_data([fitting],style=['nofit','incl_label'])
        self.rp_fitoptionsTreeView.headerItem().setText(1, key)
        
        # set visibility of reset/delete buttons
        in_settings = key.split('@')[0] in self.prefs._get_dict_of_section('fitting')
        self.rp_fitoptionsReset.setVisible(in_settings)
        self.rp_fitoptionsDelete.setVisible(in_settings==False)

    def on_fittingOption_delete(self, *args):
        key = str(self.rp_methodComboBox.currentText())
        
        do_command = "bundle.remove_fitting('%s')" % key
        undo_command = "bundle.add_fitting(label='%s')" % key
        description = "remove %s fitting options" % key
        
        self.on_param_command(do_command,undo_command,description='',thread=False,kind='sys')
        
    def on_fitoptions_param_changed(self,treeview,label,param,oldvalue,newvalue,oldunit=None,newunit=None,is_adjust=False,is_prior=False):
        #~ print "*** on_fitoptions_param_changed", label, param, oldvalue, newvalue
        #override label
        #~ label = str(self.rp_methodComboBox.currentText())
        paramname = param.get_qualifier()
        
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
            new_prior +=', %s=%f' % (key,new_dist[1][key])
        for key in old_dist[1].keys():
            old_prior +=', %s=%f' % (key,old_dist[1][key])
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,"bundle.set_prior('%s@%s', %s)" % (paramname,label,new_prior),"bundle.set_prior('%s@%s', %s)" % (paramname,label,old_prior),thread=False,description='change prior of %s@%s' % (paramname,label))
        #~ command = phoebe_widgets.CommandRun(self.PythonEdit,'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_prior(%s)' % (kind,label,paramname,new_prior),'bundle.get_%s(\'%s\').get_parameter(\'%s\').set_prior(%s)' % (kind,label,paramname,old_prior),thread=False,description='change prior of %s:%s' % (label,paramname))
        self.undoStack.push(command)
        
    def on_fit_clicked(self):
        label = str(self.rp_methodComboBox.currentText())
        server = str(self.rp_serverComboBox.currentText())

        if server == 'None':
            self.PyInterp_run("bundle.run_fitting('Compute', '%s')" % (label),thread=True,kind='sys')
        else:
            self.PyInterp_run("bundle.run_fitting('Compute', '%s', server='%s')" % (label,server),thread=True,kind='sys')
        
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
        
    def on_prefsShow(self):
        self.prefs = self.PyInterp_get('settings')
        if self.prefs_pop is None:
            self.prefs_pop = phoebe_dialogs.CreatePopPrefs(self,self.prefs,devel_version=_devel_version)
            QObject.connect(self.prefs_pop, SIGNAL("parameterCommand"), self.on_param_command)
            QObject.connect(self.prefs_pop.buttonBox, SIGNAL("accepted()"), self.on_prefsSave)
            QObject.connect(self.prefs_pop.buttonBox, SIGNAL("rejected()"), self.on_prefsUpdate)
            QObject.connect(self.prefs_pop.serverlist_treeWidget, SIGNAL("serverStatusChanged"), self.update_servers_avail)
        self.prefs_pop.show()
        
    def on_prefsSave(self,*args):
        command_do = "settings.save()"
        command_undo = ""
        
        command = phoebe_widgets.CommandRun(self.PythonEdit,command_do, command_undo, thread=False, description='save settings', kind='settings')
        
        self.undoStack.push(command)   
        
        self.on_prefsUpdate()
    
    @PyInterp_selfdebug
    def on_prefsUpdate(self,startup=False):
        try:
            self.prefs = self.PyInterp_get('settings')
        except KeyError:
            self.prefs = Settings()
        
        p = self.prefs.get_gui()
        
        if startup:
            # only apply default panels at startup (not immediately after changing preference)
            self.tb_view_lpAction.setChecked(p.get_value('panel_params'))
            self.tb_view_rpAction.setChecked(p.get_value('panel_fitting'))
            self.tb_view_versionsAction.setChecked(p.get_value('panel_versions'))
            self.tb_view_systemAction.setChecked(p.get_value('panel_system'))
            self.tb_view_datasetsAction.setChecked(p.get_value('panel_datasets'))
            self.tb_view_pythonAction.setChecked(p.get_value('panel_python'))

        self.PythonEdit.thread_enabled = p.get_value('pyinterp_thread_on')
        self.PythonEdit.write_sys = p.get_value('pyinterp_tutsys')
        self.PythonEdit.write_plots = p.get_value('pyinterp_tutplots')
        self.PythonEdit.write_settings = p.get_value('pyinterp_tutsettings')
        
        # THIS PROBABLY DOESN'T WORK
        #~ for pluginpath in p['plugins'].keys():
            #~ if pluginpath not in self.pluginsloaded and p['plugins'][pluginpath]:
                #~ # need to load and enable
                #~ self.pluginsloaded.append(pluginpath)
                #~ self.plugins.append(imp.load_source(os.path.basename(pluginpath).strip('.py'), pluginpath))
                #~ self.guiplugins.append(self.plugins[-1].GUIPlugin())
                #~ self.plugins[i].enablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)
            #~ if pluginpath in self.pluginsloaded:
                #~ i = self.pluginsloaded.index(pluginpath)
                #~ if pluginpath in p['plugins'][pluginpath]:
                    #~ # then already loaded, but need to enable
                    #~ self.plugins[i].enablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)
                #~ else:
                    #~ #then already loaded, but need to disable
                    #~ self.plugins[i].disablePlugin(self.guiplugins[i], self, gui.Ui_PHOEBE_MainWindow)

        #~ from pyphoebe.plugins import keplereb
        #~ self.keplerebplugin = keplereb.GUIPlugin()
        #~ keplereb.GUIPlugin.enablePlugin(self.keplerebplugin, self, gui.Ui_PHOEBE_MainWindow)
        #~ keplereb.GUIPlugin.disablePlugin(self.keplerebplugin)
            
    def on_aboutShow(self):
        pop = phoebe_dialogs.CreatePopAbout(self)
        pop.show()

    def on_helpShow(self):
        pop = phoebe_dialogs.CreatePopHelp(self)
        pop.show()
        
    def on_fileEntryShow(self):
        pop = phoebe_dialogs.CreatePopFileEntry(self,devel_version=_devel_version)
        pop.show()

        QObject.connect(pop.buttonBox.buttons()[1], SIGNAL("clicked()"), pop.close)
        QObject.connect(pop.buttonBox.buttons()[0], SIGNAL("clicked()"), self.on_pfe_okClicked)
                
    def on_pfe_okClicked(self):
        pop = self.sender().topLevelWidget()
        passband_filterset = str(pop.pfe_filtersetComboBox.currentText())
        passband_passband = str(pop.pfe_filterbandComboBox.currentText())
        passband = '{}.{}'.format(passband_filterset, passband_passband)
        
        name = pop.name.text() if len(pop.name.text()) > 0 else None
        
        if '--Passband--' in passband or '--Filter Set--' in passband or len(passband_filterset)==0 or len(passband_passband)==0:
            QMessageBox.information(None, "Warning", "Cannot load data: no passband provided")  
            return

        if pop.pfe_fileChooserButton.isVisible(): # then load_data
            filename=str(pop.pfe_fileChooserButton.text())

            if filename=='Choose File':
                QMessageBox.information(None, "Warning", "Cannot load data: either choose file or setup synthetic dataset")
                return
            
            columns = [str(colwidget.type_comboBox.currentText()) if '--' not in str(colwidget.type_comboBox.currentText()) else None for colwidget in pop.colwidgets]
            units_all = [str(colwidget.units_comboBox.currentText()) if ('--' not in str(colwidget.units_comboBox.currentText()) and colwidget.units_comboBox.count()>1) else None for colwidget in pop.colwidgets]
            components = [str(colwidget.comp_comboBox.currentText()) if '--' not in str(colwidget.comp_comboBox.currentText()) else None for colwidget in pop.colwidgets]
            
            units = {}
            for u,col in zip(units_all, columns):
                if u is not None:
                    units[col] = u
            
            #TODO make this more intelligent so values that weren't changed by the user aren't sent
            for i,colwidget in enumerate(pop.colwidgets):
                typecombo = colwidget.type_comboBox
                unitscombo = colwidget.units_comboBox
                compcombo = colwidget.comp_comboBox
                colcombo = colwidget.col_comboBox
                
                if typecombo.currentIndex()!=0:
                    if compcombo.isEnabled() and compcombo.currentIndex()==0:
                        QMessageBox.information(None, "Warning", "Cannot load data: no component for column %d" % (i+1))  
                        return

            do_command = "bundle.load_data(category='%s', filename='%s', passband='%s', columns=%s, objref=%s, units=%s, dataref='%s')" % (pop.category, filename, passband, columns, components, units, name)
            undo_command = "bundle.remove_data(ref='%s')" % name
            description = "load %s dataset" % name
            
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='sys',thread=False,description=description)
            self.undoStack.push(command)
            
        else: # then create_data
            
            # determine times
            if pop.times_match.isChecked():
                dataref = str(pop.datasetComboBox.currentText())
                timestr = "bundle.get_syn(dataref='%s',all=True).values()[0]['time']" % dataref
            elif pop.times_arange.isChecked():
                timestr = "np.arange(%f,%f,%f)" % (pop.arange_min.value(),pop.arange_max.value(),pop.arange_step.value())
            elif pop.times_linspace.isChecked():
                timestr = "np.linspace(%f,%f,%d)" % (pop.linspace_min.value(),pop.linspace_max.value(),pop.linspace_num.value())
            else: # custom list
                timestr = str(pop.time_custom.text())
                
            # determine components
            components = [comp for comp,check in pop.syn_components_checks.items() if check.isChecked()]
            if len(components) == 0: components = 'None'
            
            timeorphase = str(pop.times_timeorphase.currentText())
            
            do_command = "bundle.create_data(category='%s', %s=%s, objref=%s, passband='%s', dataref='%s')" % (pop.category,timeorphase,timestr,components,passband,name)
            undo_command = "bundle.remove_data(ref='%s')" % name
            description = "create %s synthetic dataset" % name
            
            command = phoebe_widgets.CommandRun(self.PythonEdit,do_command,undo_command,kind='sys',thread=False,description=description)
            self.undoStack.push(command)
            
        pop.close()
            
    def on_pfe_refreshClicked(self):
        pop = self.sender().topLevelWidget()
        # reload clicked inside popup - just refresh the data in the window but don't apply until ok clicked
        # reload clicked from main window tree view - immediately refresh data object 

    def main(self):
        self.show()


def set_gui_parameters():
    """
    Tweak some parameters to have slightly different behaviour for the GUI.
    """
    defs = parameters.defs.defs
    Npars = len(defs)
    for i in range(Npars):
        if defs[i]['qualifier'] == 'atm' and 'phoebe' in defs[i]['frame']:
            defs[i]['cast_type'] = 'choose'
            defs[i]['choices'] = ['kurucz', 'blackbody']
    

### startup the main window and pass any command-line arguments
if __name__=='__main__':
    STDOUT, STDERR, DEBUG = False, False, False
    #~ STDOUT, STDERR = True, True
    argv = sys.argv
    if "stdout" in argv:
        STDOUT = True
        argv.remove('stdout')
    if "stderr" in argv:
        STDERR = True
        argv.remove('stderr')
    if "debug" in argv:
        DEBUG = True
        argv.remove('debug')
    file_to_open = argv[-1] if len(argv)==2 else None
    set_gui_parameters()
    app = QApplication(sys.argv)
    font = app.font()
    palette = app.palette()
    phoebegui = PhoebeGUI((STDOUT, STDERR),DEBUG,font,palette,file_to_open=file_to_open)
    phoebegui._fileDialog_kwargs = _fileDialog_kwargs
    phoebegui.main()
    app.exec_()
