import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QTAgg)
from matplotlib.figure import Figure

# Qt4
from PyQt4.QtCore import *
from PyQt4.QtGui import *

Signal = pyqtSignal
Slot = pyqtSlot
Property = pyqtProperty

# PySide
#~ from PySide import QtGui, QtCore
#~ # matplotlib.rcParams['backend.qt4']='PySide'

import numpy, os

class MyMplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, thumb=False, bg='#F6F4F2', hl=''):

        self.width, self.height, self.dpi = width, height, dpi
        
        self.fig = Figure(figsize=(width, height), dpi=dpi,facecolor=bg)
        
        super(MyMplCanvas, self).__init__(self.fig)
        #~ FigureCanvas.__init__(self, self.fig)
        #~ self.setParent(parent)
        
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        #~ self.axes.hold(False)
    
        #~ if not thumb:
            #~ self.mpl_toolbar = NavigationToolbar(self, parent)
            
        self.mpl_connect ("button_release_event", self.on_button_released)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.thumb = thumb
        self.hover_effect = QGraphicsColorizeEffect()
        color = QColor()
        color.setNamedColor(hl)
        self.hover_effect.setColor(color)
        self.setGraphicsEffect(self.hover_effect)
        self.hover_effect.setEnabled(False)
        
    def enterEvent(self, event=None):
        super(MyMplCanvas, self).enterEvent(event)
        #~ self.emit(SIGNAL("plot_hover"),self,True)
        if self.thumb:
            self.hover_effect.setEnabled(True)
            #~ self.fig.set_facecolor('#85A3FF')        
            #~ self.axes.set_axis_bgcolor('#85A3FF')
            #~ self.replot()
        # highlight and show buttons

    def leaveEvent(self, event=None):
        super(MyMplCanvas, self).leaveEvent(event)
        #~ self.emit(SIGNAL("plot_hover"),self,Flase)
        if self.thumb:
            self.hover_effect.setEnabled(False)
            #~ self.fig.set_facecolor('white')        
            #~ self.axes.set_axis_bgcolor('white')
            #~ self.replot()
        # unhighlight/hide buttons

    def on_button_released(self, event):
        self.emit(SIGNAL("plot_clicked"),self,event)
        #~ print "***", 
        #~ pass

    def plot_mesh(self, bundle):
        #~ self.axes.cla()
        self.fig.clf()
        #~ ax = self.fig.gca()
        #~ ax.get_xaxis().set_visible(False)
        #~ ax.axes.get_yaxis().set_visible(False)
        #~ system.plot2D(ax=ax)
        bundle.plot_mesh(mplfig=self.fig)
        self.draw()

    def cla(self):
        self.axes.cla()
    
    def plot(self, bundle, axes):
        #~ self.fig = Figure(figsize=(self.width, self.height), dpi=self.dpi,facecolor='#F6F4F2')
        self.fig.clf()
        axes.plot(bundle, mplfig=self.fig) 
        self.xaxis = axes.get_value('xaxis')
        self.period = axes.period
        # note axes is not the same as self.axes
        
        #~ if not isinstance(plotoptions, list):
            #~ plotoptions = [plotoptions]
        #~ for po in plotoptions:
            #~ plotting.plot(system, po, self.axes)
            #~ bundle.plot(po, self.axes)
            #~ plotting.plot(system, po, self.axes)
        self.draw()
