import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QTAgg)
from matplotlib.widgets import RectangleSelector as selector
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
        
        self.mplaxes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        #~ self.axes.hold(False)
    
        #~ if not thumb:
            #~ self.mpl_toolbar = NavigationToolbar(self, parent)
            
        self.info = {}
            
        self.mpl_connect ("button_release_event", self.on_button_released)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        self.recselect = None # will be created on first plot

        self.thumb = thumb
        
    def enterEvent(self, event=None):
        super(MyMplCanvas, self).enterEvent(event)
        if self.thumb:
            self.overlay.setVisible(True)

    def leaveEvent(self, event=None):
        super(MyMplCanvas, self).leaveEvent(event)
        if self.thumb:
            self.overlay.setVisible(False)
            
    def initiate_recselect(self):
        #~ if self.recselect is None:
        if True:
            self.recselect = selector (self.fig.gca(), self.on_zoom_complete, drawtype='box', useblit=True, minspanx=5, minspany=5, spancoords='pixels')
            self.recselect.set_active(False)

    def on_button_released(self, event):
        if self.recselect.get_active():
            # ignore if coming from recselect
            return
        self.emit(SIGNAL("plot_clicked"),self,event)
        
    def on_expand_clicked(self, *args):
        self.emit(SIGNAL("expand_clicked"))
        
    def on_pop_clicked(self, *args):
        self.emit(SIGNAL("plot_pop"),self.info['axes_i'])
        
    def on_delete_clicked(self, *args):
        self.emit(SIGNAL("plot_delete"),self.info['axes_i'])
        
    def on_zoom_complete(self, event1, event2):
        xmin, xmax = min (event1.xdata, event2.xdata), max (event1.xdata, event2.xdata)
        ymin, ymax = min (event1.ydata, event2.ydata), max (event1.ydata, event2.ydata)
        self.emit(SIGNAL("plot_zoom"),self.info['axes_i'],(xmin,xmax),(ymin,ymax))
        
    def plot_mesh(self, bundle):
        #~ self.axes.cla()
        self.fig.clf()
        #~ ax = self.fig.gca()
        #~ ax.get_xaxis().set_visible(False)
        #~ ax.axes.get_yaxis().set_visible(False)
        #~ system.plot2D(ax=ax)
        bundle.plot_meshview(mplfig=self.fig)
        self.draw()
        
    def plot_orbit(self, bundle):
        self.fig.clf()
        bundle.plot_orbitview(mplfig=self.fig)
        #~ self.check_ticks()
        self.draw()

    def cla(self):
        self.mplaxes.cla()
    
    def plot(self, bundle, axes):
        self.fig.clf()
        axes.plot(bundle, mplfig=self.fig) 
        self.xaxis = axes.get_value('xaxis')
        self.period = axes.period
        self.axes = axes # this is the bundle axes not mpl axes
        
        self.check_ticks()
        self.initiate_recselect()
        
        self.draw()
        
    def update_select_time(self, bundle):
        self.axes.plot_select_time(bundle, bundle.select_time, self.fig)
        self.check_ticks()
        self.draw()

    def check_ticks(self):
        if self.thumb:
            # hide tick marks
            ax = self.fig.gca()
            ax.set_xticklabels(['']*len(ax.get_xticks()))
            self.fig.data_axes.set_yticklabels(['']*len(self.fig.data_axes.get_yticks()))
            
class PlotOverlay(QWidget):
    def __init__(self, parent=None, closable=True):
        QWidget.__init__(self, parent)
        parent.overlay = self
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)
        #~ self.setAlignment(Qt.AlignHRight|Qt.AlignVCenter)
        self.setVisible(False)
        
        hbox = QHBoxLayout(self)
        hbox.setSpacing(2)
        hbox.setMargin(0)
        
        #~ print parent.size().width()-3*22
        #~ spacer = QSpacerItem(parent.size().width()-3*22, 28, QSizePolicy.Expanding, QSizePolicy.Minimum)
        #~ hbox.addItem(spacer)    

        if closable:
            delete_icon = QIcon()
            delete_icon.addPixmap(QPixmap(":/images/icons/delete.png"), QIcon.Normal, QIcon.Off)

            delete_button = QPushButton()
            delete_button.setIcon(delete_icon)
            delete_button.setToolTip('delete axes')
            delete_button.setIconSize(QSize(16, 16))
            delete_button.setMaximumSize(QSize(22, 22))
            
            #~ delete_button.setEnabled(False)
            QObject.connect(delete_button, SIGNAL("clicked()"), parent.on_delete_clicked)
            
            hbox.addWidget(delete_button)
        
        pop_icon = QIcon()
        pop_icon.addPixmap(QPixmap(":/images/icons/pop.png"), QIcon.Normal, QIcon.Off)

        pop_button = QPushButton()
        pop_button.setIcon(pop_icon)
        pop_button.setToolTip('popout')
        pop_button.setIconSize(QSize(16, 16))
        pop_button.setMaximumSize(QSize(22, 22))
        
        #~ pop_button.setEnabled(False)
        QObject.connect(pop_button, SIGNAL("clicked()"), parent.on_pop_clicked)
        
        hbox.addWidget(pop_button)

        exp_icon = QIcon()
        exp_icon.addPixmap(QPixmap(":/images/icons/expand.png"), QIcon.Normal, QIcon.Off)
        
        exp_button = QPushButton()
        exp_button.setIcon(exp_icon)
        exp_button.setToolTip('expand plot')
        exp_button.setIconSize(QSize(16, 16))
        exp_button.setMaximumSize(QSize(22, 22))
        
        QObject.connect(exp_button, SIGNAL("clicked()"), parent.on_expand_clicked)
        
        hbox.addWidget(exp_button)
