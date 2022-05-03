class DraggableLine:

    def __init__(self, p):
        '''
        Enables manual adjustment of line positions corresponding to
        the eclipse positions and edges.

        Parameters
        ----------
        p: matplotlib ax.axvline component to be adjusted
        '''
        self.point = p
        self.press = None

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

    def disconnect(self):
        #disconnect all the stored connection ids
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


    def button_press_event(self,event):
        if event.inaxes != self.point.axes:
            return
        contains = self.point.contains(event)[0]
        if not contains: return
        self.press = self.point.get_xdata(), event.xdata

    def button_release_event(self,event):
        self.press = None
        self.point.figure.canvas.draw()

    def motion_notify_event(self, event):
        if self.press is None: return
        if event.inaxes != self.point.axes: return
        xdata, xpress = self.press
        dx = event.xdata-xpress
        self.point.set_xdata(xdata+dx)
        self.point.figure.canvas.draw()
