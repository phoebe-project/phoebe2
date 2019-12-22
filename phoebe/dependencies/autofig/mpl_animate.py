import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import LineCollection, PolyCollection
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation

from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls loop>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    """
    adapted from: http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/

    This function converts and animation object from matplotlib into HTML which can then
    be embedded in an IPython notebook.

    This requires ffmpeg to be installed in order to build the intermediate mp4 file

    To get these to display automatically, you need to set animation.Animation._repr_html_ = plotlib.anim_to_html
    (this is done on your behalf by PHOEBE)
    """
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)

# setup hooks for inline animations in IPython notebooks
try:
    from JSAnimation import IPython_display
except:
    # built-in mp4 support
    animation.Animation._repr_html_ = anim_to_html


class Animation(object):
    def __init__(self, affig, tight_layout=True,
                 draw_sidebars=True, draw_title=True,
                 subplot_grid=None, animate_callback=None):
        self.affig = affig
        self.mplfig = affig._get_backend_object()
        self.mplfig.clf()

        self.tight_layout = tight_layout
        self.draw_sidebars = draw_sidebars
        self.draw_title = draw_title
        self.subplot_grid = subplot_grid
        self.animate_callback = animate_callback

    def anim_init(self):
        return self.affig._get_backend_artists()

    def __call__(self, i):
        # print("***Animation.__call__(indep={})".format(indep))
        for mplax in self.mplfig.axes:
            mplax.cla()

        mplfig = self.affig.draw(i=i,
                                 tight_layout=self.tight_layout,
                                 draw_sidebars=self.draw_sidebars,
                                 draw_title=self.draw_title,
                                 subplot_grid=self.subplot_grid,
                                 in_animation=i+1)

        if self.animate_callback is not None:
            self.animate_callback(mplfig)

        return self.affig._get_backend_artists()
