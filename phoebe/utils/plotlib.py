"""
Plotting and multimedia routines.
"""
import os
import glob
import subprocess
import logging
import numpy as np
from scipy import ndimage
from phoebe.utils import decorators
try:
    import matplotlib as mpl
except ImportError:
    print("Soft warning: matplotlib could not be found on your system, 2D plotting is disabled, as well as IFM functionality")

logger = logging.getLogger('UTLS.PLOTLIB')

def fig2data(fig, grayscale=False):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    
    Stolen from http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image
    
    But I needed to invert the height and width.
    
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
  
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    
    if grayscale:
        #-- only return the B layer (which is equal to the R and G layer)
        return buf[:,:,-1]
    else:
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
        # have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf


@decorators.memoized
def read_bitmap(system, image_file, res=1, scale=None, hshift=0, vshift=0, invert=False):
    """
    Read a bitmap to match the coordinates of a system.
    """
    #-- get the coordinates in the original frame of reference.
    r,phi,theta = system.get_coords()
    #-- read in the data
    data = mpl.pyplot.imread(image_file)[::res,::res]
    logger.info('Read in image with resolution {}x{}'.format(*data.shape))
    #   convert color images to gray scale
    if len(data.shape)>2:
        data = data.mean(axis=2).T
    else:
        data = data.T
    data = data[::-1]
    if vshift:
        data = np.roll(data, vshift, axis=1)
    if hshift:
        data = np.roll(data, hshift, axis=0)
    
    # Rescale data if necessary:
    if invert:
        data = 1 - data
    if scale is not None:
        dvmin = data.min()
        dvmax = data.max()
        data = (data-dvmin)/(dvmax-dvmin)*(scale[1]-scale[0]) + scale[0]
    #-- normalise the coordinates so that we can map the coordinates
    PHI = (phi+np.pi)/(2*np.pi)*data.shape[0]
    THETA = (theta)/np.pi*data.shape[1]
    vals = np.array(ndimage.map_coordinates(data,[PHI,THETA], mode='nearest', order=1),float)
    #-- fix edges of image
    #vals[PHI>0.99*PHI.max()] = 1.0
    #vals[PHI<0.01*PHI.max()] = 1.0
    return vals
    

def blackbody_cmap():
    clist = [(1.0, 0.23402839584364535, 0.010288477175185372),
             (1.0, 0.27688646625836139, 0.027629287420684493),
             (1.0, 0.31859998424019975, 0.048924626410249882),
             (1.0, 0.35906476682811272, 0.073941248036770935),
             (1.0, 0.39821617549271426, 0.10238769000521743),
             (1.0, 0.4360183680088548, 0.13394028565488458),
             (1.0, 0.47245660339526402, 0.16826194367723249),
             (1.0, 0.50753169830209122, 0.2050152984298336),
             (1.0, 0.5412560073222682, 0.24387165579500372), 
             (1.0, 0.57365048363165339, 0.28451687841701045), 
             (1.0, 0.60474250481040481, 0.32665507807683547), 
             (1.0, 0.63456424053112648, 0.37001075161835467), 
             (1.0, 0.66315140501545844, 0.41432981799538082), 
             (1.0, 0.69054228478187918, 0.45937988226342857), 
             (1.0, 0.71677696611897701, 0.50494995820019151), 
             (1.0, 0.7418967105244817, 0.55084981518147946), 
             (1.0, 0.76594344278242488, 0.59690906895412787), 
             (1.0, 0.78895932748829956, 0.64297610388316206), 
             (1.0, 0.81098641724100307, 0.68891689165075853), 
             (1.0, 0.83206636056102534, 0.73461375515332961), 
             (1.0, 0.85224016070787467, 0.77996411439655389),
             (1.0, 0.87154797855255406, 0.82487924215820596), 
             (1.0, 0.89002897392479741, 0.86928305019130192), 
             (1.0, 0.90772118067749352, 0.9131109212095575), 
             (1.0, 0.92466141127303747, 0.95630859746881824), 
             (1.0, 0.94088518711207447, 0.99883113319192596), 
             (0.96094534092662098, 0.91907377281139446, 1.0), 
             (0.92446069513990714, 0.89794599735471492, 1.0), 
             (0.89125126330213333, 0.87841080194416055, 1.0), 
             (0.86092304734309, 0.86030218826396887, 1.0), 
             (0.83314036066763297, 0.84347601654381688, 1.0), 
             (0.80761569727203752, 0.82780655144571069, 1.0), 
             (0.7841016055622867, 0.81318364068172222, 1.0), 
             (0.7623841245965699, 0.79951039535457924, 1.0), 
             (0.74227744754254443, 0.78670127109266808, 1.0), 
             (0.72361955611627693, 0.77468047158410536, 1.0), 
             (0.70626862856439121, 0.76338061314627015, 1.0), 
             (0.69010006790251943, 0.75274160194947881, 1.0), 
             (0.67500403055130476, 0.7427096854877242, 1.0), 
             (0.66088336101644118, 0.73323664761004259, 1.0), 
             (0.64765185786234125, 0.72427912244472714, 1.0), 
             (0.63523281140050492, 0.71579800727173559, 1.0), 
             (0.62355776533258989, 0.7077579581289617, 1.0), 
             (0.61256546385346433, 0.70012695490161314, 1.0), 
             (0.60220095302553356, 0.69287592501235862, 1.0), 
             (0.59241481103010996, 0.68597841673251048, 1.0), 
             (0.58316248652138603, 0.67941031467132262, 1.0), 
             (0.5744037280114469, 0.67314959124760743, 1.0), 
             (0.56610209019678626, 0.66717608896504987, 1.0), 
             (0.55822450555015435, 0.66147132914561524, 1.0), 
             (0.55074091146298432, 0.65601834346101218, 1.0), 
             (0.54362392482518429, 0.65080152516832568, 1.0), 
             (0.53684855724174518, 0.64580649742560914, 1.0), 
             (0.53039196516602016, 0.64101999645427854, 1.0), 
             (0.52423323012225209, 0.63642976764182158, 1.0), 
             (0.51835316492994088, 0.63202447295235453, 1.0), 
             (0.51273414245870808, 0.62779360824305841, 1.0), 
             (0.50735994395646622, 0.62372742927913827, 1.0), 
             (0.50221562442461298, 0.61981688540465774, 1.0), 
             (0.49728739287593332, 0.61605355996659694, 1.0), 
             (0.49256250561614867, 0.6124296167086094, 1.0), 
             (0.48802917094795567, 0.60893775145280982, 1.0),
             (0.48367646391525015, 0.60557114847505367, 1.0), 
             (0.47949424989105721, 0.60232344105410296, 1.0), 
             (0.47547311597131608, 0.59918867573945778, 1.0), 
             (0.47160430927207442, 0.59616127993829449, 1.0), 
             (0.46787968134371116, 0.59323603247001588, 1.0), 
             (0.46429163801548795, 0.59040803677865517, 1.0), 
             (0.46083309406952777, 0.58767269652959431, 1.0), 
             (0.4574974322173479, 0.58502569334857624, 1.0), 
             (0.45427846591604304, 0.5824629664885812, 1.0), 
             (0.45117040561673855, 0.57998069423412335, 1.0), 
             (0.44816782808604499, 0.57757527687366594, 1.0), 
             (0.44526564848314854, 0.57524332108929699, 1.0), 
             (0.44245909491173546, 0.5729816256291197, 1.0), 
             (0.43974368519778928, 0.57078716814209673, 1.0), 
             (0.43711520567229656, 0.56865709306773948, 1.0), 
             (0.43456969176228444, 0.56658870048420751, 1.0), 
             (0.43210341021518811, 0.56457943582825232, 1.0), 
             (0.42971284280036826, 0.56262688040923003, 1.0), 
             (0.42739467134832548, 0.56072874264716055, 1.0), 
             (0.42514576400281917, 0.55888284997175153, 1.0), 
             (0.42296316257410504, 0.55708714132547354, 1.0), 
             (0.42084407089300974, 0.55533966021925263, 1.0), 
             (0.41878584407575564, 0.55363854829426618, 1.0), 
             (0.41678597861852013, 0.55198203934774059, 1.0), 
             (0.4148421032487502, 0.55036845378452148, 1.0), 
             (0.41295197046743948, 0.54879619345979913, 1.0), 
             (0.41111344872295308, 0.54726373688145802, 1.0), 
             (0.40932451516270291, 0.54576963474342222, 1.0), 
             (0.40758324891410813, 0.54431250576388068, 1.0), 
             (0.40588782485078762, 0.54289103280464546, 1.0), 
             (0.4042365078040614, 0.54150395924991257, 1.0), 
             (0.40262764718353328, 0.54015008562461708, 1.0), 
             (0.401059671973793, 0.53882826643426451, 1.0), 
             (0.39953108607722837, 0.53753740720965404, 1.0), 
             (0.39804046397571496, 0.53627646174130261, 1.0), 
             (0.39658644668619786, 0.53504442948965236, 1.0), 
             (0.39516773798752425, 0.53384035315828104, 1.0), 
             (0.39378310089774704, 0.53266331641837317, 1.0)]
    return mpl.colors.ListedColormap(clist)




def make_movie(filecode,fps=24,bitrate=None,resize=100,output='output.avi',cleanup=False):
    """
    Repeat last 10 frames!
    
    bitrate=10 seems to be good for avi
    """
    if os.path.splitext(output)[1]=='.avi':
        if bitrate is None:
            cmd = 'mencoder "mf://%s" -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=mpeg4'%(filecode,fps,output)
        else:
            bitrate*=1000
            cmd = 'mencoder "mf://%s" -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=%d'%(filecode,fps,output,bitrate)
    elif os.path.splitext(output)[1]=='.gif':
        delay = 100./fps
        cmd = 'convert -delay {:.0f} -resize {}% -layers optimizePlus -deconstruct -loop 0 {} {}'.format(delay,resize, filecode,output)
    print('Executing {}'.format(cmd))
    subprocess.call(cmd,shell=True)
    if cleanup:
        print("Cleaning up files...")
        for ff in glob.glob(filecode):
            os.unlink(ff)
        print("Done!")