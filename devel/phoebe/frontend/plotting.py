from phoebe.parameters import parameters
import matplotlib.pyplot as plt

class Figure(parameters.ParameterSet):
    def __init__(self, bundle, **kwargs):
        kwargs['context'] = 'plotting:figure'
        super(Figure, self).__init__(**kwargs)
        
        self.bundle = bundle
        #~ self.axes = []
        #~ self.axes_locations = []
        
    def _get_full_twig(self, twig):
        ti = self.bundle._get_by_search(twig, section='axes', class_name='Axes', return_trunk_item=True)
        return ti['twig']
    
    def add_axes(self, twig, loc=(1,1,1)):
        full_twig = self._get_full_twig(twig)
        
        #~ for key,append_value in zip(['axesrefs','axeslocs'],[full_twig,loc]):
            #~ arr = self.get_value(key)
            #~ arr.append(append_value)
            #~ self.set_value(arr)
        self['axesrefs'] += [full_twig]
        self['axeslocs'] += [loc]
        
    def remove_axes(self, twig):
        full_twig = self._get_full_twig(twig)
        i = self.axes.index(full_twig)
        
        self['axesrefs'].pop(i)
        self['axeslocs'].pop(i)
        
       
class Axes(parameters.ParameterSet):
    def __init__(self, bundle, **kwargs):
        kwargs['context'] = 'plotting:axes'
        super(Axes, self).__init__(**kwargs)
        
        self.bundle = bundle
        #~ self.plots = []

    def _get_full_twig(self, twig):
        ti = self.bundle._get_by_search(twig, context='plotting:plot_*', kind='ParameterSet', return_trunk_item=True)
        return ti['twig']
        
    def add_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        
        self['plotrefs'] += [full_twig]
        
    def remove_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        i = self.plots.index(full_twig)
        
        self['plotrefs'].pop(i)
    
