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
        ti = self.bundle._get_by_search(twig, section='axes', class_name='Axes', return_trunk_item=True, ignore_errors=True)
        if ti is not None:
            return ti['twig']
        else:
            return None
    
    def add_axes(self, twig, loc=(1,1,1), sharex='_auto_', sharey='_auto_'):
        full_twig = self._get_full_twig(twig)
        
        #~ for key,append_value in zip(['axesrefs','axeslocs'],[full_twig,loc]):
            #~ arr = self.get_value(key)
            #~ arr.append(append_value)
            #~ self.set_value(arr)
        self['axesrefs'] += [full_twig]
        self['axeslocs'] += [loc]
        self['axessharex'] += [sharex if sharex=='_auto_' else self._get_full_twig(sharex)]
        self['axessharey'] += [sharey if sharey=='_auto_' else self._get_full_twig(sharey)]
        
    def remove_axes(self, twig):
        full_twig = self._get_full_twig(twig)
        #~ print "*** remove_axes", twig, full_twig
        
        if full_twig in self['axesrefs']:
            i = self['axesrefs'].index(full_twig)
            #~ print "*** remove_axes FOUND", i
            
            for k in ['axesrefs','axeslocs','axessharex','axessharey']:
                self[k] = [self[k][j] for j in range(len(self[k])) if j!=i]
            
        else:
            #~ print "*** remove_axes NOT FOUND"
            # reset any axessharex or axessharey that match
            for k in ['axessharex','axessharey']:
                new = ['_auto_' if c==full_twig else c for c in self[k]]
                self[k] = new
                
    def _get_for_axes(self, twig, key):
        full_twig = self._get_full_twig(twig)
        
        if full_twig in self['axesrefs']:
            ind = self['axesrefs'].index(full_twig)
            return self[key][ind]
        else: 
            return None

    def get_loc(self, twig):
        return self._get_for_axes(twig, 'axeslocs')
        
    def get_sharex(self, twig):
        return self._get_for_axes(twig, 'axessharex')
        
    def get_sharey(self, twig):
        return self._get_for_axes(twig, 'axessharey')
       
class Axes(parameters.ParameterSet):
    def __init__(self, bundle, **kwargs):
        kwargs['context'] = 'plotting:axes'
        super(Axes, self).__init__(**kwargs)
        
        self.bundle = bundle
        #~ self.plots = []

    def _get_full_twig(self, twig):
        ti = self.bundle._get_by_search(twig, context='plotting:plot_*', kind='ParameterSet', return_trunk_item=True, ignore_errors=True)
        if ti is not None:
            return ti['twig']
        else:
            return None
        
    def add_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        
        self['plotrefs'] += [full_twig]
        
    def remove_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        if full_twig in self['plotrefs']:
            i = self['plotrefs'].index(full_twig)
            
            for k in ['plotrefs']:
                self[k] = [self[k][j] for j in range(len(self[k])) if j!=i]    
