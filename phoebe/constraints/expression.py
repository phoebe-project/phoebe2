

import re

from phoebe.parameters.twighelpers import _uniqueid_to_uniquetwig, _twig_to_uniqueid



class ConstraintVar(object):

    def __init__(self, bundle, twig):

        self._bundle = bundle
        self._parameter = None

        self._is_param = False
        self._is_constant = False
        self._is_method = False

        self._user_label = twig

        try:
            #~ print "ConstraintVar.__init__ twig: {}".format(twig)
            unique_label = _twig_to_uniqueid(bundle, twig=twig)
        except ValueError:
            # didn't return a unique match - either several or None

            # TODO: check length of results?
            self._is_param = False
            self._unique_label = None


            # TODO: check to see if valid constant, method, etc
            raise ValueError("provided twig does not result in unique match")

        else:
            # we have a unique match to a parameter
            self._is_param = True
            self._unique_label = unique_label


        if self._is_param:
            self._safe_label = ''.join(char for char in self._unique_label if char.isalpha())
            # update_user_label will also call _set_curly_label
            self.update_user_label()

    def _set_curly_label(self):
        """
        sets curly label based on user label
        """
        self._curly_label = '{'+self.user_label+'}'

    def update_user_label(self):
        """
        finds this parameter and gets the least_unique_twig from the bundle

        """
        self._user_label = _uniqueid_to_uniquetwig(self._bundle, self.unique_label)
        self._set_curly_label()

    @property
    def is_param(self):
        """
        is this variable a parameter of the system (vs a constant or method of an object)
        """
        return self._is_param

    #~ @property
    #~ def is_constant(self):
        #~ """
        #~ is this variable a constant (vs a parameter of the system or method of an object)
        #~ """
        #~ return self._is_constant
        #~
    #~ @property
    #~ def is_method(self):
        #~ """
        #~ is this variable a method of an object (vs a constant or a parameter of the system)
        #~ """
        #~ return self._is_method

    def get_parameter(self):
        """
        get the parameter object from the system for this var

        needs to be backend safe (not passing or storing bundle)
        """
        if not self.is_param:
            raise ValueError("this var does not point to a parameter")

        # this is quite expensive, so let's cache the parameter object so we only
        # have to filter on the first time this is called
        if self._parameter is None:
            self._parameter = self._bundle.get_parameter(uniqueid=self.unique_label, check_visible=False, check_default=False)

        return self._parameter

    def get_quantity(self, units=None, t=None):
        """
        """
        if self.is_param:
            # TODO: CAREFUL, this may cause infinite loops if we try to run constraints through get_value
            try:
                return self._bundle.get_quantity(uniqueid=self.unique_label, units=units, t=t, check_visible=False, check_default=False)
            except AttributeError:
                # then not a FloatParameter
                return self._bundle.get_value(uniqueid=self.unique_label, check_visible=False, check_default=False)

        else:
            # TODO: constants and methods
            raise NotImplementedError

    def get_value(self, units=None, t=None):
        """
        get the value (either of the constant or from the parameter) for this var

        needs to be backend safe (not passing or storing bundle)
        """
        if self.is_param:
            # TODO: CAREFUL, this may cause infinite loops if we try to run constraints through get_value
            return self._bundle.get_value(uniqueid=self.unique_label, units=units, t=t, check_visible=False, check_default=False)

        else:
            # TODO: constants and methods
            raise NotImplementedError


    @property
    def unique_label(self):
        """
        unique_label corresponding to parameter.get_unique_label
        call get_parameter(system) to retrieve the parameter itself


        needs to be backend safe (not passing or storing bundle)
        """
        return self._unique_label

    @property
    def user_label(self):
        """
        label as the user provided it

        needs to be backend safe (not passing or storing bundle)
        """
        return self._user_label

    @property
    def curly_label(self):
        """
        label with {} brackets - used for the user's view of the equation

        needs to be backend safe (not passing or storing bundle)
        """
        return self._curly_label

    @property
    def safe_label(self):
        """
        label that is safe to be passed through sympy - with escapes from any mathetmatical symbols

        needs to be backend safe (not passing or storing bundle)
        """
        return self._safe_label
