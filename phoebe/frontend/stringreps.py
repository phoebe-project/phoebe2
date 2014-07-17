"""
make string representations of a Bundle
"""
from phoebe.backend import universe
import textwrap
from collections import OrderedDict
import numpy as np


def make_body_label(body, level, emphasize):
    
    body_name = type(body).__name__
    if body_name == 'BodyBag':
        arrow = '='
    else:
        arrow = '-'
    
    if level == 1:
        prefix = ''
    else:
        prefix = ('|' + ' '*6) * (level-2) + '+'+arrow*6+'> ' 
        
    label = prefix + emphasize(body.get_label()) + ' ({})'.format(body_name)
    return label

def make_param_label(param, green):
    if param.has_posterior():
        loc = param.get_posterior().get_loc()
        scale = param.get_posterior().get_scale()
        label = "{}={} ({:.2g}+/-{:.2g})".format(param.get_qualifier(),
                                                param.to_str(), loc, scale)
    else:
        label = "{}={}".format(param.get_qualifier(), param.to_str())
    
    if param.has_unit():
        label += ' {}'.format(param.get_unit())
    if param.get_adjust():
        label = green(label)
    return label


def to_str(x, summary_type='full', emphasize=True, width=79):
    """
    Make a string representation of a Bundle.
    
    
    """
    # Define how to print out emphasis, italic, strikethrough and colors
    if emphasize:
        emphasize = lambda x: '\033[1m\033[4m' + x + '\033[m'
        italicize = lambda x:  '\x1B[3m' + x + '\033[m'
        strikethrough = lambda x: '\x1b[9m' + x + '\x1b[29m'
        green = lambda x: "\033[32m" + x + '\033[m'
    else:
        emphasize = lambda x: x
        italicize = lambda x: x
        strikethrough = lambda x: x
        green = lambda x: x

    # First collect all levels
    current_pset = None
    current_body = None
    current_string = []
    
    # Get the label of the total system, this is ignored when iterating over it
    system_label = x.get_system().get_label()
    total_string = OrderedDict()
    current_dict = total_string
    total_obsdep = OrderedDict()
    current_obsdep = total_obsdep
    level = None

    # Make sure to not print out all array variables (we'll reset it on function
    # leave)
    old_threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=8)
    
    correct_level = ['system', 'Bundle']
    
    for item in x.trunk:       
        
        last_two = item['twig_full'].split('@')[-2:]
        if not len(last_two) == len(correct_level):
            continue
        
        # Because the datasets are added in a new level
        if not last_two == correct_level and not (last_two[-2]=='dataset' and correct_level[-2]=='system'):
            continue
        
        if item['hidden'] is True:
            continue
        
        path = item['path']
        
        if path is None:
            path = [x.get_system()]
        else:
            path = [x.get_system()] + path
        
        # Get the item
        twig = item['twig_full']
        it = item['item']
        
        twig_split = twig.split('@')
        
        if 'bandpass' in twig:
            print twig
            print current_pset
            print it
            print corrent_pset.contains(it)
            print item['kind']
            
        # If we have a new ParameterSet, we need to string-rep the old one
        if item['kind'] == 'Parameter' and (current_pset is None or not current_pset.contains(it)):
            
            if not current_string and current_pset is not None and summary_type=='only_adjust':
                current_string = ['']
            
            if current_string:        
                
                context = current_pset.get_context()
                
                if context in ['orbit']:
                    level = level - 1
                    
                # Some contexts appear more than once, we need to make them unique
                if context in ['puls', 'circ_spot']:
                    context = '{}@{}'.format(current_pset['label'], context)
                
                current_label = make_body_label(bodies[level-1], level, emphasize)
                
                # pbdeps, obs and syn deserve special treatment
                if context[-3:] in ['obs', 'dep', 'syn'] and summary_type != 'cursory':
                    # if we want "lcdep[0]" as a string
                    seqnr = len([key for key in total_string[current_label].keys() if key[:len(context)]==context])
                    #context = context + '[{}]'.format(seqnr)
                    # if we want [ref@lcdep@body] as a string
                    #last_body = [b for b in path if isinstance(b, phoebe.backend.universe.Body)][-1]
                    #current_body_label = last_body.get_label()
                    context = '{} ({}@{})'.format(context, current_pset['ref'], context)
                        
                
                
                
                if not current_label in total_string:
                    total_string[current_label] = OrderedDict()
                    total_obsdep[current_label] = OrderedDict()
                
                if level == 1 or (hasattr(bodies[level-1], 'bodies') and len(bodies[level-1].bodies)):
                    include_last_pipe = True
                else:
                    include_last_pipe = False
                
                indent = ('|' + ' '*6) * (level-1) + (('|' + ' '*3) if include_last_pipe else '    ')
                sub_indent = indent+5*' '
                
                if summary_type in ['full', 'only_adjust']:
                    total_string[current_label][context] = textwrap.fill(context+': '+ ", ".join(current_string), initial_indent=indent, subsequent_indent=sub_indent, width=width)
                elif summary_type == 'cursory' and not (context[-3:] in ['obs', 'dep', 'syn']):
                    total_string[current_label][context] = textwrap.fill(context.split(':')[0], initial_indent=indent, subsequent_indent=sub_indent, width=width)
                elif summary_type == 'cursory':
                    context = italicize(context)
                    if not context in total_obsdep[current_label]:
                        total_obsdep[current_label][context] = []
                    this_ref = current_pset['ref']
                    if not current_pset.get_enabled():
                        this_ref = strikethrough(this_ref)
                    total_obsdep[current_label][context].append(this_ref)
            
            # On to the next parameterSet!
            current_pset = item['path'][-2]
            
            # Here we have the obs or dep or syn
            if isinstance(current_pset, str):
                ptype, ref = item['path'][-3:-1]
                if isinstance(ptype, str):
                    current_pset = item['path'][-4][ptype][ref]
                else:
                    current_pset = None
                current_string = []
                
                if current_pset is None:
                    continue
            
            current_string = []
            
            
        if item['kind'] == 'Parameter':
            if summary_type == 'full' and not it.get_hidden():
                current_string.append(make_param_label(it, green))
            elif summary_type == 'only_adjust' and it.get_adjust():
                current_string.append(make_param_label(it, green))
            elif summary_type == 'only_adjust':
                pass
            else:
                current_string.append('')
            
            # Get the number of Bodies in the path
            bodies = [body for body in path if isinstance(body, universe.Body)]
            level = len(bodies)
        
        
        # Make sure that the Bodies appear in the right order
        if item['kind'] == 'Body':
            this_bodies = [body for body in path if isinstance(body, universe.Body)]
            this_level = len(this_bodies)
            this_label = make_body_label(this_bodies[this_level-1], this_level, emphasize)
                
            if not this_label in total_string:
                total_string[this_label] = OrderedDict()
                total_obsdep[this_label] = OrderedDict()
    
    # Take care of final thing
    else:
        if current_string:        
            
            context = current_pset.get_context()
            if context in ['orbit']:
                level = level - 1
            
            # Some contexts appear more than once, we need to make them unique
            if context in ['puls', 'circ_spot']:
                context = '{}@{}'.format(current_pset['label'], context)
            
            if context[-3:] in ['obs', 'dep', 'syn'] and summary_type != 'cursory':
                # if we want "lcdep[0]" as a string
                    seqnr = len([key for key in total_string[current_label].keys() if key[:len(context)]==context])
                    #context = context + '[{}]'.format(seqnr)
                    # if we want [ref@lcdep@body] as a string
                    #last_body = [b for b in path if isinstance(b, phoebe.backend.universe.Body)][-1]
                    #current_body_label = last_body.get_label()
                    context = '{} ({}@{})'.format(context, current_pset['ref'], context)
                    
            
            current_label = make_body_label(bodies[level-1], level, emphasize)
            
            if not current_label in total_string:
                total_string[current_label] = OrderedDict()
                total_obsdep[current_label] = OrderedDict()
            
            if level == 1 or (hasattr(bodies[level-1], 'bodies') and len(bodies[level-1].bodies)):
                include_last_pipe = True
            else:
                include_last_pipe = False
            
            indent = ('|' + ' '*6) * (level-1) + (('|' + ' '*3) if include_last_pipe else '    ')
            sub_indent = indent+5*' '
            
            if summary_type in ['full', 'only_adjust']:
                total_string[current_label][context] = textwrap.fill(context+': '+ ", ".join(current_string), initial_indent=indent, subsequent_indent=sub_indent, width=79)
            elif summary_type == 'cursory' and not (context[-3:] in ['obs', 'dep', 'syn']):
                total_string[current_label][context] = textwrap.fill(context.split(':')[0], initial_indent=indent, subsequent_indent=sub_indent, width=79)
            elif summary_type == 'cursory':
                context = italicize(context)
                if not context in total_obsdep[current_label]:
                    total_obsdep[current_label][context] = []
                this_ref = current_pset['ref']
                if not current_pset.get_enabled():
                    this_ref = strikethrough(this_ref)
                total_obsdep[current_label][context].append(this_ref)    
    
    # Default printoption
    np.set_printoptions(threshold=old_threshold) 
    
    output_string = []
    
    if summary_type == 'cursory':
        output_string.append('* The system:')
    
    previous_level = -1
    for body in total_string:
        output_string.append(body)
        this_level = body.count('|   ') + body.count('+---') + body.count('+===')
        
        # First print all things that are not obs/deps etc
        for context in total_string[body]:
            if not context.split('[')[0][-3:] in ['dep','obs','syn']:
                output_string.append(total_string[body][context])
        
        # Then print all obs/deps
        for context in total_string[body]:
            if context.split('[')[0][-3:] in ['dep','obs','syn']:
                output_string.append(total_string[body][context])
            
        # For the cursory, all obs/deps are represented in one line:
        for context in total_obsdep[body]:
            indent = ('|' + ' '*6) * (this_level) + (' ' + ' '*3)
            if not indent[0] == '|':
                indent = '|' + indent[1:]
            sub_indent = indent+5*' '
            output_string.append(textwrap.fill(context+': '+ ", ".join(total_obsdep[body][context]), initial_indent=indent, subsequent_indent=sub_indent, width=79))
        
        # Last line gives some vertical spacing between the components, but
        # the pipe thingies need to be informative still
        last_line = ''
        try:
            last_line = '|'.join(total_string[body].values()[-1].split('\n')[-1].split('|')[:-1])
            if previous_level < this_level:
                last_line += '|'
            output_string.append(last_line)
        # The above fails for the structured view
        except IndexError:
            insert_line = output_string[-1].split('+')[0]
            output_string.insert(-1,insert_line + '|')
        
        previous_level = this_level
    
    # Weird fix
    if summary_type == 'structure':
        output_string = output_string[1:]
    
    # Add compute stuff for cursory summary:
    computes = x._get_dict_of_section('compute')
    if summary_type in ['cursory', 'only_adjust']:
        output_string.append('* Compute: ' + ", ".join(computes.keys()) + '\n')
    
    # Add compute stuff for full summary
    elif summary_type in 'full':
        output_string_ = ['* Compute:']
        for icomp in computes.values():
            mystring = []
            for par in icomp:
                mystring.append("{}={}".format(par,icomp.get_parameter(par).to_str()))
                if icomp.get_parameter(par).has_unit():
                    mystring[-1] += ' {}'.format(icomp.get_parameter(par).get_unit())
            mystring = ', '.join(mystring)
            output_string_ += textwrap.wrap(mystring, initial_indent='', subsequent_indent=7*' ', width=79)
            
        output_string_.append('* Other')
        if len(x._get_dict_of_section("fitting")):
            output_string_.append("{} fitting options".format(len(x._get_dict_of_section("fitting"))))
        
        for sect in ['figure', 'axes', 'plot']:
            if len(x._get_dict_of_section(sect)):
                output_string_.append("{} {}".format(len(x._get_dict_of_section("axes")), sect))
        
        output_string = output_string_ + output_string
    
    
    return ("\n".join(output_string))


if __name__== "__main__":
    from phoebe.frontend import bundle
    x = bundle.Bundle()
    y = bundle.Bundle()
    y['label@primary'] = 'prim'
    y['label@secondary'] = 'sec'
    y.get_system().params.pop('position')
    x['primary'] = y['new_system']
    #x.add_parameter('radius@primary', replaces='pot', value=1.41)
    print(to_str(x))
