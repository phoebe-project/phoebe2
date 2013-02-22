"""
Automatically generate full source documentation and tutorial web pages.

Optionally, also copy the generated pages to the dedicated website.
"""
import os
import shutil
import glob
import sys
import subprocess
import time
import datetime
from pyphoebe.parameters import definitions
from pyphoebe.parameters import parameters

def make_doc(output='html'):
    """
    Generate documentation files.
    """
    #-- list files and directories that need to be included
    include = ['phoebe/parameters/','phoebe/atmospheres/','phoebe/utils/',
               'phoebe/algorithms/','phoebe/dynamics/','phoebe/backend/',
               'phoebe/units','phoebe/wd/']
    #-- execute the Epydoc command           
    os.system('epydoc --%s %s -o phoebe-doc --parse-only --graph all -v --exclude=.*uncertainties.*'%(output,' '.join(include)))
    
    #-- possibility to output html or pdf: in either case, we need to replace
    #   the image code with the image filenames.
    direc = os.path.abspath('phoebe-doc')
    #-- when it's HTML, insert image HTML code
    if output.lower()=='html':
        files = sorted(glob.glob('phoebe-doc/*module.html'))
        files+= sorted(glob.glob('phoebe-doc/*class.html'))
        image_code = r"<img src='{0}' alt='[image example]' width=75%/>"
    #-- when it's LaTeX, insert TeX code.
    elif output.lower()=='pdf':
        files = sorted(glob.glob('phoebe-doc/*.tex'))
        image_code = r"\begin{{center}}\includegraphics[width=0.75\textwidth]{{{0}}}\end{{center}}"
        shutil.move('phoebe-doc/api.tex','phoebe-doc/api.tex_')
        ff = open('phoebe-doc/api.tex_','r')
        oo = open('phoebe-doc/api.tex','w')
        for line in ff.readlines():
            if 'usepackage' in line:
                oo.write(r'\usepackage{graphicx}'+'\n')
                break
            oo.write(line)
        ff.close()
        oo.close()
    
    #-- run over all files, and replace the ]include figure] code with the
    #   image code
    for myfile in files:
        shutil.move(myfile,myfile+'_')
        ff = open(myfile+'_','r')
        oo = open(myfile,'w')
        
        line_break = False
        for line in ff.readlines():
            
            if ']include figure]' in line or line_break:
                filenames1 = [os.path.join(direc,ifig.split(']')[-2].strip()) for ifig in line.split(';')]
                filenames = [ifig.split(']')[-2].strip() for ifig in line.split(';')]
                oo.write('<p style="white-space;nowrap;">\n\n')
                for filename1,filename in zip(filenames1,filenames):
                    if not os.path.isfile(filename1):
                        print "Skipping image, file %s not found"%(filename)
                        continue
                    oo.write(image_code.format(filename)+'\n\n')
                    print 'Added image %s to %s'%(filename,myfile)
                    oo.write('\n\n')
                line_break = False
                oo.write('</p>\n\n')
            
                
            elif ']include' in line:
                line_break = True
                
                
            else:
                oo.write(line)
        ff.close()
        oo.close()
        os.remove(myfile+'_')
    
    generate_parameterlist()



def generate_parameterlist():
    """
    Replace the documentation page of the definitions file with a list of
    all the ParameterSets and parameters.
    """
    #-- keep track of all frames and contexts.
    frames = {}
    for par in definitions.defs:
        for frame in par['frame']:
            if not frame in ['phoebe','pywd']: continue
            if frame not in frames:
                if isinstance(par['context'],list):
                    frames[frame]+= par['context']
                else:
                    frames[frame] = [par['context']]
            elif not par['context'] in frames[frame]:
                if isinstance(par['context'],list):
                    frames[frame]+= par['context']
                else:
                    frames[frame].append(par['context'])

    #-- ParameterSets
    body = '<h1 class="heading"> ParameterSets</h1>'

    frames_contexts = []
    for frame in sorted(frames.keys()):
        body += '<h2 class="heading">Frame "{frame}"</h2>'.format(frame=frame)
        for context in sorted(frames[frame]):
            if frame+context in frames_contexts: continue
            frames_contexts.append(frame+context)
            parset = parameters.ParameterSet(frame=frame,context=context)
            
            
            body += '<h3 class="heading">Frame "{frame}", context "{context}"</h3>'.format(frame=frame,context=context)
            body += """<pre class="py-doctest">
<span class="py-output">{strrep}</span>
</pre>""".format(strrep=str(parset))

    #-- Parameter
    body += '<h1 class="heading"> Parameters</h1>'

    frames_contexts = []
    for frame in sorted(frames.keys()):
        for context in sorted(frames[frame]):
            if frame+context in frames_contexts: continue
            frames_contexts.append(frame+context)
            parset = parameters.ParameterSet(frame=frame,context=context)
            for par in parset:
                par = parset.get_parameter(par)
                body += """<pre class="py-doctest">
<span class="py-output">{strrep}</span>
</pre>""".format(strrep=str(par))



    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'phoebe-doc')
    with open(os.path.join(basedir,'parameter_list_template.html'),'r') as ff:
        with open(os.path.join(basedir,'phoebe.parameters.definitions-module.html'),'w') as gg:
            gg.write(ff.read().format(body=body))


                
                
if __name__=="__main__":
    options = sys.argv[1:]
    if not options or 'doc' in options[0]:
        make_doc()
        if len(options)>1 and 'copy' in options[1]:
            os.system('scp -r doc/* copernicus.ster.kuleuven.be:public_html/pyphoebe/')