"""
Automatically generate full source documentation and tutorial web pages.

Optionally, also copy the generated pages to the dedicated website.
"""
import os
import re
import shutil
import glob
import sys
import subprocess
import time
import datetime
from phoebe.parameters import definitions
from phoebe.parameters import parameters


def python_to_sphinx(pythonfile,latex=False,type='testsuite.'):
    """
    Convert a Python script (*.py) to a Sphinx file (*.rst).
    """
    myname = type+os.path.splitext(os.path.basename(pythonfile))[0]
    
    ff = open(pythonfile,'r')
    tt = open('phoebe-doc/{}.rst'.format(myname),'w')

    start_doc = False
    inside_code = False
    
    tt.write(":download:`Download this page as a python script <../{}>`\n\n".format(pythonfile))
    
    for line in ff.readlines():
        if 'time.time(' in line: continue
        if 'import time' in line: continue
        if 'os.system' in line: continue
        if 'subprocess.call' in line: continue
        if '***time***' in line: line = line.replace('***time***','{}'.format(str(datetime.datetime.today())))
        if latex and '.gif' in line:
            line = line.replace('.gif','.png')
        if latex and '.svg' in line:
            line = line.replace('.svg','.png')
            
        if not start_doc and line[:3]=='"""':
            start_doc = True
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if start_doc and line[:3]=='"""':
            start_doc = False
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if start_doc:
            tt.write(line)
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if line[0]=='#':
            if inside_code:
                tt.write('\n')
                inside_code = False
            tt.write(line[1:].strip()+'\n')
            continue
        
        if not inside_code and not line.strip():
            continue
        
        if not inside_code:
            inside_code = True
            tt.write('\n::\n\n')
        
        tt.write('    '+line)
    ff.close()
    tt.close()

def generate_parameterlist_sphinx():
    """
    Generate a list of parameters suitable for inclusion in sphinx.
    """
    
    parsets_as_text = []
    pars_as_text = []
    toc_phoebe = []
    toc_wd = []
    
    
    with open('phoebe-doc/parlist.rst','w') as ff:
        
        #-- keep track of all frames and contexts.
        frames = {}
        for par in definitions.defs:
            for frame in par['frame']:
                if not frame in ['phoebe','wd']: continue
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

        frames_contexts = []
        for frame in sorted(frames.keys()):
            for context in sorted(frames[frame]):
                if frame+context in frames_contexts: continue
                frames_contexts.append(frame+context)
                parset = parameters.ParameterSet(frame=frame,context=context)
                if 'label' in parset:
                    parset['label'] = 'mylbl'
                if 'ref' in parset:
                    parset['ref'] = 'myref'
                if 'c1label' in parset:
                    parset['c1label'] = 'primlbl'
                if 'c2label' in parset:
                    parset['c2label'] = 'secnlbl'
                
                
                str_parset = str(parset).split('\n')
                if len(str_parset)<2: continue
                #-- add a label
                parsets_as_text.append('.. _parlabel-{}-{}:\n'.format(frame,context))
                #-- add the parameterset
                parsets_as_text.append('**{}** ({})::\n'.format(context,frame))
                str_parset = '    '+'\n    '.join(str_parset)                
                parsets_as_text.append(str_parset + '\n') 
                
                if frame=='wd':
                    toc_wd.append('- :ref:`{} <parlabel-{}-{}>`'.format(context,frame,context))
                elif frame=='phoebe':
                    toc_phoebe.append(':ref:`{} <parlabel-{}-{}>`'.format(context,frame,context))
                
                links = []
                for qual in parset:
                    try:
                        pars_as_text.append('.. _label-{}-{}-{}:\n\n::\n\n'.format(qual,context,frame)+"    "+"\n    ".join(str(parset.get_parameter(qual)).split('\n'))+'\n\n')
                        links.append(':ref:`{} <label-{}-{}-{}>`'.format(qual,qual,context,frame))
                    except:
                        print("Failed {}".format(qual))
                        raise
                        
                    
                parsets_as_text.append(", ".join(links) + '\n')    
        
        ff.write("""

.. _list-of-parameters:        
        
List of ParameterSets and Parameters
=============================================

""")

        
        ff.write('**Phoebe 2.0 (phoebe) frame**\n\n')
        
        ncol = 3
        cw = 100
        Nt = len(toc_phoebe)
        N = Nt/ncol
        
        ff.write('+'+ ('-'*cw+'+')*ncol + '\n')
        for i in range(N):
            col1, col2, col3 = '','',''
            col1 = toc_phoebe[i]
            if (i+1*N)<Nt: col2 = toc_phoebe[i+1*N]
            if (i+2*N)<Nt: col3 = toc_phoebe[i+2*N]
            
            ff.write('| {:98s} | {:98s} | {:98s} |\n'.format(col1,col2,col3))
            ff.write('+'+ ('-'*cw+'+')*ncol + '\n')
        
        ff.write('\n\n')
        
        ff.write('**Wilson-Devinney (wd) frame**\n\n')
        ff.write('\n'.join(toc_wd))
        
        ff.write('\n\n')
        
        ff.write("""
ParameterSets
-------------

""")
        

        
        ff.write('\n'.join(parsets_as_text))
        
        ff.write("""

Parameters
-------------

""")
        ff.write('\n\n'.join(pars_as_text))
        



#{ For bibliography

def collect_refs_from_rst():
    re_ref = re.compile('\[(.*?)\]_')
    all_refs = []
    for root,dirs,files in os.walk('phoebe'):
        files = [os.path.join(root,ff) for ff in files if os.path.splitext(ff)[1]=='.py']
        if not files: continue
        for rstfile in files:
            with open(rstfile,'r') as ff:
                for line in ff.readlines():
                    all_refs += re.findall(re_ref,line)
    for root,dirs,files in os.walk('phoebe-testsuite'):
        files = [os.path.join(root,ff) for ff in files if os.path.splitext(ff)[1]=='.py']
        if not files: continue
        for rstfile in files:
            with open(rstfile,'r') as ff:
                for line in ff.readlines():
                    all_refs += re.findall(re_ref,line)
    return sorted(set(all_refs))

def make_html_friendly(text):
    remove = ['{','}']
    insert_space = ['~']
    for char in remove:
        text = text.replace(char,'')
    if not 'http' in text:
        for char in insert_space:
            text = text.replace(char,' ')
    return text
    



def write_bib(refs):
    
    names = dict()
    names[r'{\mnras}'] = 'MNRAS'
    names[r'{\aap}'] = 'A&A'
    names[r'{\aaps}'] = 'A&A Supplement Series'
    names[r'{\apjl}'] = 'ApJ Letters'
    names[r'{\apj}'] = 'ApJ'
    names[r'{Communications in Asteroseismology}'] = 'Communications in Asteroseismology'
    names[r'{Astronomische Nachrichten}'] = 'Astronomische Nachrichten'
    names[r'{\nat}'] = 'Nature'
    names[r'{\apss}'] = 'APSS'
    names[r'{ArXiv e-prints}'] = 'ArXiv'
    names[r'{\pasp}'] = 'PASP'
    names[r'{\zap}'] = 'Zeitschrift fur Astrophysik'
    names[r'{\aj}'] = 'Astrophysical Journal'
    names[r'{The Observatory}'] = 'The Observatory'


    
    __refs = [ref.lower() for ref in refs]
    
    with open('/home/pieterd/articles/templates/complete.bib','r') as bib:
        whole_bib = bib.readlines()
        
    with open('phoebe-doc/bibliography.rst','w') as ff:
        
        ff.write(".. _bibliography:\n\n")
        ff.write("Bibliography\n")
        ff.write("============\n\n")
        
        linenr = 0
        
        while linenr<len(whole_bib):
            line = whole_bib[linenr]
            if not line:
                linenr += 1
                continue
            if line[0]=='@':
                thisref = line.split('{')[1].split(',')[0].strip().lower()
                if thisref in __refs:
                    info = dict(refname=refs[__refs.index(thisref)],
                                author='NA',year='NA',journal='NA',
                                title='NA',adsurl='NA')
                    __refs.remove(thisref)
                    refs.remove(info['refname'])
                    linenr += 1
                    if linenr>=len(whole_bib): break
                    while whole_bib[linenr][0]!='}' and linenr<len(whole_bib):
                        thisline = whole_bib[linenr].strip()
                        if not '=' in thisline:
                            linenr += 1
                            continue
                        key,value = thisline.split('=')
                        key = key.strip()
                        value = value.strip().rstrip(',')
                        if key=='author':
                            info['author'] = make_html_friendly(value)
                        elif key=='year':
                            info['year'] = value
                        elif key=='journal':
                            info['journal'] = names[value]
                        elif key=='title':
                            info['title'] = make_html_friendly(value)
                        elif key=='adsurl':
                            info['adsurl'] = make_html_friendly(value)
                            
                        linenr += 1
                    ff.write('.. [{refname}] {author}, {year}, {journal}, {title} (`ADS <{adsurl}>`_)\n'.format(**info))
            linenr += 1
    return __refs

def make_bib():
    refs = collect_refs_from_rst()
    nrefs = write_bib(refs)
    print("Bibliography")
    print("============") 
    print("Did not find refs")
    print(nrefs)

#}








if __name__=="__main__":
    
    make_bib()
    generate_parameterlist_sphinx()
    giffiles = sorted(glob.glob('phoebe-doc/images_tut/*.gif'))
    for giffile in giffiles:
        subprocess.call('convert {}[0] {}'.format(giffile,os.path.splitext(giffile)[0]+'.png'),shell=True)
    
    test_suite = ['phoebe-testsuite/solar_calibration/solar_calibration.py',
                  'phoebe-testsuite/vega/vega.py',
                  'phoebe-testsuite/vega/vega_sed.py',
                  'phoebe-testsuite/sirius/sirius.py',
                  'phoebe-testsuite/wilson_devinney/wd_vs_phoebe.py',
                  'phoebe-testsuite/wilson_devinney/eccentric_orbit.py',
                  'phoebe-testsuite/wilson_devinney/reflection_effect.py',
                  'phoebe-testsuite/wilson_devinney/body_emul.py',
                  'phoebe-testsuite/venus/venus.py',
                  'phoebe-testsuite/differential_rotation/differential_rotation.py',
                  'phoebe-testsuite/fast_rotator/fast_rotator.py',
                  'phoebe-testsuite/critical_rotator/critical_rotator.py',
                  'phoebe-testsuite/spotted_star/spotted_star.py',
                  'phoebe-testsuite/pulsating_star/pulsating_star.py',
                  'phoebe-testsuite/pulsating_binary/pulsating_binary.py',
                  'phoebe-testsuite/pulsating_binary/pulsating_binary2.py',
                  'phoebe-testsuite/pulsating_rotating/pulsating_rotating.py',
                  'phoebe-testsuite/oblique_magnetic_dipole/oblique.py',
                  'phoebe-testsuite/traditional_approximation/traditional_approximation.py',
                  'phoebe-testsuite/beaming/KPD1946+4340.py',
                  'phoebe-testsuite/example_systems/example_systems.py',
                  'phoebe-testsuite/contact_binary/contact_binary.py',
                  'phoebe-testsuite/occulting_dark_sphere/occulting_dark_sphere.py',
                  'phoebe-testsuite/occulting_dark_sphere/transit_colors.py',
                  'phoebe-testsuite/misaligned_binary/misaligned_binary.py',
                  'phoebe-testsuite/accretion_disk/accretion_disk.py',
                  'phoebe-testsuite/accretion_disk/T_CrB.py']
    
    for pythonfile in test_suite:
        python_to_sphinx(pythonfile,type='testsuite.',latex=False)
    python_to_sphinx('phoebe-doc/scripts/how_to_binary.py',type='',latex=False)
    python_to_sphinx('phoebe-doc/scripts/how_to_bundle.py',type='',latex=False)
    
    subprocess.call('sphinx-apidoc -f -o phoebe-doc phoebe',shell=True)
    os.chdir('phoebe-doc')
    subprocess.call('make html',shell=True)
    
    if 'pdf' in sys.argv[1:]:
        os.chdir('..')
        for pythonfile in test_suite:
            python_to_sphinx(pythonfile,type='testsuite.',latex=True)
        os.chdir('phoebe-doc')
        subprocess.call('make latexpdf',shell=True)
        shutil.copy('_build/latex/phoebe.pdf','_build/html/phoebe.pdf')
    
    if 'copy' in sys.argv[1:]:
         #subprocess.call('scp -r _build/html/* copernicus.ster.kuleuven.be:public_html/phoebe_alt',shell=True)
         subprocess.call('scp -r _build/html/* clusty.ast.villanova.edu:/srv/www/phoebe/2.0/docs/',shell=True)
    elif 'copyhtml' in sys.argv[1:]:
         #subprocess.call('scp -r _build/html/* copernicus.ster.kuleuven.be:public_html/phoebe_alt',shell=True)
         subprocess.call('scp -r _build/html/*.html _build/html/_modules clusty.ast.villanova.edu:/srv/www/phoebe/2.0/docs/',shell=True)