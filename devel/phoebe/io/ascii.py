"""
Read and write ASCII files.
"""
import gzip
import bz2
import logging
import os
import re
import numpy as np
try:
    import StringIO
except ImportError: # for Python3
    from io import StringIO

logger = logging.getLogger("IO.ASCII")
logger.addHandler(logging.NullHandler())

def write_array(data, filename, **kwargs):
    """
    Save a numpy array to an ASCII file.
    
    Add comments via keyword comments (a list of strings denoting every comment
    line). By default, the comment lines will be preceded by the C{commentchar}.
    If you want to override this behaviour, set C{commentchar=''}.
    
    If you give a record array, you can simply set C{header} to C{True} to write
    the header, instead of specifying a list of strings.
    
    @param header: optional header for column names
    @type header: list of str (or boolean for record arrays)
    @param comments: comment lines
    @type comments: list of str
    @param commentchar: comment character
    @type commentchar: str
    @param sep: separator for the columns and header names
    @type sep: str
    @param axis0: string denoting the orientation of the matrix. If you gave
     a list of columns, set C{axis0='cols'}, otherwise C{axis='rows'} (default).
    @type axis0: str, one of C{cols}, C{rows}.
    @param mode: file mode (a for appending, w for (over)writing...)
    @type mode: char (one of 'a','w'...)
    @param auto_width: automatically determine the width of the columns
    @type auto_width: bool
    @param formats: formats to use to write each column
    @type formats: list of string formatters
    """
    header = kwargs.get('header',None)
    comments = kwargs.get('comments',None)
    commentchar = kwargs.get('commentchar','#')
    sep = kwargs.get('sep',' ')
    axis0 = kwargs.get('axis0','rows')
    mode = kwargs.get('mode','w')
    auto_width = kwargs.get('auto_width',False)
    formats = kwargs.get('formats',None)
    # use '%g' or '%f' or '%e' for writing floats automatically from record arrays with auto width
    use_float = kwargs.get('use_float','%f') 
    
    #-- switch to rows first if a list of columns is given
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    if not 'row' in axis0.lower():
        data = data.T
    
    if formats is None:
        try:
            formats = [('S' in str(data[col].dtype) and '%s' or use_float) for col in data.dtype.names]
        except TypeError:
            formats = [('S' in str(col.dtype) and '%s' or '%s') for col in data.T]
    #-- determine width of columns: also take the header label into account
    col_widths = []
    #-- for record arrays
    if auto_width is True and header==True:
        for fmt,head in zip(formats,data.dtype.names):
            col_widths.append(max([len('%s'%(fmt)%(el)) for el in data[head]]+[len(head)]))
    #-- for normal arrays and specified header
    elif auto_width is True and header is not None:
        for i,head in enumerate(header):
            col_widths.append(max([len('%s'%(formats[i])%(el)) for el in data[:,i]]+[len(head)]))
    #-- for normal arrays without header
    elif auto_width is True and header is not None:
        for i in range(data.shape[1]):
            col_widths.append(max([len('%s'%(formats[i])%(el)) for el in data[:,i]]))
    
    if header is True:
        col_fmts = [str(data.dtype[i]) for i in range(len(data.dtype))]
        header = data.dtype.names
    else:
        col_fmts = None
    
    ff = open(filename,mode)
    if comments is not None:
        ff.write('\n'.join(comments)+'\n')
    
    #-- WRITE HEADER
    #-- when header is desired and automatic width
    if header is not None and col_widths:
        ff.write('#'+sep.join(['%%%s%ss'%(('s' in fmt and '-' or ''),cw)%(head) for fmt,head,cw in zip(formats,header,col_widths)])+'\n')
    #-- when header is desired
    elif header is not None:
        ff.write('#'+sep.join(header)+'\n')
    
    #-- WRITE COLUMN FORMATS
    if col_fmts is not None and col_widths:
        ff.write('#'+sep.join(['%%%s%ss'%(('s' in fmt and '-' or ''),cw)%(colfmt) for fmt,colfmt,cw in zip(formats,col_fmts,col_widths)])+'\n')
    elif col_fmts is not None:
        ff.write('#'+sep.join(['%%%ss'%('s' in fmt and '-' or '')%(colfmt) for fmt,colfmt in zip(formats,col_fmts)])+'\n')
    
    #-- WRITE DATA
    #-- with automatic width
    if col_widths:
        for row in data:
            ff.write(' '+sep.join(['%%%s%s%s'%(('s' in fmt and '-' or ''),cw,fmt[1:])%(col) for col,cw,fmt in zip(row,col_widths,formats)])+'\n')
    #-- without automatic width
    else:
        for row in data:
            ff.write(sep.join(['%s'%(col) for col in row])+'\n')
    ff.close()
    
def read2list(filename,**kwargs):
    """
    Load an ASCII file to list of lists.
    
    The comments and data go to two different lists.
    
    Also opens gzipped files.
    
    @param filename: name of file with the data
    @type filename: string
    @param comments: character(s) denoting comment rules
    @type comments: list of str
    @param delimiter: character seperating entries in a row (default: whitespace)
    @type delimiter: str or None
    @param skipempty: skip empty lines
    @type skipempty: bool
    @param skiprows: skip nr of lines (including comment and empty lines)
    @type skiprows: integer
    @return: list of lists (data rows)
             list of lists (comments lines without commentchar),
    @rtype: (list,list)
    """
    commentchar = kwargs.get('comments',['#'])
    splitchar = kwargs.get('delimiter',None)
    skip_empty = kwargs.get('skipempty',True)
    skip_lines = kwargs.get('skiprows',0)
    exp_fmt = kwargs.get('exp_fmt', None)
    
    if isinstance(commentchar,str):
        commentchar = [commentchar]
    
    extension = os.path.splitext(filename)[1]
    if extension == '.gz':
        ff = gzip.open(filename)
    elif extension == '.bz2':
        ff = bz2.BZ2File(filename)
    else:
        ff = open(filename)
        
    data = []  # data
    comm = []  # comments
    
    line_nr = -1
    while 1:  # might call read several times for a file
        line = ff.readline()
        if not line: break  # end of file
        line_nr += 1
        if line_nr<skip_lines:        
            continue
        
        #-- strip return character from line
        if skip_empty and line.isspace():
            continue # empty line
        
        #-- remove return characters
        line = line.replace('\n','')
        
        #-- replace exponential formats
        if exp_fmt is not None:
            line = line.replace(exp_fmt+'+', 'E+')
            line = line.replace(exp_fmt+'-', 'E-')
        
        #-- when reading a comment line
        if line[0] in commentchar:
            comm.append(line[1:])
            continue # treat next line
        
        #-- when reading data, split the line
        data.append(line.split(splitchar))
    ff.close()
    
    #-- report that the file has been read
    #logger.debug('Data file %s read'%(filename))
    
    #-- and return the contents
    return data,comm    
    
def read2recarray(filename,**kwargs):
    """
    Load ASCII file to a numpy record array.
    
    For a list of extra keyword arguments, see C{<read2list>}.
    
    FI dtypes is None, we have some room to automatically detect the contents
    of the columns. This is not implemented yet.
    
    the keyword 'dtype' should be equal to a list of tuples, e.g.
    
    C{dtype = [('col1','a10'),('col2','>f4'),..]}
    
    @param filename: name of file with the data
    @type filename: string
    @param dtype: dtypes of record array 
    @type dtype: list of tuples
    @param return_comments: flag to return comments (default: False)
    @type return_comments: bool
    @return: data array (, list of comments)
    @rtype: ndarray (, list)
    """
    dtype = kwargs.get('dtype',None)
    return_comments = kwargs.get('return_comments',False)
    splitchar = kwargs.get('delimiter',None)
    
    #-- first read in as a normal array
    data,comm = read2list(filename,**kwargs)
    
    #-- if dtypes is None, we have some room to automatically detect the contents
    #   of the columns. This is not fully implemented yet, and works only
    #   if the second-to-last and last columns of the comments denote the
    #   name and dtype, respectively
    if dtype is None:
        data = np.array(data,dtype=str).T
        header = comm[-2].replace('|',' ').split()
        types = comm[-1].replace('|','').split()
        dtype = [(head,typ) for head,typ in zip(header,types)]
        dtype = np.dtype(dtype)
    elif isinstance(dtype,list):
        data = np.array(data,dtype=str).T
        dtype = np.dtype(dtype)
    #-- if dtype is a list, assume it is a list of fixed width stuff.
    elif isinstance(splitchar,list):
        types,lengths = fws2info(splitchar)
        dtype = []
        names = range(300)
        for i,(fmt,lng) in enumerate(zip(types,lengths)):
            if fmt.__name__=='str':
                dtype.append((str(names[i]),(fmt,lng)))
            else:
                dtype.append((str(names[i]),fmt))
        dtype = np.dtype(dtype)
    
    #-- cast all columns to the specified type
    data = [np.cast[dtype[i]](data[i]) for i in range(len(data))]
        
    #-- and build the record array
    data = np.rec.array(data, dtype=dtype)
    return return_comments and (data,comm) or data

def loadtxt(filename, *args, **kwargs):
    """
    Load a text file using numpy's loadtxt, but fix Fortran float "D" notation.
    """
    with open(filename, 'r') as ff:
        contents = "".join(ff.readlines())
        contents = contents.replace('D','E')
        contents = contents.replace('d','e')
        c = StringIO(contents)
    data = np.loadtxt(c, *args, **kwargs)
    return data
    