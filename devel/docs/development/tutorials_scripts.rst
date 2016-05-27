
Tutorials and Example Scripts
****************************************


File locations
==================================

Tutorials belong in docs/tutorials, example scripts belong in docs/examples and 
should be written as IPython notebooks (see the section below for how to create,
edit, and run notebooks).


Adding to Documentation
----------------------------------

Once you've created an IPython notebook in the correct directory, make sure it
passes testing (see below), and then you're ready to add it to the documentation.

The rst file (which is created when you build documentation) needs to be linked
to from index.rst (see how existing tutorials are linked).  Rebuild the documentation
to make sure the link works correctly, then commit changes to index.rst and add
the IPython Notebook and any input files (but not the rst file or any of the created images).  
Once the documentation is rebuilt on the server, the online version will be updated.


IPython Notebooks
==================================

COMING SOON


General Structure
=================================

The first line of the notebook should include links to the IPython Notebook
and automatically generated Python script versions of the tutorial/script.
See existing tutorials for how to format this line or simply copy the template_tutorial.ipynb
to get started.

The second line should be a top-level header (as a 'Markdown' cell) that is the name of the tutorial
or example script.  This name is used to build the table of contents within 
the documentation.  

Tutorials
----------------------------------

Tutorials are generally quite verbose and should build upon each other in sequential
order.  Generally the first header (after the name of the tutorial) should be
'Setup' followed by lines to reproduce steps learned in previous tutorials.


Example Scripts
----------------------------------

Example scripts have less of a strict format but can also include a 'Setup' header
to import and setup the logger.  Try to follow the same conventions for importing
and logger (ie import matplotlib.pyplot as plt instead of some other variation).


Testing
==================================

Tutorials and example scripts are tested whenever documentation is rebuilt
or run_tests.py is run.  The tests make sure that all tutorials successfully 
run and also makes sure that the shown outputs remains unchanged.  Since run_tests.py
will be run before each commit, we should always have tutorials that work and 
provide the same output as shown on the website.

In cases where a tutorial fails the test due to changing output - there could
be several solutions based on the situation.

* Changes to the code make acceptable output changes which are not major.  In
this case the source of the tutorial needs to be updated so that it passes the
test.  Load the ipython notebook (ipython notebook tutorial_name.ipynb) and run
all cells.  Then save the source and retest.  If this fixes the problem, don't 
forget to commit the changes to the tutorial.

* Changes to the code make acceptable output change but which are somewhat major.
In this case, the source of the tutorial needs to be updated as above *but* a note
should also be included mentioning what version/release made the change.

* Changes to the code break the output and are unacceptable.  In this case PHOEBE
needs to be fixed to make the output acceptable and make sure the tests succeed.




