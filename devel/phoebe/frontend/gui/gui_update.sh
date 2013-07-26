#!/bin/bash

# creates a ui_phoebe_pyqt4.py and ui_phoebe_pyside.py file using all files ./*.ui
# pyside-uic requires pyside-tools

echo "# Generated using gui_update.sh and pyuic4 from reading files" > ui_phoebe_pyqt4.py
ls ./*.ui | awk '{print "#", $0}' >> ui_phoebe_pyqt4.py
echo "# WARNING! All changes made in this file will be lost!" >> ui_phoebe_pyqt4.py
echo >> ui_phoebe_pyqt4.py
echo "from PyQt4 import QtCore, QtGui" >> ui_phoebe_pyqt4.py
echo "try:" >> ui_phoebe_pyqt4.py
echo "    _fromUtf8 = QtCore.QString.fromUtf8" >> ui_phoebe_pyqt4.py
echo "except AttributeError:" >> ui_phoebe_pyqt4.py
echo "    _fromUtf8 = lambda s: s" >> ui_phoebe_pyqt4.py
for file in ./*.ui
do
pyuic4 $file | tail -n +16 | grep -v "import.*_rc" >> ui_phoebe_pyqt4.py
done
for file in ./*.qrc
do
pyrcc4 $file | tail -n +11 >> ui_phoebe_pyqt4.py
done
