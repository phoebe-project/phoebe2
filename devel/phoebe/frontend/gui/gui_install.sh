#!/bin/bash

sudo cp phoebe_gui.py /usr/bin/phoebe_gui
sudo chmod +x /usr/bin/phoebe_gui

desktop_file=/usr/share/applications/phoebe_gui.desktop

echo "[Desktop Entry]" > $desktop_file
echo "Name=PHOEBE 2.0" >> $desktop_file
echo "Comment=PHysics Of Eclipsing BinariEs" >> $desktop_file
echo "Exec=/usr/bin/phoebe_gui" >> $desktop_file
echo "Terminal=false" >> $desktop_file
echo "Type=Application" >> $desktop_file
echo "Icon=phoebe-gui" >> $desktop_file
echo "Categories=Education;Science;Astronomy" >> $desktop_file

