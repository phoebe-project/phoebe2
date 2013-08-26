#!/bin/bash

sudo cp phoebe_gui.py /usr/local/bin/phoebe_gui
sudo chmod +x /usr/local/bin/phoebe_gui

desktop_file=/usr/share/applications/phoebe_gui.desktop

sudo echo "[Desktop Entry]" > $desktop_file
sudo echo "Name=PHOEBE 2.0" >> $desktop_file
sudo echo "Comment=PHysics Of Eclipsing BinariEs" >> $desktop_file
sudo echo "Exec=/usr/bin/phoebe_gui" >> $desktop_file
sudo echo "Terminal=false" >> $desktop_file
sudo echo "Type=Application" >> $desktop_file
sudo echo "Icon=phoebe-gui" >> $desktop_file
sudo echo "Categories=Education;Science;Astronomy" >> $desktop_file

