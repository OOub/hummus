# Baal: clock-based spiking neural network simulator


## Dependencies


### Homebrew is used to install the dependencies on macOS:


Run \*[/usr/bin/ruby -e \"\$(curl -fsSL
https://raw.githubusercontent.com/Homebrew/install/master install)\"\*
to install homebrew]{.s1}

\

\#\#\# Premake4 is used to build this project

\

\*\*On macOS:\*\*run \*brew install premake\*

\*\*On Linux Debian:\*\* run \*sudo apt-get install premake4\*

\

\

\#\#\# If the Display Class is being used:

\

-   Install Qt 5.7+ and make sure the Qt Charts add-on is installed

\

\*\*On macOS:\*\*

\

\*\*option 1\*\*

run \*brew install qt5\*

\

\*\*option 2\*\*

download directly from: \*https://www.qt.io/download/\*

\

\*\*On Linux Debian:\*\*

\

\*\*option 1\*\*

1\. download directly from: \*https://www.qt.io/download/ \*

\

2\. permanently add the Qt dynamic lib path to the LD\_LIBRARY\_PATH by
opening the .bashrc file in your home directory and adding at the end
the lines:[ ]{.Apple-converted-space}

\

[\*LD\_LIBRARY\_PATH=\[path the Qt dynamic lib path\]\*]{.s1}

[\*export LD\_LIBRARY\_PATH\*]{.s1}

\

3\. modify the include and library paths of Qt in the premake4 file

\

\*\*option 2\*\*

\

1\. run \*sudo apt-get install qtbase5-dev qtdeclarative5-dev\*

\

2\. follow instructions from this website to build the Qt Charts add-on
from source: \*https://github.com/qt/qtcharts\*

\

3\. make sure QT Charts libraries are in the same folder as the rest of
Qt libraries

\

\#\# Installation

\

\*\*Installation:\*\* \*premake4 install\*

\

\*\*Uninstallation:\*\* \*premake4 uninstall\*

\

\#\# Testing

\

-   1\. Go to the baal directory and run \*premake4 gmake && cd build &&
    make\*
-   [2. Run the executable release/testNetwork ]{.s2}(you need to be in
    the level of the executable to correctly run the display)
