# PyInstaller manual: https://www.pyinstaller.org/

# Setup the location to the main script file.  This is the 
# script file you would run to start your application.  You can 
# put the full file path here, as well.
$programMainScriptFile = "TestBuild.py"   

# Navigate to your program, and run the following command
pyinstaller $programMainScriptFile

# If you wish to build all supporting binaries, such as .dll files, 
# into a single file, use the following command (uses the -F flag)
pyinstaller $programMainScriptFile -F



# We can also run the python file of PyInstaller without actually installing it.  Note that all dependencies still need installed to run PyInstaller.
$pythonInterpreterPath = "C:\Software\Programming\Anaconda3\python.exe"
$pyinstallerScriptPath = ""