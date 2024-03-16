import sys
from cx_Freeze import setup, Executable

sys.setdefaultencoding("utf-8")

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["networkx", "pandas", "math", "numpy", "geopandas", "shapely", "libpysal", "esda", "splot", "kneed", "matplotlib", "copy" , "pymoo", "typing", "momepy", "contextily"],
    "excludes": [],
    "include_files": ["logo.jpg"],'include_msvcr': True, 'add_to_path': True
}   
# build_exe_options = {'include_msvcr': True, 'add_to_path': True}


# GUI applications require a different base on Windows (the default is for a    
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"
    
setup(  name = "EYDAP Pipe Replacement Tool",
        version = "0.1",
        description = "EYDAP Pipe Replacement Tool",
        options = {"build_exe": build_exe_options},
        executables = [Executable("main.py", base=base)])
