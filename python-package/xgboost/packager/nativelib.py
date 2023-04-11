from platform import system


def _lib_name() -> str:
    """Return platform dependent shared object name."""
    if system() in ["Linux", "OS400"] or system().upper().endswith("BSD"):
        name = "libxgboost.so"
    elif system() == "Darwin":
        name = "libxgboost.dylib"
    elif system() == "Windows":
        name = "xgboost.dll"
    else:
        raise NotImplementedError(f"System {system()} not supported")
    return name


def locate_libxgboost(toplevel_dir):
    libxgboost = toplevel_dir.parent / "lib" / _lib_name()
    if libxgboost.exists():
        return libxgboost
    raise NotImplementedError(f"Please build native lib first using CMake")
