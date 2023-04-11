import shutil


def copy_cpp_src_tree(cpp_src_dir, target_dir):
    """Copy C++ source tree into build directory"""

    for subdir in [
        "src",
        "include",
        "dmlc-core",
        "gputreeshap",
        "rabit",
        "cmake",
        "plugin",
    ]:
        shutil.copytree(cpp_src_dir.joinpath(subdir), target_dir.joinpath(subdir))

    for filename in ["CMakeLists.txt", "LICENSE"]:
        shutil.copy(cpp_src_dir.joinpath(filename), target_dir)
