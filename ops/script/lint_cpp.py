import argparse
import os
import re
import sys
from typing import TextIO

import cpplint
from cpplint import _cpplint_state

CXX_SUFFIX = set(["cc", "c", "cpp", "h", "cu", "hpp"])


def filepath_enumerate(paths: list[str]) -> list[str]:
    """Enumerate the file paths of all subfiles of the list of paths"""
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    out.append(os.path.normpath(os.path.join(root, name)))
    return out


def get_header_guard_dmlc(filename: str) -> str:
    """Get Header Guard Convention for DMLC Projects.

    For headers in include, directly use the path
    For headers in src, use project name plus path

    Examples: with project-name = dmlc
        include/dmlc/timer.h -> DMLC_TIMTER_H_
        src/io/libsvm_parser.h -> DMLC_IO_LIBSVM_PARSER_H_
    """
    fileinfo = cpplint.FileInfo(filename)
    file_path_from_root = fileinfo.RepositoryName()
    inc_list = ["include", "api", "wrapper", "contrib"]
    if os.name == "nt":
        inc_list.append("mshadow")

    if file_path_from_root.find("src/") != -1 and _HELPER.project_name is not None:
        idx = file_path_from_root.find("src/")
        file_path_from_root = _HELPER.project_name + file_path_from_root[idx + 3 :]
    else:
        idx = file_path_from_root.find("include/")
        if idx != -1:
            file_path_from_root = file_path_from_root[idx + 8 :]
        for spath in inc_list:
            prefix = spath + "/"
            if file_path_from_root.startswith(prefix):
                file_path_from_root = re.sub("^" + prefix, "", file_path_from_root)
                break
    return re.sub(r"[-./\s]", "_", file_path_from_root).upper() + "_"


class Lint:
    def __init__(self) -> None:
        self.project_name = "xgboost"
        self.cpp_header_map: dict[str, dict[str, int]] = {}
        self.cpp_src_map: dict[str, dict[str, int]] = {}

        self.pylint_cats = set(["error", "warning", "convention", "refactor"])
        # setup cpp lint
        cpplint_args = ["--quiet", "--extensions=" + (",".join(CXX_SUFFIX)), "."]
        _ = cpplint.ParseArguments(cpplint_args)
        cpplint._SetFilters(
            ",".join(
                [
                    "-build/c++11",
                    "-build/include,",
                    "+build/namespaces",
                    "+build/include_what_you_use",
                    "+build/include_order",
                ]
            )
        )
        cpplint._SetCountingStyle("toplevel")
        cpplint._line_length = 100

    def process_cpp(self, path: str, suffix: str) -> None:
        """Process a cpp file."""
        _cpplint_state.ResetErrorCounts()
        cpplint.ProcessFile(str(path), _cpplint_state.verbose_level)
        _cpplint_state.PrintErrorCounts()
        errors = _cpplint_state.errors_by_category.copy()

        if suffix == "h":
            self.cpp_header_map[str(path)] = errors
        else:
            self.cpp_src_map[str(path)] = errors

    @staticmethod
    def _print_summary_map(
        strm: TextIO, result_map: dict[str, dict[str, int]], ftype: str
    ) -> int:
        """Print summary of certain result map."""
        if len(result_map) == 0:
            return 0
        npass = sum(1 for x in result_map.values() if len(x) == 0)
        strm.write(f"====={npass}/{len(result_map)} {ftype} files passed check=====\n")
        for fname, emap in result_map.items():
            if len(emap) == 0:
                continue
            strm.write(
                f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={str(emap)}\n"
            )
        return len(result_map) - npass

    def print_summary(self, strm: TextIO) -> int:
        """Print summary of lint."""
        nerr = 0
        nerr += Lint._print_summary_map(strm, self.cpp_header_map, "cpp-header")
        nerr += Lint._print_summary_map(strm, self.cpp_src_map, "cpp-source")
        if nerr == 0:
            strm.write("All passed!\n")
        else:
            strm.write(f"{nerr} files failed lint\n")
        return nerr


_HELPER = Lint()

cpplint.GetHeaderGuardCPPVariable = get_header_guard_dmlc


def process(fname: str, allow_type: list[str]) -> None:
    """Process a file."""
    fname = str(fname)
    arr = fname.rsplit(".", 1)
    if fname.find("#") != -1 or arr[-1] not in allow_type:
        return
    if arr[-1] in CXX_SUFFIX:
        _HELPER.process_cpp(fname, arr[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="run cpp lint")
    parser.add_argument(
        "path",
        nargs="*",
        help="Path to traverse",
        default=[
            "src",
            "include",
            os.path.join("R-package", "src"),
            "python-package",
            "plugin/sycl",
        ],
    )
    parser.add_argument(
        "--exclude_path",
        nargs="+",
        default=[],
        help="exclude this path, and all subfolders if path is a folder",
    )
    args = parser.parse_args()
    excluded_paths = filepath_enumerate(args.exclude_path)

    allow_type: list[str] = []
    allow_type += CXX_SUFFIX

    for path in args.path:
        if not os.path.exists(path):
            raise ValueError(f"Unknown path: {path}")
        if os.path.isfile(path):
            normpath = os.path.normpath(path)
            if normpath not in excluded_paths:
                process(path, allow_type)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_path = os.path.normpath(os.path.join(root, name))
                    if file_path not in excluded_paths:
                        process(file_path, allow_type)
    nerr = _HELPER.print_summary(sys.stderr)
    sys.exit(nerr > 0)


if __name__ == "__main__":
    main()
