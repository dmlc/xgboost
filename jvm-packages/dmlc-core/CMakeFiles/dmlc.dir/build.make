# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.24.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.24.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/centerco/PycharmProjects/sovrn/xgboost_orig

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages

# Include any dependencies generated for this target.
include dmlc-core/CMakeFiles/dmlc.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.make

# Include the progress variables for this target.
include dmlc-core/CMakeFiles/dmlc.dir/progress.make

# Include the compile flags for this target's objects.
include dmlc-core/CMakeFiles/dmlc.dir/flags.make

dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/config.cc
dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o -MF CMakeFiles/dmlc.dir/src/config.cc.o.d -o CMakeFiles/dmlc.dir/src/config.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/config.cc

dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/config.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/config.cc > CMakeFiles/dmlc.dir/src/config.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/config.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/config.cc -o CMakeFiles/dmlc.dir/src/config.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/data.cc
dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o -MF CMakeFiles/dmlc.dir/src/data.cc.o.d -o CMakeFiles/dmlc.dir/src/data.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/data.cc

dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/data.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/data.cc > CMakeFiles/dmlc.dir/src/data.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/data.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/data.cc -o CMakeFiles/dmlc.dir/src/data.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o -MF CMakeFiles/dmlc.dir/src/io.cc.o.d -o CMakeFiles/dmlc.dir/src/io.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io.cc > CMakeFiles/dmlc.dir/src/io.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io.cc -o CMakeFiles/dmlc.dir/src/io.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/recordio.cc
dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o -MF CMakeFiles/dmlc.dir/src/recordio.cc.o.d -o CMakeFiles/dmlc.dir/src/recordio.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/recordio.cc

dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/recordio.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/recordio.cc > CMakeFiles/dmlc.dir/src/recordio.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/recordio.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/recordio.cc -o CMakeFiles/dmlc.dir/src/recordio.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/line_split.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o -MF CMakeFiles/dmlc.dir/src/io/line_split.cc.o.d -o CMakeFiles/dmlc.dir/src/io/line_split.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/line_split.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/line_split.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/line_split.cc > CMakeFiles/dmlc.dir/src/io/line_split.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/line_split.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/line_split.cc -o CMakeFiles/dmlc.dir/src/io/line_split.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/recordio_split.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o -MF CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o.d -o CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/recordio_split.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/recordio_split.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/recordio_split.cc > CMakeFiles/dmlc.dir/src/io/recordio_split.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/recordio_split.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/recordio_split.cc -o CMakeFiles/dmlc.dir/src/io/recordio_split.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/indexed_recordio_split.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o -MF CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o.d -o CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/indexed_recordio_split.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/indexed_recordio_split.cc > CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/indexed_recordio_split.cc -o CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/input_split_base.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o -MF CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o.d -o CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/input_split_base.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/input_split_base.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/input_split_base.cc > CMakeFiles/dmlc.dir/src/io/input_split_base.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/input_split_base.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/input_split_base.cc -o CMakeFiles/dmlc.dir/src/io/input_split_base.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/filesys.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o -MF CMakeFiles/dmlc.dir/src/io/filesys.cc.o.d -o CMakeFiles/dmlc.dir/src/io/filesys.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/filesys.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/filesys.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/filesys.cc > CMakeFiles/dmlc.dir/src/io/filesys.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/filesys.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/filesys.cc -o CMakeFiles/dmlc.dir/src/io/filesys.cc.s

dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o: dmlc-core/CMakeFiles/dmlc.dir/flags.make
dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o: /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/local_filesys.cc
dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o: dmlc-core/CMakeFiles/dmlc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o -MF CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o.d -o CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o -c /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/local_filesys.cc

dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dmlc.dir/src/io/local_filesys.cc.i"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/local_filesys.cc > CMakeFiles/dmlc.dir/src/io/local_filesys.cc.i

dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dmlc.dir/src/io/local_filesys.cc.s"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core/src/io/local_filesys.cc -o CMakeFiles/dmlc.dir/src/io/local_filesys.cc.s

# Object files for target dmlc
dmlc_OBJECTS = \
"CMakeFiles/dmlc.dir/src/config.cc.o" \
"CMakeFiles/dmlc.dir/src/data.cc.o" \
"CMakeFiles/dmlc.dir/src/io.cc.o" \
"CMakeFiles/dmlc.dir/src/recordio.cc.o" \
"CMakeFiles/dmlc.dir/src/io/line_split.cc.o" \
"CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o" \
"CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o" \
"CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o" \
"CMakeFiles/dmlc.dir/src/io/filesys.cc.o" \
"CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o"

# External object files for target dmlc
dmlc_EXTERNAL_OBJECTS =

dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/config.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/data.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/recordio.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/line_split.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/recordio_split.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/indexed_recordio_split.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/input_split_base.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/filesys.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/src/io/local_filesys.cc.o
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/build.make
dmlc-core/libdmlc.a: dmlc-core/CMakeFiles/dmlc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library libdmlc.a"
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && $(CMAKE_COMMAND) -P CMakeFiles/dmlc.dir/cmake_clean_target.cmake
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dmlc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dmlc-core/CMakeFiles/dmlc.dir/build: dmlc-core/libdmlc.a
.PHONY : dmlc-core/CMakeFiles/dmlc.dir/build

dmlc-core/CMakeFiles/dmlc.dir/clean:
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core && $(CMAKE_COMMAND) -P CMakeFiles/dmlc.dir/cmake_clean.cmake
.PHONY : dmlc-core/CMakeFiles/dmlc.dir/clean

dmlc-core/CMakeFiles/dmlc.dir/depend:
	cd /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/centerco/PycharmProjects/sovrn/xgboost_orig /Users/centerco/PycharmProjects/sovrn/xgboost_orig/dmlc-core /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core /Users/centerco/PycharmProjects/sovrn/xgboost_orig/jvm-packages/dmlc-core/CMakeFiles/dmlc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dmlc-core/CMakeFiles/dmlc.dir/depend

