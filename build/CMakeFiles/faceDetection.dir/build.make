# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python2.7/site-packages/cmake/data/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python2.7/site-packages/cmake/data/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/prawira/projects/EE4208_FD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/prawira/projects/EE4208_FD/build

# Include any dependencies generated for this target.
include CMakeFiles/faceDetection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/faceDetection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/faceDetection.dir/flags.make

CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o: CMakeFiles/faceDetection.dir/flags.make
CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o: ../src/faceDetectionV2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/prawira/projects/EE4208_FD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o -c /Users/prawira/projects/EE4208_FD/src/faceDetectionV2.cpp

CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/prawira/projects/EE4208_FD/src/faceDetectionV2.cpp > CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.i

CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/prawira/projects/EE4208_FD/src/faceDetectionV2.cpp -o CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.s

# Object files for target faceDetection
faceDetection_OBJECTS = \
"CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o"

# External object files for target faceDetection
faceDetection_EXTERNAL_OBJECTS =

../faceDetection: CMakeFiles/faceDetection.dir/src/faceDetectionV2.cpp.o
../faceDetection: CMakeFiles/faceDetection.dir/build.make
../faceDetection: /usr/local/lib/libopencv_gapi.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_stitching.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_aruco.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_bgsegm.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_bioinspired.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_ccalib.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_dnn_objdetect.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_dnn_superres.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_dpm.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_face.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_freetype.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_fuzzy.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_hfs.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_img_hash.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_line_descriptor.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_quality.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_reg.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_rgbd.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_saliency.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_sfm.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_stereo.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_structured_light.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_superres.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_surface_matching.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_tracking.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_videostab.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_xfeatures2d.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_xobjdetect.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_xphoto.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_highgui.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_shape.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_datasets.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_plot.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_text.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_dnn.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_ml.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_phase_unwrapping.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_optflow.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_ximgproc.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_video.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_videoio.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_imgcodecs.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_objdetect.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_calib3d.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_features2d.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_flann.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_photo.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_imgproc.4.2.0.dylib
../faceDetection: /usr/local/lib/libopencv_core.4.2.0.dylib
../faceDetection: CMakeFiles/faceDetection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/prawira/projects/EE4208_FD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../faceDetection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faceDetection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/faceDetection.dir/build: ../faceDetection

.PHONY : CMakeFiles/faceDetection.dir/build

CMakeFiles/faceDetection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/faceDetection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/faceDetection.dir/clean

CMakeFiles/faceDetection.dir/depend:
	cd /Users/prawira/projects/EE4208_FD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/prawira/projects/EE4208_FD /Users/prawira/projects/EE4208_FD /Users/prawira/projects/EE4208_FD/build /Users/prawira/projects/EE4208_FD/build /Users/prawira/projects/EE4208_FD/build/CMakeFiles/faceDetection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/faceDetection.dir/depend

