# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.5)

project( geqr2 CXX )
set(CMAKE_PREFIX_PATH "~/Documents/tlapack/")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG  "-g")
# Load <T>LAPACK

if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()
find_package(OpenCV REQUIRED)
# add the example geqr2
add_executable( example_geqr2 example_geqr2.cpp )
target_link_libraries( example_geqr2 PRIVATE tlapack ${OpenCV_LIBS})
target_include_directories(example_geqr2 PRIVATE ${OpenCV_INCLUDE_DIRS})

# add the example vanilla_IR
add_executable( example_vanilla_IR example_vanilla_IR.cpp )
target_link_libraries( example_vanilla_IR PRIVATE tlapack ${OpenCV_LIBS})
target_include_directories(example_vanilla_IR PRIVATE ${OpenCV_INCLUDE_DIRS})
