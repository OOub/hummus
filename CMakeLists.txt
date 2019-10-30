cmake_minimum_required(VERSION 3.12)

# project description
project(Hummus VERSION 1.0
                DESCRIPTION "Hummus spiking neural network simulator"
                LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY AUTOGEN_TARGETS_FOLDER _autogen)

# setting default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# options to handle compilation with or without external dependencies
option(QT    "Compiles with Qt5"  ON)
option(TBB   "Compiles with TBB"  OFF)
option(TORCH "Compile with libtorch" OFF)

######### HUMMUS LIBRARY #########

# creating header-only library
add_library(hummus_library INTERFACE)

# collecting headers automatically - only for CMake 3.12 or later
file(GLOB_RECURSE header_list CONFIGURE_DEPENDS "third_party/*.hpp"
                                                "source/*.hpp")

# allowing IDE to display the headers
target_sources(hummus_library INTERFACE ${header_list})

# preserving directory hierarchy inside IDE
foreach(header IN LISTS header_list)
    file(RELATIVE_PATH path "${CMAKE_SOURCE_DIR}" "${header}")
    source_group(TREE ${CMAKE_SOURCE_DIR} FILES "${path}")
endforeach()

# Setting which directories are needed
target_include_directories(hummus_library INTERFACE third_party/
                                                    source/
                                                    source/GUI
                                                    source/addons
                                                    source/learning_rules
                                                    source/neurons
                                                    source/random_distributions
                                                    source/synapses
                                                    /usr/local/include)

# handles optional Intel Threading Building Blocks (TBB) linking and includes
if(TBB)
    message("-- Compiling with intel TBB")
    add_compile_definitions(TBB)
    find_library(tbb_location NAMES tbb)
    add_library(tbb SHARED IMPORTED)
    set_target_properties(tbb PROPERTIES IMPORTED_LOCATION ${tbb_location})
    target_link_libraries(hummus_library INTERFACE tbb)
endif()

# handles optional Qt 5 linking and includes
if(QT)
    message("-- Compiling with Qt5")
    add_compile_definitions(QT QT_NO_KEYWORDS)
    set_property(GLOBAL PROPERTY AUTOGEN_SOURCE_GROUP _moc)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    # Qt for Linux
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        if(EXISTS /usr/include/qt5/)
            set(CMAKE_PREFIX_PATH /usr/include/qt5/)
        else()
            set(CMAKE_PREFIX_PATH /usr/include/x86_64-linux-gnu/qt5/)
        endif()

        find_package(Qt5 COMPONENTS Core Gui Qml Quick Widgets Charts REQUIRED)

        target_include_directories(hummus_library INTERFACE ${Qt5Core_INCLUDE_DIRS}
                                                            ${Qt5Gui_INCLUDE_DIRS}
                                                            ${Qt5Qml_INCLUDE_DIRS}
                                                            ${Qt5Quick_INCLUDE_DIRS}
                                                            ${Qt5Widgets_INCLUDE_DIRS}
                                                            ${Qt5Charts_INCLUDE_DIRS})

        target_link_libraries(hummus_library INTERFACE Qt5::Core
                                                       Qt5::Gui
                                                       Qt5::Qml
                                                       Qt5::Quick
                                                       Qt5::Widgets
                                                       Qt5::Charts)

    # Qt for macOS
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_PREFIX_PATH /usr/local/opt/qt/)
        find_package(Qt5 COMPONENTS Core Gui Qml Quick Widgets QuickControls2 Charts REQUIRED)

        target_include_directories(hummus_library INTERFACE ${Qt5Core_INCLUDE_DIRS}
                                                            ${Qt5Gui_INCLUDE_DIRS}
                                                            ${Qt5Qml_INCLUDE_DIRS}
                                                            ${Qt5Quick_INCLUDE_DIRS}
                                                            ${Qt5Widgets_INCLUDE_DIRS}
                                                            ${Qt5QuickControls2_INCLUDE_DIRS}
                                                            ${Qt5Charts_INCLUDE_DIRS})

        target_link_libraries(hummus_library INTERFACE Qt5::Core
                                                       Qt5::Gui
                                                       Qt5::Qml
                                                       Qt5::Quick
                                                       Qt5::Widgets
                                                       Qt5::QuickControls2
                                                       Qt5::Charts)

    endif()

    source_group(_moc REGULAR_EXPRESSION "\\.rule$")
endif()

# handles optional libtorch (C++ frontend of pytorch) linking and includes
if(TORCH)
    message("-- Compiling with Libtorch")
    find_package(Torch REQUIRED)
    target_link_libraries(hummus_library INTERFACE ${TORCH_LIBRARIES})
    add_compile_definitions(TORCH)
endif()

# All users of this library will need at least C++17
target_compile_features(hummus_library INTERFACE cxx_std_17)

######### HUMMUS EXECUTABLES #########

# gather all .cpp inside the applications folder
file(GLOB application_list CONFIGURE_DEPENDS "applications/*.cpp")

# build executable for each application
foreach(app IN LISTS application_list)
    get_filename_component(app_name "${app}" NAME_WE)
    if (QT)
        add_executable(${app_name} applications/${app_name}.cpp source/GUI/gui.qrc)
    else()
        add_executable(${app_name} applications/${app_name}.cpp)
    endif()
    source_group("applications" FILES applications/${app_name}.cpp)
    target_compile_features(${app_name} PRIVATE cxx_std_17)
    target_link_libraries(${app_name} PRIVATE hummus_library)
endforeach()
