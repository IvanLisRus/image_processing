cmake_minimum_required(VERSION 3.10)
project(image_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(OpenCV REQUIRED)

add_library(image_processing SHARED image_processing.cpp)

target_include_directories(image_processing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS})

# Убираем префикс "lib" для обеих платформ
set_target_properties(image_processing PROPERTIES PREFIX "")

# Настройка RPATH для Linux (поиск зависимостей в той же папке)
if(UNIX AND NOT APPLE)
    set_target_properties(image_processing PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
endif()
