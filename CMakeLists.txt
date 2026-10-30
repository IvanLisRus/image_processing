cmake_minimum_required(VERSION 3.10)
project(image_processing LANGUAGES CXX)

# Требую стандарт C++11 (достаточно для этой задачи)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Оптимизация для Release сборок
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Поиск OpenCV
find_package(OpenCV REQUIRED)

# Создание разделяемой библиотеки (DLL в Windows, SO в Linux)
add_library(image_processing SHARED image_processing.cpp)

# Подключение заголовочных файлов OpenCV
target_include_directories(image_processing PRIVATE ${OpenCV_INCLUDE_DIRS})

# Линковка библиотек OpenCV
target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS})

# Для Windows: гарантируем корректный экспорт без префиксов lib
if(WIN32)
    set_target_properties(image_processing PROPERTIES PREFIX "")
endif()
