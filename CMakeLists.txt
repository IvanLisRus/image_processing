cmake_minimum_required(VERSION 3.10)
project(image_processing LANGUAGES CXX)

# Стандарт C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Оптимизация для Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Поиск OpenCV
find_package(OpenCV REQUIRED)

# Создание библиотеки
add_library(image_processing SHARED image_processing.cpp)

# Подключение заголовков и библиотек OpenCV
target_include_directories(image_processing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS})

# Убираем префикс "lib" для всех платформ
# Windows: image_processing.dll (не libimage_processing.dll)
# Linux: image_processing.so (не libimage_processing.so)
set_target_properties(image_processing PROPERTIES PREFIX "")

# 🔥 RPATH для Linux: поиск зависимостей в той же папке, что и .so
# Это позволяет библиотеке работать портативно без установки OpenCV в систему
if(UNIX AND NOT APPLE)
    set_target_properties(image_processing PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
endif()

# Windows-specific настройки (если нужны)
if(WIN32)
    set_target_properties(image_processing PROPERTIES
        PREFIX ""
    )
endif()
