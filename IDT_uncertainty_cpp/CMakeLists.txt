cmake_minimum_required(VERSION 3.14)
project(UncertaintyQuantification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(/bigobj)

include(FetchContent)

find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(pugixml REQUIRED)

add_executable(IDT_uncertainty src/UncertaintyQuantification.cpp)

# Tell compiler where to find headers
target_include_directories(IDT_uncertainty PRIVATE
    ${CMAKE_SOURCE_DIR}/libs
)

target_link_libraries(IDT_uncertainty 
  PRIVATE
    nlohmann_json::nlohmann_json
    Ceres::ceres
    pugixml::pugixml
)


