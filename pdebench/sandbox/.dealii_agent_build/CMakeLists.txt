cmake_minimum_required(VERSION 3.13.4)

# --- deal.II ----------------------------------------------------------------
find_package(deal.II 9.3
  REQUIRED
  HINTS
    $ENV{DEAL_II_DIR}
    /opt/homebrew/opt/deal.ii
    /usr/local/opt/deal.ii
    /usr/local
    $ENV{HOME}/dealii-candi/deal.II-v9.7.0
)
deal_ii_initialize_cached_variables()

project(agent_solver CXX)

# --- nlohmann/json ----------------------------------------------------------
find_package(nlohmann_json 3.9 QUIET)
if(NOT nlohmann_json_FOUND)
  foreach(_prefix
      /opt/homebrew/include
      /usr/local/include
      "/Users/yusan/agent/pdebench/pdebench/oracle/dealii_oracle/build/_deps/nlohmann_json-src/include"
      "/Users/yusan/agent/pdebench/pdebench/oracle/dealii_oracle/build/_deps/nlohmann_json-src/single_include"
  )
    if(EXISTS "${_prefix}/nlohmann/json.hpp")
      add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
      target_include_directories(nlohmann_json::nlohmann_json
        INTERFACE "${_prefix}")
      set(nlohmann_json_FOUND TRUE)
      break()
    endif()
  endforeach()
endif()
if(NOT nlohmann_json_FOUND)
  # Last resort: FetchContent (requires internet)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- Agent solver -----------------------------------------------------------
add_executable(agent_solver solver.cpp)
deal_ii_setup_target(agent_solver)
# deal_ii_setup_target() uses the plain target_link_libraries signature in
# newer deal.II releases, so keep the signature consistent here.
target_link_libraries(agent_solver nlohmann_json::nlohmann_json)
target_compile_features(agent_solver PRIVATE cxx_std_17)
set_target_properties(agent_solver
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
