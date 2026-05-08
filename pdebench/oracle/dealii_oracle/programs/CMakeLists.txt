cmake_minimum_required(VERSION 3.13.4)

# ---------------------------------------------------------------------------
# 1. Locate deal.II  (set DEAL_II_DIR via env or -DDEAL_II_DIR=...)
# ---------------------------------------------------------------------------
find_package(deal.II 9.3
  REQUIRED
  HINTS
    $ENV{DEAL_II_DIR}
    /opt/homebrew/opt/deal.ii            # Apple-Silicon Homebrew (future)
    /usr/local/opt/deal.ii               # Intel Homebrew
    /usr/local                           # Linux system install
    $ENV{HOME}/dealii-candi/deal.II-v9.5.2   # candi default install prefix
)

deal_ii_initialize_cached_variables()

project(dealii_oracle_solvers CXX)

# ---------------------------------------------------------------------------
# 2. nlohmann/json  (header-only JSON library)
#    Priority:  system cmake package  →  Homebrew include  →  FetchContent
# ---------------------------------------------------------------------------
find_package(nlohmann_json 3.9 QUIET)

if(NOT nlohmann_json_FOUND)
  # Try Homebrew paths
  foreach(_prefix
      /opt/homebrew          # Apple-Silicon Homebrew
      /usr/local             # Intel Homebrew / Linux
  )
    if(EXISTS "${_prefix}/include/nlohmann/json.hpp")
      add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
      target_include_directories(nlohmann_json::nlohmann_json
        INTERFACE "${_prefix}/include")
      set(nlohmann_json_FOUND TRUE)
      message(STATUS "Found nlohmann/json at ${_prefix}/include")
      break()
    endif()
  endforeach()
endif()

if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann/json not found locally – fetching from GitHub …")
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---------------------------------------------------------------------------
# 3. Common include directory (case_spec_reader.h, grid_writer.h)
# ---------------------------------------------------------------------------
set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

# ---------------------------------------------------------------------------
# 4. Helper macro to register one oracle executable
# ---------------------------------------------------------------------------
macro(add_oracle_executable TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  deal_ii_setup_target(${TARGET_NAME})
  # deal_ii_setup_target() uses the plain target_link_libraries signature in
  # deal.II 9.7, so we must stay consistent here.
  target_link_libraries(${TARGET_NAME} nlohmann_json::nlohmann_json)
  target_include_directories(${TARGET_NAME}
    PRIVATE ${COMMON_INCLUDE_DIR})
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
  # Install to the build directory so Python can find them by name
  set_target_properties(${TARGET_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endmacro()

# ---------------------------------------------------------------------------
# 5. Oracle executables  (add more in Phase 5)
# ---------------------------------------------------------------------------
add_oracle_executable(poisson_solver              poisson.cc)
add_oracle_executable(heat_solver                 heat.cc)
add_oracle_executable(convection_diffusion_solver convection_diffusion.cc)
add_oracle_executable(helmholtz_solver            helmholtz.cc)
add_oracle_executable(biharmonic_solver           biharmonic.cc)
add_oracle_executable(linear_elasticity_solver    linear_elasticity.cc)
add_oracle_executable(reaction_diffusion_solver   reaction_diffusion.cc)
add_oracle_executable(stokes_solver               stokes.cc)
add_oracle_executable(navier_stokes_solver        navier_stokes.cc)
