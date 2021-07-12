cmake_minimum_required(VERSION 2.6)

add_executable(brake_assistant brake_assistant.cpp)
target_link_libraries(brake_assistant ${TRONIS_CORE_LIB} ${TRONIS_GRABBER_LIB})
target_link_libraries(brake_assistant ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

if (NOT SDK_MODE)
    add_dependencies(brake_assistant tronis_core tronis_grabber)
endif()
set_property(TARGET brake_assistant
             PROPERTY EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE)
