# Stage the importable 'polymerfts' package in the build tree:
#   cmake -DSRC=<src/python> -DDST=<stage dir> -DCORE=<_core.so> -P stage_python.cmake
# Modules that no longer exist in SRC are removed from DST, so the staged
# package never carries files from an earlier version.
file(MAKE_DIRECTORY "${DST}")
file(GLOB_RECURSE _staged RELATIVE "${DST}" "${DST}/*.py")
foreach(_name ${_staged})
    if(NOT EXISTS "${SRC}/${_name}")
        file(REMOVE "${DST}/${_name}")
    endif()
endforeach()
# A failed copy (e.g. a _core module that a running Python process keeps locked
# on Windows) fails the build, so that tests never import a stale package.
file(GLOB_RECURSE _sources RELATIVE "${SRC}" "${SRC}/*.py")
foreach(_name ${_sources})
    get_filename_component(_folder "${_name}" DIRECTORY)
    file(MAKE_DIRECTORY "${DST}/${_folder}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SRC}/${_name}" "${DST}/${_folder}"
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Could not stage ${_name} in ${DST}")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CORE}" "${DST}" RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Could not stage ${CORE} in ${DST}")
endif()
