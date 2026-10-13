# Assembles a release folder from the built artefacts.
#
# Deliberately a copy of exactly three things:
#
#   <name>.vst3/        the VST3 bundle, as a bundle
#   <name>.exe          the standalone build
#   README.txt          what the files are and where they go
#
# Three details this exists to get right:
#
#   * The VST3 is copied as a directory, preserving Contents/. Copying the file out of
#     the bundle - or zipping the bundle's contents rather than the bundle - produces
#     something several hosts will not load, and they fail silently by not listing the
#     plugin rather than by reporting an error.
#
#   * <name>.exp and <name>.lib sit next to the bundle in the build tree. They
#     are link-time artefacts, they are 3 KB, and shipping them invites the question of
#     what they are. They are excluded by copying a named list rather than by copying the
#     directory and deleting afterwards.
#
#   * DEST is emptied first, and it is allowed to be inside the source tree. So the
#     README is read from READMESRC, not from DEST - a script that clears its own input
#     directory works exactly once and then silently ships with no README.
#
# Run with:
#   cmake -DSTAGE=<artefacts dir> -DDEST=<output dir> -DREADMESRC=<file> \
#         -DBUNDLE=<CrunchyPy.vst3> -DEXE=<CrunchyPy.exe> -P Package.cmake
#
# BUNDLE and EXE are passed in rather than hardcoded so the script keeps working
# when the product is renamed, and so a stale name fails loudly at configure time
# instead of at packaging time.

foreach (required STAGE DEST READMESRC BUNDLE EXE)
    if (NOT DEFINED ${required})
        message (FATAL_ERROR "Package.cmake requires -D${required}=<path>")
    endif ()
endforeach ()

if (NOT EXISTS "${READMESRC}")
    message (FATAL_ERROR "Package.cmake: -DREADMESRC does not exist: ${READMESRC}")
endif ()

get_filename_component (STAGE "${STAGE}" ABSOLUTE)
get_filename_component (DEST "${DEST}" ABSOLUTE)

# The bundle is located by name from the caller's BUNDLE argument rather than
# hardcoded. A hardcoded product name here is how the packaging step ends up
# looking for a bundle from a previous project, which fails long after the build
# itself succeeded and gives no hint that the script is simply stale.
set (bundle "${STAGE}/VST3/${BUNDLE}")
set (standalone "${STAGE}/Standalone/${EXE}")

foreach (required "${bundle}" "${standalone}")
    if (NOT EXISTS "${required}")
        message (FATAL_ERROR "missing build artefact: ${required}")
    endif ()
endforeach ()

# Refuse to stage a bundle whose moduleinfo.json is still invalid JSON. The build has a
# normalisation step for this, but that step is a separate target and a partially built
# tree is exactly when someone would package by hand. Better to refuse than to ship a
# bundle that no strict parser will accept.
set (moduleinfo "${bundle}/Contents/Resources/moduleinfo.json")

if (NOT EXISTS "${moduleinfo}")
    message (FATAL_ERROR "bundle has no moduleinfo.json: ${moduleinfo}")
endif ()

file (READ "${moduleinfo}" moduleinfoContents)

foreach (bad ",\\}" ",\\]")
    string (REGEX MATCHALL "[ \t\r\n]*,[ \t\r\n]*${bad}" strays "${moduleinfoContents}")

    if (strays)
        message (FATAL_ERROR "moduleinfo.json still has trailing commas; run the FixBundle target")
    endif ()
endforeach ()

# Read the README before clearing DEST, in case the two ever point at the same place.
file (READ "${READMESRC}" readmeContents)

file (REMOVE_RECURSE "${DEST}")
file (MAKE_DIRECTORY "${DEST}")

# The bundle, as a bundle. copy_directory keeps Contents/ intact, which is the part that
# has to survive.
file (COPY "${bundle}" DESTINATION "${DEST}")
file (COPY "${standalone}" DESTINATION "${DEST}")
file (WRITE "${DEST}/README.txt" "${readmeContents}")

# --- report what was actually written -----------------------------------------------

set (total 0)
set (files 0)

file (GLOB_RECURSE staged "${DEST}/*")

foreach (path IN LISTS staged)
    if (NOT IS_DIRECTORY "${path}")
        file (SIZE "${path}" size)
        math (EXPR total "${total} + ${size}")
        math (EXPR files "${files} + 1")
    endif ()
endforeach ()

math (EXPR mb "${total} / 1024")

message (STATUS "packaged to ${DEST}")
message (STATUS "  ${files} files, ${mb} KB")
message (STATUS "  ${BUNDLE}  (VST3 - copy the whole bundle into your DAW's VST3 folder)")
message (STATUS "  ${EXE}    (standalone - no host required)")
message (STATUS "  README.txt         (copied from ${READMESRC})")