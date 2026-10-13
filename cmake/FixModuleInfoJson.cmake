# Rewrites a VST3 bundle's moduleinfo.json as strict JSON.
#
# JUCE 8.0.12's moduleinfo.json generator emits trailing commas:
#
#     "Component Non Discardable": false,
#     },
#
#     "Snapshots": [
#     ],
#
# Neither is legal JSON. Hosts parse this file to enumerate the module's classes before
# they will load it, and a strict parser rejects the whole file - which makes the plugin
# fail to appear at all rather than merely mis-behave. It is generated output, so the
# fix has to run after the generator rather than in the source tree.
#
# Run with -DSTART=<a file or directory near the bundle>. The exact layout of a VST3
# bundle differs between build systems and JUCE versions, and TARGET_FILE_DIR for a VST3
# target points at the inner binary rather than the bundle root, so rather than assume a
# path this searches for the file: START itself, the standard spots below it, then
# each parent directory up to six levels deep. It is product-agnostic and does not
# name a bundle, so it survives a rename or a second format. Exits non-zero if it
# cannot find the file or if the result still does not parse, so a broken bundle
# cannot be produced quietly.
#
# Deliberately only removes commas that sit immediately before a closing brace or
# bracket, ignoring whitespace between them. Anything else it cannot handle is left
# alone and reported.

if (NOT DEFINED START)
    message (FATAL_ERROR "FixModuleInfoJson.cmake requires -DSTART=<file or directory>")
endif ()

# --- locate moduleinfo.json -------------------------------------------------------

set (target "")

# Is START itself, or the file it names, the moduleinfo.json we want?
get_filename_component (startAbs "${START}" ABSOLUTE)

if (EXISTS "${startAbs}" AND NOT IS_DIRECTORY "${startAbs}")
    set (target "${startAbs}")
endif ()

if (NOT target)
    # START is a directory. Check the standard spots relative to it.
    foreach (candidate IN ITEMS
            "${startAbs}/Contents/Resources/moduleinfo.json"
            "${startAbs}/moduleinfo.json")
        if (EXISTS "${candidate}")
            set (target "${candidate}")
            break ()
        endif ()
    endforeach ()
endif ()

# START may point *inside* a bundle, because TARGET_FILE_DIR for a VST3 target is
# the inner binary (Contents/x86_64-win) rather than the bundle root. Walking up
# finds the bundle from anywhere inside it. This is deliberately not a recursive
# glob: descending from Contents/x86_64-win can never reach Contents/Resources,
# which is a sibling rather than a child.
if (NOT target)
    set (walk "${startAbs}")

    foreach (unused RANGE 6)
        file (GLOB found "${walk}/*.vst3/Contents/Resources/moduleinfo.json")
        list (LENGTH found foundCount)

        if (foundCount GREATER 0)
            list (GET found 0 target)
            break ()
        endif ()

        get_filename_component (parent "${walk}" DIRECTORY)
        if (parent STREQUAL walk)
            break ()
        endif ()
        set (walk "${parent}")
    endforeach ()
endif ()

# Last resort: sweep the subtree in case the layout is neither of the above.
if (NOT target)
    file (GLOB_RECURSE found "${startAbs}/*.vst3/Contents/Resources/moduleinfo.json")
    list (LENGTH found foundCount)

    if (foundCount EQUAL 1)
        list (GET found 0 target)
    endif ()
endif ()

if (NOT target)
    message (FATAL_ERROR "no moduleinfo.json found near ${START}")
endif ()

# --- strip trailing commas --------------------------------------------------------

# Two passes rather than one combined character class. CMake's regex engine does not
# treat "\]" as an escaped bracket inside a class - "[}\]]" silently fails to match "}",
# which made an earlier version of this script report success while changing nothing.
# "[}]" and "[]]" each work, so they are applied separately.
#
# The comma has to be immediately followed by the closing brace or bracket (bar
# whitespace) so that a comma separating real content, as in "a": 1, is left alone.

set (beforePattern "([ \t\r\n]*),([ \t\r\n]*[}])")
set (afterPattern  "([ \t\r\n]*),([ \t\r\n]*[]])")
set (beforeFind    "[ \t\r\n]*,[ \t\r\n]*[}]")
set (afterFind     "[ \t\r\n]*,[ \t\r\n]*[]]")

file (READ "${target}" contents)

# Repeat until stable: removing a comma before "]" can expose one before "}", and a
# single pass would leave that behind.
set (previous "x")
set (passes 0)

while (NOT contents STREQUAL previous)
    set (previous "${contents}")
    string (REGEX REPLACE "${beforePattern}" "\\1\\2" contents "${contents}")
    string (REGEX REPLACE "${afterPattern}"  "\\1\\2" contents "${contents}")
    math (EXPR passes "${passes} + 1")

    if (passes GREATER 20)
        message (FATAL_ERROR "trailing-comma removal did not converge in ${target}")
    endif ()
endwhile ()

file (WRITE "${target}" "${contents}")

# --- verify rather than assume ----------------------------------------------------

# A silently malformed bundle is the failure mode this change exists to prevent, so it
# must not be able to reintroduce one.
file (READ "${target}" verify)

set (remaining 0)

foreach (findPattern IN ITEMS "${beforeFind}" "${afterFind}")
    string (REGEX MATCHALL "${findPattern}" strays "${verify}")

    if (strays)
        list (LENGTH strays strayCount)
        math (EXPR remaining "${remaining} + ${strayCount}")
    endif ()
endforeach ()

if (remaining GREATER 0)
    message (FATAL_ERROR "${target} still has ${remaining} trailing comma(s)")
endif ()

message (STATUS "moduleinfo.json normalised to strict JSON: ${target}")
