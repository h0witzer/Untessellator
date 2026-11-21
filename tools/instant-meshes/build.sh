#!/usr/bin/env bash
# Build Instant Meshes' remeshing core to WebAssembly for the engine (js/vendor/instant-meshes/).
#   tools/instant-meshes/build.sh <src-dir>
# <src-dir> holds im/ (github.com/wjakob/instant-meshes, commit 7b3160864a2e1025af498c84cfed91cbfb613698) and ext/
# with pcg32, dset (github.com/wjakob) and eigen (gitlab.com/libeigen/eigen, tag 3.3.9). Needs Emscripten (emsdk).
# Single-threaded: TBB and pss are replaced by the serial stand-ins in shim/. Licences: tools/instant-meshes/NOTICE.
set -euo pipefail
SRC="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../js/vendor/instant-meshes"
mkdir -p "$OUT"
CORE="adjacency bvh cleanup dedge diff extract field hierarchy meshio meshstats normal reorder serializer smoothcurve subdivide"
FILES=""
for f in $CORE; do FILES="$FILES $SRC/im/src/$f.cpp"; done
TMP="$(mktemp -d)"
emcc -O3 -c "$SRC/im/ext/rply/rply.c" -o "$TMP/rply.o"          # C, not C++
em++ -O3 -std=c++14 -fexceptions -DNDEBUG -DEIGEN_MPL2_ONLY \
  -I"$HERE/shim" -I"$SRC/im/src" -I"$SRC/im/ext/half" -I"$SRC/im/ext/rply" -I"$SRC/ext/eigen" -I"$SRC/ext/pcg32" -I"$SRC/ext/dset" \
  $FILES "$TMP/rply.o" "$HERE/remesh.cpp" \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=InstantMeshes -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=8MB -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS=_im_remesh -sEXPORTED_RUNTIME_METHODS=FS,ccall \
  -o "$OUT/instant-meshes.mjs"
ls -la "$OUT"
