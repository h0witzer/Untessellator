#!/usr/bin/env bash
# Build QuadriFlow's remeshing core to WebAssembly for the engine (js/vendor/quadriflow/).
#   tools/quadriflow/build.sh <qf-dir> <eigen-dir> <boost-dir>
# <qf-dir>: github.com/hjwdzh/QuadriFlow at commit 810b7a0967c35b0dc85b4464e3835e26a756c967 (lemon 1.3.1 and pcg32
# are bundled in its 3rd/); <eigen-dir>: Eigen 3.3.9; <boost-dir>: Boost 1.83.0 (headers only, for its max-flow).
# Needs Emscripten (emsdk). Serial: QuadriFlow's TBB, OpenMP and CUDA paths sit behind macros left undefined here.
# Licenses: tools/quadriflow/NOTICE.
set -euo pipefail
QF="$1"; EIGEN="$2"; BOOST="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../js/vendor/quadriflow"
mkdir -p "$OUT"
# The sources QuadriFlow's CMakeLists.txt compiles, minus main.cpp (remesh.cpp replaces it); post-solver.cpp needs
# Ceres and is not in that list (its one call is commented out).
FILES=""
for f in adjacent-matrix dedge hierarchy loader localsat merge-vertex optimizer parametrizer parametrizer-flip \
         parametrizer-int parametrizer-mesh parametrizer-scale parametrizer-sing subdivide; do FILES="$FILES $QF/src/$f.cpp"; done
em++ -O3 -std=c++14 -fexceptions -DNDEBUG -DEIGEN_MPL2_ONLY \
  -I"$HERE/lemon-config" -I"$QF/src" -I"$QF/3rd/lemon-1.3.1" -I"$QF/3rd/pcg32" -I"$EIGEN" -I"$BOOST" \
  $FILES "$HERE/remesh.cpp" \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=QuadriFlow -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=8MB -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS=_qf_remesh,_malloc,_free -sEXPORTED_RUNTIME_METHODS=FS,ccall,HEAPF64,HEAP32 \
  -o "$OUT/quadriflow.mjs"
ls -la "$OUT"
