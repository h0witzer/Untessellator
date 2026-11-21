// Serial stand-in for pss::parallel_stable_sort (single-threaded WebAssembly build).
#pragma once
#include <algorithm>

namespace pss {
template <typename It, typename Cmp> void parallel_stable_sort(It b, It e, Cmp c) { std::stable_sort(b, e, c); }
}
