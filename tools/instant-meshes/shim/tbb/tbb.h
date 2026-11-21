// Serial stand-ins for the parts of Intel TBB that Instant Meshes uses, for a single-threaded WebAssembly build
// (GitHub Pages cannot send the cross-origin isolation headers that browser threads need). Same results, one core.
#pragma once
#include <algorithm>
#include <cstddef>
#include <functional>
#include <new>
#include <queue>
#include <vector>

namespace tbb {

template <typename T> class blocked_range {
public:
    typedef T const_iterator;
    typedef std::size_t size_type;
    blocked_range() : b_(), e_(), g_(1) { }
    blocked_range(T b, T e, std::size_t g = 1) : b_(b), e_(e), g_(g) { }
    T begin() const { return b_; }
    T end() const { return e_; }
    std::size_t size() const { return std::size_t(e_ - b_); }
    std::size_t grainsize() const { return g_; }
    bool empty() const { return !(b_ < e_); }
    bool is_divisible() const { return false; }
private:
    T b_, e_;
    std::size_t g_;
};

template <typename Range, typename Body> void parallel_for(const Range &r, const Body &body) { body(r); }
template <typename Index, typename Func> void parallel_for(Index first, Index last, const Func &f) {
    for (Index i = first; i < last; ++i) f(i);
}

template <typename Range, typename Value, typename Func, typename Reduction>
Value parallel_reduce(const Range &r, const Value &identity, const Func &f, const Reduction &) { return f(r, identity); }
template <typename Range, typename Value, typename Func, typename Reduction>
Value parallel_deterministic_reduce(const Range &r, const Value &identity, const Func &f, const Reduction &) {
    return f(r, identity);
}

template <typename It, typename Cmp> void parallel_sort(It b, It e, const Cmp &c) { std::sort(b, e, c); }
template <typename It> void parallel_sort(It b, It e) { std::sort(b, e); }

class spin_mutex {
public:
    class scoped_lock {
    public:
        scoped_lock() { }
        explicit scoped_lock(spin_mutex &) { }
        void acquire(spin_mutex &) { }
        void release() { }
    };
    void lock() { }
    void unlock() { }
};

class task_scheduler_init {
public:
    enum { automatic = -1, deferred = -2 };
    explicit task_scheduler_init(int = automatic) { }
    static int default_num_threads() { return 1; }
};

template <typename T> class concurrent_vector : public std::vector<T> {
public:
    using std::vector<T>::vector;
    typename std::vector<T>::iterator push_back(const T &v) { std::vector<T>::push_back(v); return this->end() - 1; }
};

template <typename T, typename Cmp = std::less<T>> class concurrent_priority_queue {
public:
    void push(const T &v) { q_.push(v); }
    bool try_pop(T &v) { if (q_.empty()) return false; v = q_.top(); q_.pop(); return true; }
    bool empty() const { return q_.empty(); }
    std::size_t size() const { return q_.size(); }
private:
    std::priority_queue<T, std::vector<T>, Cmp> q_;
};

// Tasks, run depth-first: a spawned task runs at once; a task that returns itself (recycled) runs again.
struct allocate_root_proxy { };
struct allocate_continuation_proxy { };
struct allocate_child_proxy { };
class task {
public:
    virtual ~task() { }
    virtual task *execute() = 0;
    static allocate_root_proxy allocate_root() { return {}; }
    allocate_continuation_proxy allocate_continuation() { return {}; }
    allocate_child_proxy allocate_child() { return {}; }
    void set_ref_count(int) { }
    void recycle_as_child_of(task &) { }
    static void spawn(task &t) { run(&t); }
    static void spawn_root_and_wait(task &t) { run(&t); }
    static void run(task *t) {
        while (t) {
            task *next = t->execute();
            if (next != t) delete t;
            t = next;
        }
    }
};
class empty_task : public task {
public:
    task *execute() override { return nullptr; }
};

} // namespace tbb

inline void *operator new(std::size_t s, const tbb::allocate_root_proxy &) { return ::operator new(s); }
inline void *operator new(std::size_t s, const tbb::allocate_continuation_proxy &) { return ::operator new(s); }
inline void *operator new(std::size_t s, const tbb::allocate_child_proxy &) { return ::operator new(s); }
inline void operator delete(void *p, const tbb::allocate_root_proxy &) { ::operator delete(p); }
inline void operator delete(void *p, const tbb::allocate_continuation_proxy &) { ::operator delete(p); }
inline void operator delete(void *p, const tbb::allocate_child_proxy &) { ::operator delete(p); }
