#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <new>

// Cache line size: 128 bytes on Apple Silicon M1/M2/M3, 64 bytes on x86-64.
// Aligning to 128 bytes guarantees alignment on both architectures.
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 128
#endif

namespace utils {

// 128-byte aligned memory allocation using posix_memalign
template <typename T>
inline T* aligned_malloc(size_t count, size_t alignment = CACHE_LINE_SIZE) {
    if (count == 0) return nullptr;
    size_t bytes = count * sizeof(T);
    // Ensure bytes is at least a multiple of alignment for safety
    if (bytes % alignment != 0) {
        bytes = ((bytes + alignment - 1) / alignment) * alignment;
    }
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, bytes);
    if (ret != 0 || ptr == nullptr) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

inline void aligned_free(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

// Custom deleter for unique_ptr with aligned memory
template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        aligned_free(ptr);
    }
};

// High-resolution wall-clock timer
class Timer {
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time_;
        return diff.count();
    }

    double elapsed_ms() const {
        return elapsed_seconds() * 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// Fast 64-bit pseudo-random number generator (xoshiro256**)
// Excellent statistical properties, period 2^256 - 1, extremely fast.
class Xoshiro256StarStar {
public:
    Xoshiro256StarStar(uint64_t seed = 0x853c49e6748fea9bULL) {
        init(seed);
    }

    void init(uint64_t seed) {
        // Initialize state using splitmix64
        uint64_t sm = seed;
        for (int i = 0; i < 4; ++i) {
            sm += 0x9e3779b97f4a7c15ULL;
            uint64_t z = sm;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    // Returns random uint64_t
    inline uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    // Uniform real in [0, 1)
    inline double next_double() {
        return (next() >> 11) * 0x1.0p-53;
    }

    // Advance state by 2^128 steps (for independent thread streams)
    void jump() {
        static const uint64_t JUMP[] = { 
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL, 
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL 
        };

        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t i = 0; i < sizeof(JUMP) / sizeof(*JUMP); ++i) {
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }
        }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

private:
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t s[4];
};

} // namespace utils

#endif // UTILS_HPP
