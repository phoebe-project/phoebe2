#pragma once

#include <cstdint>
#include <string>

/*
  String hashing for fast switching based on string argument.

  Implemenation of FNV (Fowler–Noll–Vo hash)-1a hash based on the references.

  Author: Martin Horvat, August 2016

  Ref:
  64bit:
  * https://dev.krzaq.cc/post/switch-on-strings-with-c11/
  * http://siliconkiwi.blogspot.com/2012/04/c11-string-switch.html
  * http://seanmiddleditch.com/compile-time-string-hashing-in-c0x/
  32bit:
  * http://create.stephan-brumme.com/fnv-hash/
  Theory:
  * http://www.isthe.com/chongo/tech/comp/fnv/
  * http://engineering.chartbeat.com/2014/08/13/you-dont-know-jack-about-hashing/

*/


/* ====================================================================
  Hasking into 64bit integer
======================================================================*/
namespace fnv1a_64 {

  typedef std::uint64_t hash_t;

  constexpr hash_t prime = 0x100000001B3ull;        // called FNV_prime
  constexpr hash_t basis = 0xCBF29CE484222325ull;   // called offset basis

  // compiler time hashing
  constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis) {
    // recursive way of writing hashing
    return *str ? hash_compile_time(str + 1, (*str ^ last_value) * prime) : last_value;
  }

  // run time  hashing
  hash_t hash(char const* str) {

    hash_t ret{basis};

    while (*str) { // looping over string byte by byte until '\0'
      ret ^= *(str++);
      ret *= prime;
    }

    return ret;
  }

  // run time  hashing
  hash_t hash(const std::string & s) { return hash(s.c_str()); }

} // namespace fnv1a_64

constexpr unsigned long long operator "" _hash64(char const* p, size_t)
{
  return fnv1a_64::hash_compile_time(p);
}


/* ====================================================================
  Hasking into 32bit integer
======================================================================*/
namespace fnv1a_32 {

  typedef std::uint32_t hash_t;

  const hash_t prime = 0x01000193; //   16777619, FNV prime
  const hash_t basis = 0x811C9DC5; // 2166136261, offset basis

  // compiler time hashing
  constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis) {
    // recursive way of writing hashing
    return *str ? hash_compile_time(str + 1, (*str ^ last_value) * prime) : last_value;
  }

  // run time  hashing
  hash_t hash(char const* str) {

    hash_t ret{basis};

    while (*str) { // looping over string byte by byte until '\0'
      ret ^= *(str++);
      ret *= prime;
    }

    return ret;
  }

  hash_t hash(const std::string & s) { return hash(s.c_str()); }

} // namespace fnv1a_32

constexpr unsigned long long operator "" _hash32(char const* p, size_t)
{
  return fnv1a_32::hash_compile_time(p);
}
