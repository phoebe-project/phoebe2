#if !defined (__hash_h)
#define __hash_h

#include <cstdint>

/*
  String hashing for fast switching based on string argument.
   
  Implemenation of FNV (Fowler–Noll–Vo hash)-1a hash based on the references.

  Author: Martin Horvat, August 2016

  Ref:
  * https://dev.krzaq.cc/post/switch-on-strings-with-c11/
  * http://siliconkiwi.blogspot.com/2012/04/c11-string-switch.html
  * http://seanmiddleditch.com/compile-time-string-hashing-in-c0x/
  * https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
*/  
  
namespace fnv1a_64 {
  
  typedef std::uint64_t hash_t;
 
  constexpr hash_t prime = 0x100000001B3ull;
  constexpr hash_t basis = 0xCBF29CE484222325ull;
  
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
   
} // namespace fnv1a_64
   
constexpr unsigned long long operator "" _hash(char const* p, size_t)
{
  return fnv1a_64::hash_compile_time(p);
}

#endif
