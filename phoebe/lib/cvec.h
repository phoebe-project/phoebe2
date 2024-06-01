# pragma once

/*
  Library supporting circular/periodic vectors defined as
  by iterators

    it_first ->  it_last

  or vector container.

  Author: Martin Horvat, June 2024
*/


/*
  Next interator:  it + 1 mod range

  Input:
    it : iterator
    it_first : iterator, pointer to first element
    it_last  : iterator, pointer to last element

  Return:
    next it
*/
template <class Tit>
Tit cnext(Tit it, Tit it_first, Tit it_last) {
  return it == it_last ? it_first : it + 1;
}

/*
  Next interator:  it + 1 mod range of vector

  Input:
    it : iterator
    P: vector

  Return:
    next it
*/
template <class V>
typename V::iterator cnext(typename V::iterator it, V & P) {
  return cnext(it, P.begin(), P.end() - 1);
}


/*
  Previous interator:  it - 1 mod range

  Input:
    it  : iterator
    it_first : iterator, pointer to first element
    it_last  : iterator, pointer to last element

  Return:
    previous it
*/
template <class Tit>
Tit cprev(Tit it, Tit it_first, Tit it_last) {
  return it == it_first ? it_last : it - 1;
}


/*
  Previous interator : it - 1 mod range of vector

  Input:
    it : iterator
    P : vector

  Return:
    iterator: previous it
*/
template <class V>
typename V::iterator cprev(typename V::iterator it, V & P) {
  return cprev(it, P.begin(), P.end() - 1);
}

/*
  Copying elements from circular/periodic vector.

  Input:
    it0 : iterator, pointer to first element that needs to be copied
    it1 : iterator, pointer to last element that needs to be copied
    it_first : iterator, pointer to first element of container
    it_last  : iterator, pointer to last element  of container

  Return:
    vector: copied elements from t0 to t1
*/
template <class V>
V ccopy (typename V::iterator it0, typename V::iterator it1,
         typename V::iterator it_first, typename V::iterator it_last) {
  V r;

  auto it = it0;

  while (true) {
    r.push_back(*it);
    if (it == it1) break;
    it = cnext(it, it_first, it_last);
  }

  return r;
}

template <class V>
V ccopy (typename V::iterator it0, typename V::iterator it1, V & P) {
  return ccopy<V>(it0, it1, P.begin(), P.end() - 1);
}

/*
   Erasing elements from circular/periodic vector.

   Input:
      it0 : iterator, pointer to first element that needs to be erased
      it1 : iterator, pointer to last element that needs to be erased
      it_first : iterator, pointer to first element of container
      it_last  : iterator, pointer to last element of container

    Return:
      vector of remainder
*/
template <class V>
V cerase (typename V::iterator it0, typename V::iterator it1,
          typename V::iterator it_first, typename V::iterator it_last) {

  int size = it_last + 1 - it_first;

  std::vector<bool> mask(size, false);

  auto it = it0;

  while (true) {
    mask[int(it - it_first)] = true;
    if (it == it1) break;
    it = cnext(it, it_first, it_last);
  }

  V R;

  for (int i = 0; i < size; ++i) {
    it = it_first + i;
    if (!mask[i]) R.push_back(*it);
  }

  return R;
}

/*
   Erasing elements from circular/periodic vector.

   Input:
      it0 : iterator, pointer to first element that needs to be erased
      it1 : iterator, pointer to last element that needs to be erased
      P :  vector

    Return:
      vector of remainder
*/
template <class V>
V cerase (typename V::iterator it0, typename V::iterator it1, V & P) {

  int size = P.size();

  std::vector<bool> mask(size, false);

  auto it = it0, it_first = P.begin(), it_last = P.end() - 1;

  while (true) {
    mask[int(it - it_first)] = true;
    if (it == it1) break;
    it = cnext(it, it_first, it_last);
  }

  V R;

  for (int i = 0; i < size; ++i) {
    it = it_first + i;
    if (!mask[i]) R.push_back(*it);
  }

  return R;
}



/*
    Split elements from circular/periodic vector into a pair of vectors
    (first, second):

      for it = it0 up to it1  -> first
      everything else         -> second

    Input:
      it0 : iterator, pointer to first element that needs to be copied into first
      it1 : iterator, pointer to last element that needs to be copied into first
      it_first : iterator, pointer to first element of container vector
      it_last  : iterator, pointer to last element  of container vector

    Return:
      pair of vectors (first, second)
*/
template <class V>
std::pair<V,V>
  csplit (typename V::iterator it0, typename V::iterator it1,
          typename V::iterator it_first, typename V::iterator it_last) {

  int size = it_last + 1 - it_first;

  std::vector<bool> mask(size, false);

  auto it = it0;

  while (true) {
    mask[int(it - it_first)] = true;
    if (it == it1) break;
    it = cnext(it, it_first, it_last);
  }

  std::pair<V,V> p;

  for (int i = 0 ; i < size; ++i) {
    it = it_first + i;
    if (mask[i])
      p.first.push_back(*it);
    else
      p.second.push_back(*it);
  }

  return p;
}

/*
    Split elements from circular/periodic vector into a pair of vectors
    (first, second):

      for it = it0 up to it1  -> first
      everything else         -> second

    Input:
      it0 : iterator, pointer to first element that needs to be copied to first
      it1 : iterator, pointer to last element that needs to be copied to first
      P : vector

    Return:
      pair of vectors (first, second)
*/
template <class V>
std::pair<V,V>
  csplit (typename V::iterator it0, typename V::iterator it1, V & P) {

  int size = P.size();

  std::vector<bool> mask(size, false);

  auto it = it0, it_first = P.begin(), it_last = P.end() - 1 ;

  while (true) {
    mask[int(it - it_first)] = true;
    if (it == it1) break;
    it = cnext(it, it_first, it_last);
  }

  std::pair<V,V> p;

  for (int i = 0 ; i < size; ++i) {
    it = it_first + i;
    if (mask[i])
      p.first.push_back(*it);
    else
      p.second.push_back(*it);
  }

  return p;
}

