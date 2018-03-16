#ifndef __TUPLE
#define __TUPLE

template <class T>
class tuple
{
// I genuinly just wanted a generic struct...
public:
    tuple() = default;
    tuple( T _x, T _y ) { x = _x; y = _y; }
    T x;
    T y;
};

#endif // __TUPLE