# ifndef _ARRAY_HPP
# define _ARRAY_HPP

# include <iostream>
# include <map>


/// A wrapper class for holding an array.
///
/// An array can be implemented by an array or a std::map. If the
/// array is sparse, the std::map would be the better implementation.
/// Otherwise an array is more efficient. The class would automatically
/// choose its implementation during construction according to different
/// inputs.
template<typename _INDEX, typename _VALUE>
class Array {
public:
    Array(_VALUE _default_value = 0);
    Array(Array& some);
    /// Constructor of an Array.
    ///
    /// @param[in] _array A simple array. If sparse enough, it will
    ///                   be converted into a std::map and the array will be
    ///                   released. Otherwise it will be directly encapsulated
    ///                   into the class.
    Array(_VALUE* _array, _INDEX _length, _VALUE _default_value = 0);
    ~Array();

    Array& operator=(Array& some);

    /// Insert or replace the array inside the class
    Array& insert(_VALUE* _array, _INDEX _length);
    Array& insert(_VALUE* _array, _INDEX _length, _VALUE _default_value);
    /// Make a histogram of length "_bin" by counting the number of times that
    /// the numbers 1~"_bin" appear in the array of length "_length".
    Array& histogram(_INDEX* _array, _VALUE _length, _INDEX _bin = 0);
    /// Read-only access to an arbitary element of index i.
    _VALUE operator[](const _INDEX i) const;
    _INDEX length() const { return length_; }
    Array& clear();

private:
    bool   alloc;
    bool   use_map;
    _INDEX length_;
    _VALUE default_value;
    union {
        _VALUE*                   array;
        std::map<_INDEX, _VALUE>* map;
    } arr;
};


template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>::Array(_VALUE _default_value):
                       alloc(false),
                       use_map(false),
                       length_(0),
                       default_value(_default_value) {
    arr.array = NULL;
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>::Array(Array& some):
                       alloc(false),
                       use_map(some.use_map),
                       length_(some.length_),
                       default_value(some.default_value) {
    if (use_map) arr.map = some.arr.map;
    else arr.array = some.arr.array;
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>::Array(_VALUE* _array, _INDEX _length, _VALUE _default_value):
                       alloc(false),
                       use_map(false),
                       length_(0),
                       default_value(_default_value) {
    arr.array = NULL;
    insert(_array, _length);
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>::~Array() {
    clear();
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>& Array<_INDEX, _VALUE>::
operator=(Array& some) {
    if (alloc && ((use_map != some.use_map) ||
           (!use_map && arr.array != some.arr.array) ||
           (use_map && arr.map != some.arr.map))) {
        clear();
    }
    use_map       = some.use_map;
    length_       = some.length_;
    default_value = some.default_value;
    if (use_map) arr.map = some.arr.map;
    else arr.array = some.arr.array;
    return *this;
}

template<typename _INDEX, typename _VALUE>
Array<_INDEX, _VALUE>& Array<_INDEX, _VALUE>::
insert(_VALUE* _array, _INDEX _length) {
    clear();
    _INDEX n_nondefault = 0, i_nondefault;
    for (_INDEX i = 0; i < _length; ++i) {
        if (_array[i] != default_value) { n_nondefault++; i_nondefault = i; }
    }
    if (n_nondefault > 0) {
        use_map = (_length / n_nondefault) > 3;
        if (use_map) {
            arr.map = new std::map<_INDEX, _VALUE>;
            typename std::map<_INDEX, _VALUE>::iterator arr_it;
            arr_it = arr.map->emplace(i_nondefault, _array[i_nondefault]).first;
            for (_INDEX i = i_nondefault - 1; i > 0; --i) {
                if (_array[i] != default_value)
                    arr_it = arr.map->emplace_hint(arr_it, i, _array[i]);
            }
            if (_array[0] != default_value) {
                arr_it = arr.map->emplace_hint(arr_it, 0, _array[0]);
            }
            delete[] _array;
        }
        else {
            arr.array = _array;
        }
        alloc = true;
    }
    length_ = _length;
    return *this;
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>& Array<_INDEX, _VALUE>::
insert(_VALUE* _array, _INDEX _length, _VALUE _default_value) {
    default_value = _default_value;
    return insert(_array, _length);
}

template<typename _INDEX, typename _VALUE>
Array<_INDEX, _VALUE>& Array<_INDEX, _VALUE>::
histogram(_INDEX* _array, _VALUE  _length, _INDEX  _bin) {
    clear();
    if (!_bin) {
        for (_VALUE i = 0; i < _length; ++i) {
            if (_array[i] > _bin) _bin = _array[i];
        }
    }
    _VALUE* array = new _VALUE[_bin];
    for (_VALUE i = 0; i < _length; ++i) {
        if (_array[i]) ++array[_array[i] - 1];
    }
    insert(array, _bin);
    return *this;
}

template<typename _INDEX, typename _VALUE> inline
_VALUE Array<_INDEX, _VALUE>::
operator[](const _INDEX i) const {
    if (use_map) {
        typename std::map<_INDEX, _VALUE>::iterator it = arr.map->find(i);
        if (it != arr.map->end()) return it->second;
        else return default_value;
    }
    else {
        if (arr.array != NULL && i < length_)
            return arr.array[i];
        else return default_value;
    }
}

template<typename _INDEX, typename _VALUE> inline
Array<_INDEX, _VALUE>& Array<_INDEX, _VALUE>::
clear() {
    if (alloc) {
        if (use_map) delete arr.map;
        else delete[] arr.array;
        alloc = false;
    }
    use_map   = false;
    length_   = 0;
    arr.array = NULL;
    return *this;
}

template<typename _INDEX, typename _VALUE>
std::ostream& operator<<(std::ostream& out, Array<_INDEX, _VALUE>& h) {
    out << h[0];
    _INDEX length = h.length();
    for (_INDEX i = 1; i < length; ++i) {
        out << " " << h[i];
    }
    return out;
}

# endif
