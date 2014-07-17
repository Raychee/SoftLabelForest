# ifndef _LABELSTAT_HPP
# define _LABELSTAT_HPP

# include <iostream>
# include <fstream>
# include <iomanip>
# include <cstring>
# include <cstdlib>
# include <ctime>
# include <cmath>

# include "Qlib.hpp"
# include "Array.hpp"

/// A template class which calculates the statistics of a label set.
///
/// The "LabelStat" class will calculate:
///     Number of labels in the set;
///     What are the labels in the set;
///     Number of samples that have the same label;
///     The indexes of samples that have the same label.
/// The labels are represented as integers, which range from 1~s, where "s" is
/// the total number of the labels.
/// @warning            Any label smaller than 1 will cause segment fault.
/// @warning            Only the first object created will have the power to
///                     deallocate the memory. Any object that is copied or
///                     assigned will only have the pointer to the memory.
/// @tparam _SUPV_T     Type of labels. (Only integer types are allowed.)
/// @tparam _N_DAT_T    Type of indexes. (Only integer types are allowed.)
template<typename _SUPV_T, typename _N_DAT_T>
class LabelStat {
public:
    LabelStat();
    LabelStat(_SUPV_T* y, _N_DAT_T n, _N_DAT_T* x = NULL);
    LabelStat(LabelStat& some);
    LabelStat& operator=(LabelStat& some);
    ~LabelStat();

    /// The main method of the class.
    ///
    /// Calculates all the information mentioned.
    /// @param  y   A sequence of label samples.
    /// @param  n   The length of "y" if x==NULL; the length of "x"
    ///             if x!=NULL.
    /// @param  x   The indexes of samples that is really taken
    ///             into consideration. If NULL, then all the
    ///             samples in "y" will be involved.
    LabelStat& stat(_SUPV_T*  y, _N_DAT_T  n, _N_DAT_T* x = NULL);
    LabelStat& insert(_SUPV_T n_label, _SUPV_T* _label, _N_DAT_T* _n_x_of_label);
    LabelStat& clear();

    _SUPV_T    num_of_labels()  const { return n_label; }
    _N_DAT_T   num_of_samples() const { return n_sample; }
    /// Return the ith label in the set.
    _SUPV_T    label(_SUPV_T i) const { return label_[i]; }
    /// Return the index of label "k" in the label set.
    _SUPV_T    index_of_label(_SUPV_T k) { return i_label[k - 1]; }
    /// Return the number of samples that have the ith label in the label set.
    _N_DAT_T   num_of_samples_with_label(_SUPV_T i) const { return n_x_of_label[i]; }
    /// Return the index of the sample which has the kth label and appears at the
    /// ith place in the set "y".
    _N_DAT_T   index_of_sample_with_label(_SUPV_T k, _N_DAT_T i) const
        { return x_of_label[k][i]; }
    /// A more convenient way to retrieve the indexes of every sample.
    _N_DAT_T*  operator[](_SUPV_T k) { return x_of_label[k]; }
    /// Output all the indexes to the array "x". "x" must be at least "n_sample"
    /// in size.
    _N_DAT_T*  index_of_samples();
    /// Shuffle the indexes of the samples with the kth label.
    ///
    /// @param[in]  k   the kth label.
    /// @param[in]  m   If 0, then all the indexes of kth label will be shuffled;
    ///                 Otherwise only m indexes will be randomly chosen and be
    ///                 put ahead of the index array.
    LabelStat& rand_index(_SUPV_T k, _N_DAT_T m = 0);
    /// Uniformly shuffle all the indexes in the set and store in "x".
    /// @param[out] x   The memory space for the shuffled indexes.
    /// @param[in]  n   If 0, then "x" is assumed to be of length "n_sample";
    ///                 Otherwise only "n" indexes are randomly chosen out of the
    ///                 set and stored in "x".
    LabelStat& rand_index(_N_DAT_T* x, _N_DAT_T n = 0);
    double     entropy() const;

    LabelStat& ostream_this(std::ostream& out);
    LabelStat& ofstream_this(std::ofstream& out);
    LabelStat& istream_this(std::istream& in);

private:
    bool       alloc;           ///< whether this has allocated some memory
    _N_DAT_T*  x_sample;
    _SUPV_T    n_label;
    _N_DAT_T   n_sample;
    _SUPV_T*   label_;
    _N_DAT_T*  n_x_of_label;
    _N_DAT_T** x_of_label;
    Array<_SUPV_T, _SUPV_T> i_label;
};

template<typename _SUPV_T, typename _N_DAT_T> inline
std::ostream& operator<<(std::ostream& out, LabelStat<_SUPV_T, _N_DAT_T>& st) {
    st.ostream_this(out);
    return out;
}

template<typename _SUPV_T, typename _N_DAT_T> inline
std::ofstream& operator<<(std::ofstream& out, LabelStat<_SUPV_T, _N_DAT_T>& st) {
    st.ofstream_this(out);
    return out;
}

template<typename _SUPV_T, typename _N_DAT_T> inline
std::istream& operator>>(std::istream& in, LabelStat<_SUPV_T, _N_DAT_T>& st) {
    st.istream_this(in);
    return in;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>::LabelStat():
                              alloc(false),
                              x_sample(NULL),
                              n_label(0),
                              n_sample(0),
                              label_(NULL),
                              n_x_of_label(NULL),
                              x_of_label(NULL) {
    std::srand(std::time(NULL));
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>::LabelStat(_SUPV_T*  y,
                                        _N_DAT_T  n,
                                        _N_DAT_T* x):
                              alloc(false),
                              x_sample(NULL),
                              n_label(0),
                              n_sample(0),
                              label_(NULL),
                              n_x_of_label(NULL),
                              x_of_label(NULL) {
    std::srand(std::time(NULL));
    stat(y, n, x);
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>::LabelStat(LabelStat& some):
                              alloc(false),
                              x_sample(some.x_sample),
                              n_label(some.n_label),
                              n_sample(some.n_sample),
                              label_(some.label_),
                              n_x_of_label(some.n_x_of_label),
                              x_of_label(some.x_of_label),
                              i_label(some.i_label) {
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>::~LabelStat() {
    clear();
}

template<typename _SUPV_T, typename _N_DAT_T> inline
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
operator=(LabelStat& some) {
    if (alloc && x_sample != some.x_sample) clear();
    x_sample     = some.x_sample;
    n_label      = some.n_label;
    n_sample     = some.n_sample;
    label_       = some.label_;
    n_x_of_label = some.n_x_of_label;
    x_of_label   = some.x_of_label;
    i_label      = some.i_label;
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
stat(_SUPV_T*  y, _N_DAT_T  n, _N_DAT_T* x) {
    clear();
    n_sample = n;
    bool alloc_x = !x;
    if (alloc_x) {
        x = new _N_DAT_T[n];
        for (_N_DAT_T i = 0; i < n; ++i) x[i] = i;
    }
    _SUPV_T max_label = 0;
    for (_N_DAT_T i = 0; i < n; ++i) {
        if (y[x[i]] > max_label) max_label = y[x[i]];
    }
    _N_DAT_T* max_n_x_of_label = new _N_DAT_T[max_label];
    std::memset(max_n_x_of_label, 0, max_label * sizeof(_N_DAT_T));
    for (_N_DAT_T i = 0; i < n; ++i) {
        ++max_n_x_of_label[y[x[i]] - 1];
    }
    n_label = 0;
    for (_SUPV_T i = 0; i < max_label; ++i) {
        if (max_n_x_of_label[i]) ++n_label;
    }
    label_       = new _SUPV_T[n_label];
    n_x_of_label = new _N_DAT_T[n_label];
    x_of_label   = new _N_DAT_T*[n_label];
    _SUPV_T count_label  = 0;
    _SUPV_T* max_i_label = new _SUPV_T[max_label];
    std::memset(n_x_of_label, 0, n_label * sizeof(_N_DAT_T));
    std::memset(max_i_label, 0, max_label * sizeof(_SUPV_T));
    for (_SUPV_T i = 0; i < max_label; ++i) {
        if (max_n_x_of_label[i]) {
            label_[count_label] = i + 1;
            n_x_of_label[count_label] = max_n_x_of_label[i];
            max_i_label[i] = count_label;
            ++count_label;
        }
    }
    i_label.insert(max_i_label, max_label);
    delete[] max_n_x_of_label;
    _N_DAT_T* x_of_label_count = new _N_DAT_T[n_label];
    std::memset(x_of_label_count, 0, n_label * sizeof(_N_DAT_T));
    for (_SUPV_T i = 0; i < n_label; ++i) {
        x_of_label[i] = new _N_DAT_T[n_x_of_label[i]];
    }
    for (_N_DAT_T i = 0; i < n; ++i) {
        _N_DAT_T x_i     = x[i];
        _SUPV_T  label_i = i_label[y[x_i] - 1];
        x_of_label[label_i][x_of_label_count[label_i]++] = x_i;
    }
    delete[] x_of_label_count;
    if (alloc_x) delete[] x;
    // if (x) {
    //     // _N_DAT_T*  max_x_of_label_count = new _N_DAT_T[max_label];
    //     // _N_DAT_T** max_x_of_label       = new _N_DAT_T*[max_label];
    //     // std::memset(max_x_of_label_count, 0, max_label * sizeof(_N_DAT_T));
    //     // for (_SUPV_T i = 0; i < n_label; ++i) {
    //     //     max_x_of_label[label_[i] - 1] = new _N_DAT_T[n_x_of_label[i]];
    //     // }
    //     // for (_N_DAT_T i = 0; i < n; ++i) {
    //     //     _N_DAT_T x_i = x[i];
    //     //     _SUPV_T  y_i = y[x_i] - 1;
    //     //     max_x_of_label[y_i][max_x_of_label_count[y_i]++] = x_i;
    //     // }
    //     // delete[] max_x_of_label_count;
    //     // for (_SUPV_T i = 0; i < n_label; ++i) {
    //     //     x_of_label[i] = max_x_of_label[label_[i] - 1];
    //     // }
    //     // delete[] max_x_of_label;
    // }
    // else {
    //     n_label = 0;
    //     for (_N_DAT_T i = 0; i < n; ++i) {
    //         if (y[i] > n_label) n_label = y[i];
    //     }
    //     label_       = new _SUPV_T[n_label];
    //     n_x_of_label = new _N_DAT_T[n_label];
    //     x_of_label   = new _N_DAT_T*[n_label];
    //     _SUPV_T* max_i_label = new _SUPV_T[n_label];
    //     std::memset(n_x_of_label, 0, n_label * sizeof(_N_DAT_T));
    //     for (_SUPV_T i = 0; i < n_label; ++i) {
    //         label_[i] = i + 1;
    //         max_i_label[i] = i;
    //     }
    //     i_label.insert(max_i_label, n_label);
    //     for (_N_DAT_T i = 0; i < n; ++i) {
    //         ++n_x_of_label[y[i] - 1];
    //     }
    //     for (_SUPV_T i = 0; i < n_label; ++i) {
    //         x_of_label[i] = new _N_DAT_T[n_x_of_label[i]];
    //     }
    //     _N_DAT_T* x_of_label_count = new _N_DAT_T[n_label];
    //     std::memset(x_of_label_count, 0, n_label * sizeof(_N_DAT_T));
    //     for (_N_DAT_T i = 0; i < n; ++i) {
    //         _SUPV_T  y_i = y[i] - 1;
    //         x_of_label[y_i][x_of_label_count[y_i]++] = i;
    //     }
    //     delete[] x_of_label_count;
    // }
    alloc = true;
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T> inline
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
insert(_SUPV_T _n_label, _SUPV_T* _label, _N_DAT_T* _n_x_of_label) {
    clear();
    n_label      = _n_label;
    label_       = _label;
    n_x_of_label = _n_x_of_label;
    _SUPV_T max_label = 0;
    for (_SUPV_T i = 0; i < n_label; ++i) {
        n_sample += n_x_of_label[i];
        if (label_[i] > max_label) max_label = label_[i];
    }
    _SUPV_T* max_i_label = new _SUPV_T[max_label];
    for (_SUPV_T i = 0; i < n_label; ++i) {
        max_i_label[label_[i] - 1] = i;
    }
    i_label.insert(max_i_label, max_label);
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
ostream_this(std::ostream& out) {
    out << "Label set: Total " << n_sample << " samples, "
        << n_label << " labels: ";
    for (_SUPV_T i = 0; i < n_label; ++i) {
        out << std::setw(4) << label_[i];
    }
    for (_SUPV_T i = 0; i < n_label; ++i) {
        _N_DAT_T  n_x_label = n_x_of_label[i];
        // _N_DAT_T* x_label   = x_of_label[i];
        out << "\n    Label " << std::setw(4) << label_[i]
            << " has " << std::setw(4) << n_x_label << " samples. ";
        // for (_N_DAT_T j = 0; j < n_x_label; ++j) {
        //     out << std::setw(4) << x_label[j];
        // }
    }
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
ofstream_this(std::ofstream& out) {
    out << n_sample << "    # number of samples\n"
        << n_label << "    # number of labels\n";
    for (_SUPV_T i = 0; i < n_label; ++i) {
        out << label_[i] << " " << n_x_of_label[i]
            << "    # label | number of samples\n";
    }
    out << entropy() << "    # entropy\n";
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
istream_this(std::istream& in) {
    clear();
    char line_str[SIZEOF_LINE];
    in.getline(line_str, SIZEOF_LINE);
    _N_DAT_T total_samples = strto<_N_DAT_T>(line_str);
    in.getline(line_str, SIZEOF_LINE);
    n_label = strto<_SUPV_T>(line_str);
    label_ = new _SUPV_T[n_label];
    n_x_of_label = new _N_DAT_T[n_label];
    alloc = true;
    for (_SUPV_T i = 0; i < n_label; ++i) {
        in.getline(line_str, SIZEOF_LINE);
        char* delimit;
        label_[i] = strto<_SUPV_T>(line_str, delimit);
        n_x_of_label[i] = strto<_N_DAT_T>(delimit);
        n_sample += n_x_of_label[i];
    }
    if (n_sample != total_samples) {
        std::cerr << "Number of samples mismatch." << std::endl;
    }
    in.getline(line_str, SIZEOF_LINE);
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T> inline
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
clear() {
    if (alloc) {
        delete[] x_sample;
        delete[] label_;
        delete[] n_x_of_label;
        if (x_of_label) {
            for (_SUPV_T i = 0; i < n_label; ++i) {
                delete[] x_of_label[i];
            }
            delete[] x_of_label;
        }
        alloc = false;
    }
    x_sample     = NULL;
    n_label      = 0;
    n_sample     = 0;
    label_       = NULL;
    n_x_of_label = NULL;
    x_of_label   = NULL;
    i_label.clear();
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
_N_DAT_T* LabelStat<_SUPV_T, _N_DAT_T>::
index_of_samples() {
    if (!x_sample) {
        x_sample = new _N_DAT_T[n_sample];
        _N_DAT_T count = 0;
        for (_SUPV_T i = 0; i < n_label; ++i) {
            _N_DAT_T  n_x_label = n_x_of_label[i];
            _N_DAT_T* x_label   = x_of_label[i];
            for (_N_DAT_T i = 0; i < n_x_label; ++i) {
                x_sample[count++] = x_label[i];
            }
        }
        if (count != n_sample) {
            std::cerr << "Unknown error in LabelStat::index_of_samples()."
                      << std::endl;
        }
    }
    return x_sample;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
rand_index(_SUPV_T k, _N_DAT_T m) {
    _N_DAT_T n = n_x_of_label[k];
    if (!m) m = n;
    _N_DAT_T* index = x_of_label[k];
    for (_N_DAT_T i = 0; i < m; ++i) {
        _N_DAT_T temp_i = std::rand() % (n - i) + i;
        _N_DAT_T temp = index[temp_i];
        index[temp_i] = index[i];
        index[i] = temp;
    }
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
LabelStat<_SUPV_T, _N_DAT_T>& LabelStat<_SUPV_T, _N_DAT_T>::
rand_index(_N_DAT_T* x, _N_DAT_T n) {
    if (!n || n > n_sample) n = n_sample;
    _N_DAT_T* n_subsample_of_label = new _N_DAT_T[n_label];
    _N_DAT_T max_n_x_of_label = 0;
    for (_SUPV_T i = 0; i < n_label; ++i) {
        _N_DAT_T n_subsample = n_subsample_of_label[i]
                             = n_x_of_label[i] * n / n_sample;
        if (n_subsample > max_n_x_of_label)
            max_n_x_of_label = n_subsample;
        rand_index(i, n_subsample);
    }
    float*    step_of_label    = new float[n_label];
    float*    level_of_label   = new float[n_label];
    _N_DAT_T* x_of_label_count = new _N_DAT_T[n_label];
    std::memset(x_of_label_count, 0, n_label * sizeof(_N_DAT_T));
    for (_SUPV_T i = 0; i < n_label; ++i) level_of_label[i] = 0;
    for (_SUPV_T i = 0; i < n_label; ++i)
        step_of_label[i] = (float) n_subsample_of_label[i] / max_n_x_of_label;
    _N_DAT_T count = 0;
    _SUPV_T  i     = 0;
    while (count < n) {
        level_of_label[i] += step_of_label[i];
        if (level_of_label[i] > 1) {
            level_of_label[i] -= 1;
            x[count++] = x_of_label[i][x_of_label_count[i]++];
        }
        if (++i >= n_label) i = 0;
    }
    delete[] step_of_label;
    delete[] level_of_label;
    delete[] x_of_label_count;
    delete[] n_subsample_of_label;
    return *this;
}

template<typename _SUPV_T, typename _N_DAT_T>
double LabelStat<_SUPV_T, _N_DAT_T>::
entropy() const {
    if (n_label <= 1) return 0;
    double  ent = 0;
    for (_SUPV_T i = 0; i < n_label; ++i) {
        double p = (double)n_x_of_label[i] / n_sample;
        ent -= p * std::log2(p);
    }
    return ent;
}


# endif
