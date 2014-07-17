# ifndef DATA_INL_
# define DATA_INL_

# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

# include <cstring>
# include <cmath>
# include <random>
# include <boost/python.hpp>
# include <numpy/arrayobject.h>
# include "Q.hpp"
# include "Array.hpp"

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
Data() : buffer(NULL),
         managed_index(NULL),
         num_of_samples_(0),
         labels(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
Data(_COMP_T* x, _SUPV_T* y,
     _DAT_DIM_T dimension, _N_DAT_T total_samples,
     _N_DAT_T _num_of_samples, _N_DAT_T* index, bool copy_index)
      : buffer(NULL),
        managed_index(NULL),
        num_of_samples_(0),
        labels(NULL) {
    load(x, y, dimension, total_samples, _num_of_samples, index, copy_index);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
Data(const Data& some, _N_DAT_T _num_of_samples, _N_DAT_T* index, bool copy_index)
      : buffer(NULL),
        managed_index(NULL),
        num_of_samples_(0),
        labels(NULL) {
    load(some, _num_of_samples, index, copy_index);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
~Data() {
    clear();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load(_COMP_T* x, _SUPV_T* y,
     _DAT_DIM_T dimension, _N_DAT_T total_samples,
     _N_DAT_T _num_of_samples, _N_DAT_T* index, bool copy_index) {
    load_buffer(x, y, dimension, total_samples);
    if (_num_of_samples <= 0) _num_of_samples = total_samples;
    load_index(_num_of_samples, index, NULL, copy_index);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load(const Data& some,
     _N_DAT_T _num_of_samples, _N_DAT_T* index, bool copy_index) {
    if (&some != this) {
        clear_buffer();
        buffer = some.buffer;
        if (buffer != NULL) {
            buffer->incre_ref();
            if (_num_of_samples <= 0) {
                _num_of_samples = some.num_of_samples_;
                index = NULL;
                copy_index = true;
            }
        }
    }
    if (buffer != NULL)
        load_index(_num_of_samples, index, some.managed_index, copy_index);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
void Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
check(_N_DAT_T _num_of_samples, bool random) {
    if (_num_of_samples > num_of_samples_) {
        _num_of_samples = num_of_samples_;
        random = false;
    }
    iterator* it;
    if (random) {
        it = new permute_iterator(*this, _num_of_samples);
    } else {
        it = new iterator(*this);
    }
    for (_N_DAT_T i = 0; i < _num_of_samples && *it; ++i, ++(*it)) {
        std::cout << "Local index = " << it->index()
                  << ", Global index = " << managed_index[it->index()]
                  << ", y = " << it->y()
                  << ", X = [";
        for (_DAT_DIM_T d = 0; d < dimension(); ++d) {
            std::cout << " " << (*it)[d];
        }
        std::cout << " ]" << std::endl;
    }
    delete it;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
operator bool() const {
    return buffer != NULL && managed_index != NULL && (*buffer);
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T* Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
x(const _N_DAT_T i) const {
    if (i < 0 || i >= num_of_samples_ ) {
        std::cerr << "**ERROR** -> Data::x(" << i << ") -> Index exceeds."
                  << std::endl;
        return NULL;
    }
    return buffer->x(managed_index[i]);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T& Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
x(const _N_DAT_T i, const _DAT_DIM_T d) const {
    if (d < 0 || d >= dimension()) {
        std::cerr << "**ERROR** -> Data::x(" << i << ", " << d
                  << ") -> Dimension exceeds." << std::endl;
        return x(0)[0];
    }
    return x(i)[d];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
y(const _N_DAT_T i) const {
    return buffer->y(managed_index[i]);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T& Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
ry(const _N_DAT_T i) {
    delete labels; labels = NULL;
    return buffer->y(managed_index[i]);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
num_of_samples() const {
    return num_of_samples_;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_DAT_DIM_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
dimension() const {
    return buffer->dimension();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T* Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
operator[](const _N_DAT_T i) const {
    return buffer->x(managed_index[i]);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
num_of_labels() {
    if (labels == NULL) labels = new Labels(*this);
    return labels->num_of_labels;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
label(const _SUPV_T i) {
    if (labels == NULL) labels = new Labels(*this);
    return labels->label[i];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
index_of_label(const _SUPV_T k) {
    if (labels == NULL) labels = new Labels(*this);
    return labels->index_of_label[k - 1];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
num_of_samples_with_label(const _SUPV_T i) {
    if (labels == NULL) labels = new Labels(*this);
    return labels->num_of_samples_with_label[i];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T* Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
indices_of_samples_with_label(const _SUPV_T i) {
    if (labels == NULL) labels = new Labels(*this);
    return labels->index_of_sample_with_label[i];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
entropy() {
    if (labels == NULL) labels = new Labels(*this);
    return labels->entropy;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
Data(bpy::object& x, bpy::object& y)
      : buffer(NULL),
        managed_index(NULL),
        num_of_samples_(0),
        labels(NULL) {
    _load(x, y);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
Data(bpy::object& x, bpy::object& y, bpy::object& indices)
      : buffer(NULL),
        managed_index(NULL),
        num_of_samples_(0),
        labels(NULL) {
    _load(x, y, indices);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_load(bpy::object& x, bpy::object& y) {
    bpy::extract<Data> x_is_Data(x);
    if (x_is_Data.check()) {
        return _load(x_is_Data(), y);
    }
    load_buffer(x, y);
    load_index(buffer->num_of_samples());
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_load(bpy::object& x, bpy::object& y, bpy::object& indices) {
    load_buffer(x, y);
    load_index(indices);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_load(const Data& some, const bpy::object& indices) {
    if (&some != this) {
        clear_buffer();
        buffer = some.buffer;
        if (buffer != NULL) buffer->incre_ref();
    }
    if (buffer != NULL) load_index(indices, some.managed_index);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
bpy::tuple Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_data() const {
    if (buffer == NULL) return bpy::make_tuple();
    return buffer->buffer();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
bpy::object Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_indices() const {
    return c_array_to_numpy_1d_array(managed_index, num_of_samples_, false);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
bpy::object Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_labels() {
    if (labels == NULL) labels = new Labels(*this);
    return c_array_to_numpy_1d_array(labels->label, labels->num_of_labels, false);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
bpy::object Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
_num_of_sample_of_each_label() {
    if (labels == NULL) labels = new Labels(*this);
    return c_array_to_numpy_1d_array(labels->num_of_samples_with_label, labels->num_of_labels, false);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
iterator() : data(NULL),
             x(NULL),
             it_index(0),
             it_index_range(0) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
iterator(Data& _data) : data(NULL),
                        x(NULL),
                        it_index(0),
                        it_index_range(0) {
    begin(_data);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
index() const {
    return it_index;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
pos() const {
    return it_index;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
pos(const _N_DAT_T _pos) {
    it_index = _pos;
    if (*this) x = (*data)[index()];
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
begin() {
    return pos(0);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
begin(Data& _data) {
    data = &_data;
    it_index_range = data->num_of_samples();
    return begin();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
operator++() {
    ++it_index;
    if (it_index < it_index_range)
        x = (*data)[index()];
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T& Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
operator[](const _DAT_DIM_T d) const {
    return x[d];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>*
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
operator->() const {
    return data;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_DAT_DIM_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
dimension() const {
    return data->dimension();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
y() const {
    return data->y(index());
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T& Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
ry() const {
    return data->ry(index());
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
operator bool() const {
    return it_index < it_index_range;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator::
operator Data&() const {
    return *data;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
permute_iterator()
      : permute_range(0),
        permute_index(NULL) {
    init_rand();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
permute_iterator(Data& _data, _N_DAT_T _permute_range)
      : permute_range(0),
        permute_index(NULL) {
    init_rand();
    begin(_data, _permute_range);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
~permute_iterator() {
    delete[] permute_index;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
index() const {
    return permute_index[this->it_index];
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
begin() {
    if (this->it_index == 0) return *this;
    permute();
    this->pos(0);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
begin(Data& _data) {
    return begin(_data, 0);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
begin(Data& _data, _N_DAT_T _permute_range) {
    if (this->it_index_range != _data.num_of_samples()) {
        delete[] permute_index;
        this->it_index_range = _data.num_of_samples();
        permute_index = new _N_DAT_T[this->it_index_range];
        for (_N_DAT_T i = 0; i < this->it_index_range; ++i) {
            permute_index[i] = i;
        }
        this->it_index = this->it_index_range;
    }
    if (_permute_range == 0) _permute_range = _data.num_of_samples();
    permute_range = _permute_range;
    this->data = &_data;
    return begin();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
permute() {
    // static std::mt19937 rand_gen;
    for (_N_DAT_T i = 0; i < permute_range; ++i) {
        std::uniform_int_distribution<_N_DAT_T> distrib(i, this->it_index_range - 1);
        _N_DAT_T temp_i       = distrib(rand_gen);
        _N_DAT_T temp         = permute_index[temp_i];
        permute_index[temp_i] = permute_index[i];
        permute_index[i]      = temp;
    }
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
int Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator::
init_rand() {
    std::seed_seq::result_type rand_seq[std::mt19937::state_size];
    std::random_device rand_dev;
    std::generate(rand_seq, rand_seq + std::mt19937::state_size,
                  std::ref(rand_dev));
    std::seed_seq rand_seed(rand_seq, rand_seq + std::mt19937::state_size);
    rand_gen.seed(rand_seed);
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
label_iterator() : index_of_sample_with_label(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
label_iterator(Data& _data, _SUPV_T i)
      : index_of_sample_with_label(NULL) {
    begin(_data, i);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
index() const {
    return index_of_sample_with_label[this->it_index];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
begin(Data& _data) {
    return begin(_data, 0);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
begin(Data& _data, _SUPV_T i) {
    this->data = &_data;
    return begin(i);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::label_iterator::
begin(_SUPV_T i) {
    this->it_index_range = this->data->num_of_samples_with_label(i);
    index_of_sample_with_label = this->data->indices_of_samples_with_label(i);
    begin();
    return *this;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
Buffer() : ref_count(1),
           pyobj_x(NULL),
           pyobj_y(NULL),
           dimension_(0),
           total_samples(0),
           buffer_x(NULL),
           buffer_y(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
Buffer(_COMP_T* _buffer_x, _SUPV_T* _buffer_y,
       _DAT_DIM_T _dimension, _N_DAT_T _total_samples)
      : ref_count(1),
        pyobj_x(NULL),
        pyobj_y(NULL),
        dimension_(_dimension),
        total_samples(_total_samples),
        buffer_x(_buffer_x),
        buffer_y(_buffer_y) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
~Buffer() {
    if (pyobj_x != NULL) Py_DECREF(pyobj_x);
    else delete[] buffer_x;
    if (pyobj_y != NULL) Py_DECREF(pyobj_y);
    else delete[] buffer_y;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
void Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
incre_ref() {
    ++ref_count;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
void Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
decre_ref() {
    --ref_count;
    if (ref_count <= 0) {
        delete this;
    }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
operator bool() const {
    return ref_count >= 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_DAT_DIM_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
dimension() const {
    return dimension_;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_N_DAT_T Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
num_of_samples() const {
    return total_samples;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T* Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
x(_N_DAT_T i) const {
    return buffer_x + dimension_ * i;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_SUPV_T& Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
y(_N_DAT_T i) const {
    return buffer_y[i];
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
Buffer(bpy::object& x, bpy::object& y)
      : ref_count(-1),
        pyobj_x(NULL),
        pyobj_y(NULL),
        dimension_(0),
        total_samples(0),
        buffer_x(NULL),
        buffer_y(NULL) {
    int      ndim_x, ndim_y;
    npy_intp *shape_x, *shape_y;
    if (parse_numpy_array(x, pyobj_x, buffer_x, ndim_x, shape_x))
        return;
    if (parse_numpy_array(y, pyobj_y, buffer_y, ndim_y, shape_y))
        return;
    if (ndim_x != 2) {
        std::cerr << "**ERROR** -> Buffer(x, y)@" << this
                  << " -> x is not a matrix (ndim = " << ndim_x << ")."
                  << std::endl;
        return;
    }
    if (ndim_y != 1 && ndim_y != 2) {
        std::cerr << "**ERROR** -> Buffer(x, y)@" << this
                  << " -> y is not a matrix or vector (ndim = " << ndim_y
                  << ")." << std::endl;
        return;
    }
    dimension_    = shape_x[1];
    total_samples = shape_x[0];
    _N_DAT_T total_samples_y = (ndim_y == 1 ?
                                shape_y[0] : (shape_y[0] > shape_y[1] ?
                                shape_y[0] : shape_y[1]));
    if (total_samples != total_samples_y) {
        std::cerr << "**ERROR** -> Buffer(x, y)@" << this
                  << " -> Number of samples in x (" << total_samples
                  << ") and y (" << total_samples_y << ") mismatch."
                  << std::endl;
        return;
    }
    ref_count = 1;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
bpy::tuple Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Buffer::
buffer() const {
    if (pyobj_x == NULL || pyobj_y == NULL) return bpy::make_tuple();
    return bpy::make_tuple(bpy::object(bpy::handle<>(bpy::borrowed(pyobj_x))),
                           bpy::object(bpy::handle<>(bpy::borrowed(pyobj_y))));
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Labels::
Labels(Data& data) {
    _SUPV_T max_label = 0;
    for (_N_DAT_T i = 0; i < data.num_of_samples(); ++i) {
        if (data.y(i) > max_label) max_label = data.y(i);
    }
    _N_DAT_T* max_label_sample_count = new _N_DAT_T[max_label];
    std::memset(max_label_sample_count, 0, sizeof(_N_DAT_T) * max_label);
    for (_N_DAT_T i = 0; i < data.num_of_samples(); ++i) {
        ++max_label_sample_count[data.y(i) - 1];
    }
    num_of_labels = 0;
    for (_SUPV_T i = 0; i < max_label; ++i) {
        if (max_label_sample_count[i] > 0) ++num_of_labels;
    }
    label = new _SUPV_T[num_of_labels];
    num_of_samples_with_label = new _N_DAT_T[num_of_labels];
    index_of_sample_with_label = new _N_DAT_T*[num_of_labels];
    _SUPV_T* full_index_of_label = new _SUPV_T[max_label];
    std::memset(num_of_samples_with_label, 0, sizeof(_N_DAT_T) * num_of_labels);
    std::memset(full_index_of_label, -1, sizeof(_SUPV_T) * max_label);
    _SUPV_T num_of_labels_count = 0;
    for (_SUPV_T i = 0; i < max_label; ++i) {
        if (max_label_sample_count[i] > 0) {
            label[num_of_labels_count] = i + 1;
            num_of_samples_with_label[num_of_labels_count] = max_label_sample_count[i];
            index_of_sample_with_label[num_of_labels_count] = new _N_DAT_T[max_label_sample_count[i]];
            full_index_of_label[i] = num_of_labels_count;
            ++num_of_labels_count;
        }
    }
    delete[] max_label_sample_count;
    index_of_label.insert(full_index_of_label, max_label, -1);
    _N_DAT_T* num_of_samples_with_label_count = new _N_DAT_T[num_of_labels];
    std::memset(num_of_samples_with_label_count, 0, sizeof(_N_DAT_T) * num_of_labels);
    for (_N_DAT_T i = 0; i < data.num_of_samples(); ++i) {
        _SUPV_T index_of_label_of_sample = index_of_label[data.y(i) - 1];
        index_of_sample_with_label[index_of_label_of_sample][num_of_samples_with_label_count[index_of_label_of_sample]++] = i;
    }
    delete[] num_of_samples_with_label_count;
    entropy = 0;
    if (num_of_labels > 1) {
        for (_SUPV_T i = 0; i < num_of_labels; ++i) {
            if (num_of_samples_with_label[i] == 0) continue;
            _COMP_T p = static_cast<_COMP_T>(num_of_samples_with_label[i]) /
                        data.num_of_samples();
            entropy -= p * std::log2(p);
        }
    }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::Labels::
~Labels() {
    delete[] label;
    delete[] num_of_samples_with_label;
    if (index_of_sample_with_label != NULL) {
        for (_SUPV_T i = 0; i < num_of_labels; ++i) {
            delete[] index_of_sample_with_label[i];
        }
        delete[] index_of_sample_with_label;
    }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
clear() {
    clear_buffer();
    clear_index();
    clear_labels();
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
clear_buffer() {
    if (buffer != NULL) {
        buffer->decre_ref();
        buffer = NULL;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
clear_index() {
    delete[] managed_index;
    managed_index   = NULL;
    num_of_samples_ = 0;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
clear_labels() {
    delete labels;
    labels = NULL;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load_buffer(_COMP_T* x, _SUPV_T* y,
            _DAT_DIM_T dimension, _N_DAT_T total_samples) {
    clear_buffer();
    buffer = new Buffer(x, y, dimension, total_samples);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load_index(_N_DAT_T _num_of_samples, _N_DAT_T* index,
           _N_DAT_T* index_from, bool copy_index) {
    if (_num_of_samples <= 0) return *this;
    clear_labels();
    bool index_from_self = (index_from == managed_index);
    if (index_from_self) managed_index = NULL;
    if (index == NULL) {
        copy_index = num_of_samples_ < _num_of_samples ||
                     managed_index == NULL;
        if (copy_index) {
            clear_index();
            managed_index = new _N_DAT_T[_num_of_samples];
        }
        if (index_from == NULL)
            for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
                managed_index[i] = i;
            }
        else
            for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
                managed_index[i] = index_from[i];
            }
    } else {
        clear_index();
        if (copy_index) {
            managed_index = new _N_DAT_T[_num_of_samples];
            if (index_from == NULL)
                std::memcpy(managed_index, index,
                            sizeof(_N_DAT_T) * _num_of_samples);
        } else {
            managed_index = index;
        }
        if (index_from != NULL)
            for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
                managed_index[i] = index_from[index[i]];
            }
    }
    if (index_from_self) delete[] index_from;

    // if (index == NULL && index_from == NULL) {
    //     copy_index = copy_index ||
    //                  num_of_samples_ < _num_of_samples ||
    //                  managed_index == NULL;
    //     if (copy_index) {
    //         clear_index();
    //         managed_index = new _N_DAT_T[_num_of_samples];
    //     }
    //     clear_labels();
    //     for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
    //         managed_index[i] = i;
    //     }
    // } else if (index == NULL && index_from != NULL) {
    //     copy_index = copy_index ||
    //                  num_of_samples_ < _num_of_samples ||
    //                  managed_index == NULL;
    //     if (copy_index) {
    //         clear_index();
    //         managed_index = new _N_DAT_T[_num_of_samples];
    //     }
    //     clear_labels();
    //     for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
    //         managed_index[i] = index_from[i];
    //     }
    // } else if (index != NULL && index_from == NULL) {
    //     clear_index();
    //     clear_labels();
    //     if (copy_index) {
    //         managed_index = new _N_DAT_T[_num_of_samples];
    //         std::memcpy(managed_index, index,
    //                     sizeof(_N_DAT_T) * _num_of_samples);
    //     } else {
    //         managed_index = index;
    //     }
    // } else {
    //     clear_index();
    //     clear_labels();
    //     if (copy_index) {
    //         managed_index = new _N_DAT_T[_num_of_samples];
    //     } else {
    //         managed_index = index;
    //     }
    //     for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
    //         managed_index[i] = index_from[index[i]];
    //     }
    // }

    // if (index == NULL || copy_index) {
    //     new_managed_index = new _N_DAT_T[_num_of_samples];
    //     if (index == NULL) {
    //         for (_N_DAT_T i = 0; i < _num_of_samples; ++i) {
    //             new_managed_index[i] = i;
    //         }
    //     } else {
    //         std::memcpy(new_managed_index, index,
    //                     sizeof(_N_DAT_T) * _num_of_samples);
    //     }
    // } else {
    //     new_managed_index = index;
    // }
    // clear_index();
    // managed_index   = new_managed_index;
    num_of_samples_ = _num_of_samples;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load_buffer(bpy::object& x, bpy::object& y) {
    clear_buffer();
    buffer = new Buffer(x, y);
    if (!(*buffer)) {
        buffer->decre_ref();
        buffer = NULL;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
load_index(const bpy::object& bpy_indices, _N_DAT_T* index_from) {
    PyObject* py_indices;
    _N_DAT_T* indices;
    int       ndim;
    npy_intp* shape;
    if (parse_numpy_array(bpy_indices, py_indices, indices, ndim, shape)) {
        std::cerr << "**ERROR** -> Data::load(data, indices) "
                     "-> Cannot parse indices." << std::endl;
        clear();
        return *this;
    }
    if (ndim > 2) {
        std::cerr << "**ERROR** -> Data::load(data, indices) "
                     "-> indices is an ndarray which has more "
                     "than 2 dimensions." << std::endl;
        clear();
        return *this;
    }
    _N_DAT_T num_of_samples = (ndim == 1 ?
                               shape[0] : (shape[0] > shape[1] ?
                                           shape[0] : shape[1]));
    load_index(num_of_samples, indices, index_from, true);
    Py_DECREF(py_indices);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
bpy::tuple Data_bpy_pickle<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
getinitargs(const Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data) {
    if (data) {
        bpy::tuple  x_and_y = data._data();
        bpy::object indices = data._indices();
        return bpy::make_tuple(x_and_y[0], x_and_y[1], indices);
    }
    return bpy::make_tuple();
}

# endif
