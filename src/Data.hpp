# ifndef DATA_HPP_
# define DATA_HPP_

# include <iostream>
# include <random>
# include <boost/python.hpp>
# include "Array.hpp"

namespace bpy = boost::python;

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class Data {
  public:
    Data();
    Data(_COMP_T* x, _SUPV_T* y,
         _DAT_DIM_T dimension, _N_DAT_T total_samples,
         _N_DAT_T num_of_samples = 0, _N_DAT_T* index = NULL,
         bool copy_index = true);
    Data(const Data& some,
         _N_DAT_T num_of_samples = 0, _N_DAT_T* index = NULL,
         bool copy_index = true);
    virtual ~Data();

    Data& load(_COMP_T* x, _SUPV_T* y,
               _DAT_DIM_T dimension, _N_DAT_T total_samples,
               _N_DAT_T num_of_samples = 0, _N_DAT_T* index = NULL,
               bool copy_index = true);
    Data& load(const Data& some,
               _N_DAT_T num_of_samples = 0, _N_DAT_T* index = NULL,
               bool copy_index = true);

    void  check(_N_DAT_T num_of_samples = 10, bool random = true);

    operator bool() const;

    _COMP_T*   x(const _N_DAT_T i)                     const;
    _COMP_T&   x(const _N_DAT_T i, const _DAT_DIM_T d) const;
    _SUPV_T    y(const _N_DAT_T i)                     const;
    _SUPV_T&   ry(const _N_DAT_T i);
    _N_DAT_T   num_of_samples()                        const;
    _DAT_DIM_T dimension()                             const;
    _COMP_T*   operator[](const _N_DAT_T i)            const;

    _SUPV_T    num_of_labels();
    _SUPV_T    label(const _SUPV_T i);
    _SUPV_T    index_of_label(const _SUPV_T k);
    _N_DAT_T   num_of_samples_with_label(const _SUPV_T i);
    _N_DAT_T*  indices_of_samples_with_label(const _SUPV_T i);
    _COMP_T    entropy();

    /// Boost.Python interface
    Data(bpy::object& x, bpy::object& y);
    Data(bpy::object& x, bpy::object& y, bpy::object& indices);

    Data& _load(bpy::object& x, bpy::object& y);
    Data& _load(bpy::object& x, bpy::object& y, bpy::object& indices);
    Data& _load(const Data& some, const bpy::object& indices);

    bpy::tuple  _data()                        const;
    bpy::object _indices()                     const;
    bpy::object _labels();
    bpy::object _num_of_sample_of_each_label();

    class iterator {
      public:
        iterator();
        iterator(Data& data);
        virtual ~iterator() {}
        virtual _N_DAT_T  index()                        const;
        _N_DAT_T          pos()                          const;
        iterator&         pos(const _N_DAT_T _pos);
        virtual iterator& begin();
        virtual iterator& begin(Data& data);
        iterator&         operator++();
        _COMP_T&          operator[](const _DAT_DIM_T d) const;
        Data*             operator->()                   const;
        _DAT_DIM_T        dimension()                    const;
        _SUPV_T           y()                            const;
        _SUPV_T&          ry()                           const;
        operator          bool()                         const;
        operator          Data&()                        const;
      protected:
        Data*      data;
        _COMP_T*   x;
        _N_DAT_T   it_index;
        _N_DAT_T   it_index_range;
    };

    class permute_iterator : public iterator {
      public:
        permute_iterator();
        permute_iterator(Data& data, _N_DAT_T permute_range = 0);
        virtual ~permute_iterator();
        virtual _N_DAT_T          index() const;
        virtual permute_iterator& begin();
        virtual permute_iterator& begin(Data& data);
        permute_iterator&         begin(Data& data, _N_DAT_T permute_range);
      protected:
        _N_DAT_T     permute_range;
        _N_DAT_T*    permute_index;
        std::mt19937 rand_gen;
        int                       permute();
        int                       init_rand();
    };

    class label_iterator : public iterator {
      public:
        label_iterator();
        label_iterator(Data& data, _SUPV_T i = 0);
        virtual _N_DAT_T        index() const;
        using iterator::begin;
        virtual label_iterator& begin(Data& data);
        label_iterator&         begin(Data& data, _SUPV_T i);
        label_iterator&         begin(_SUPV_T i);
      protected:
        _N_DAT_T* index_of_sample_with_label;
    };

  protected:
    class Buffer {
      public:
        Buffer();
        Buffer(_COMP_T* buffer_x, _SUPV_T* buffer_y,
               _DAT_DIM_T dimension, _N_DAT_T total_samples);
        ~Buffer();

        void       incre_ref();
        void       decre_ref();
        operator   bool() const;

        _DAT_DIM_T dimension()      const;
        _N_DAT_T   num_of_samples() const;
        _COMP_T*   x(_N_DAT_T i)    const;
        _SUPV_T&   y(_N_DAT_T i)    const;

        /// Python interface
        Buffer(bpy::object& x, bpy::object& y);
        bpy::tuple buffer() const;

      private:
        int         ref_count;       ///< reference count of this
        PyObject*   pyobj_x;
        PyObject*   pyobj_y;

        _DAT_DIM_T  dimension_;
        _N_DAT_T    total_samples;
        _COMP_T*    buffer_x;        ///< buffer for training samples' features
        _SUPV_T*    buffer_y;        ///< buffer for training samples' labels
    };

    Buffer*    buffer;
    _N_DAT_T*  managed_index;   ///< global indexes of samples which is actually managed
    _N_DAT_T   num_of_samples_; ///< number of samples specified by managed_index

    struct Labels {
        _SUPV_T                 num_of_labels;
        _SUPV_T*                label;
        _N_DAT_T*               num_of_samples_with_label;
        _N_DAT_T**              index_of_sample_with_label;
        Array<_SUPV_T, _SUPV_T> index_of_label;

        _COMP_T                 entropy;

        Labels(Data& data);
        ~Labels();
    } *labels;

    Data& clear();
    Data& clear_buffer();
    Data& clear_index();
    Data& clear_labels();
    Data& load_buffer(_COMP_T* x, _SUPV_T* y,
                      _DAT_DIM_T dimension, _N_DAT_T total_samples);
    Data& load_index(_N_DAT_T num_of_samples = 0, _N_DAT_T* index = NULL,
                     _N_DAT_T* index_from = NULL, bool copy_index = true);

    /// Boost.Python interface
    Data& load_buffer(bpy::object& x, bpy::object& y);
    Data& load_index(const bpy::object& indices, _N_DAT_T* index_from = NULL);

  private:
    // Disable assignment operator
    Data& operator=(Data& some);
};

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
struct Data_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple
    getinitargs(const Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&);
};

# include "Data.inl"

# endif
