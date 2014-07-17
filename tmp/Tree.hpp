# ifndef _TREE_HPP
# define _TREE_HPP

# include <iostream>
# include <fstream>
# include <iomanip>
# include <cstring>
# include <queue>
# include "my_lib.hpp"

/// A basic tree structure.
/// 
/// @tparam _DATA   Type of data in every tree node.
/// @warning        _DATA must not be a pointer.
template<class _DATA>
class Tree {
public:
    Tree(): root_(NULL) {}
    virtual ~Tree();

    virtual Tree& ostream_this(std::ostream& out);
    virtual Tree& save_this(const char* dir);
    virtual Tree& read_this(const char* dir);

    /// A basic tree node structure.
    /// 
    class TreeNode {
    public:
        TreeNode(_DATA*       _data     = NULL,
                 TreeNode*    _parent   = NULL,
                 unsigned int _n_branch = 2);
        TreeNode(_DATA&       _data,
                 TreeNode*    _parent   = NULL,
                 unsigned int _n_branch = 2);
        virtual ~TreeNode();

        TreeNode&    content(_DATA& _data);
        TreeNode&    pcontent(_DATA* _data);
        TreeNode&    clear();
        /// Make current node ready to branch into several nodes.
        /// (make sure this operation be done before attaching any child!)
        TreeNode&    branches(unsigned int _n_branch);
        TreeNode&    attach_child();
        TreeNode&    attach_child(TreeNode* node);
        unsigned int num_of_children()      const { return n_child; }
        unsigned int num_of_branches()      const { return n_branch; }
        TreeNode&    child(unsigned int i)  const { return *child_[i]; }
        TreeNode*    pchild(unsigned int i) const { return child_[i]; }
        TreeNode**   children()             const { return child_; }
        bool         has_child()            const { return (bool)n_child; }
        TreeNode&    parent()               const { return *parent_; }
        TreeNode*    pparent()              const { return parent_; }
        bool         has_parent()           const { return (bool)parent_; }
        _DATA&       content()              const { return *data; }
        _DATA*       pcontent()             const { return data; }
        unsigned int depth()                const { return depth_; }
    private:
        _DATA*       data;     ///< The content of a node. (internal)
        unsigned int depth_;
        unsigned int n_branch; ///< Max number of children.
                               ///< (Reallocation may be needed if it is modified)
        unsigned int n_child;  ///< Current number of children
        TreeNode*    parent_;
        TreeNode**   child_;
    };

    TreeNode& root() const { return *root_; }

    /// The iterator class is designed to traverse through each node of the tree.
    /// 
    class iterator {
    public:
        iterator(TreeNode* node);
        ~iterator() {}
        iterator& operator++();
        iterator  operator++(int);
        TreeNode& operator*()  const { return *pointer; }
        TreeNode* operator&()  const { return pointer; }
        TreeNode* operator->() const { return pointer; }
        bool      operator==(const iterator& some) const
            { return pointer == some.pointer; }
        bool      operator!=(const iterator& some) const
            { return pointer != some.pointer; }
    private:
        TreeNode* pointer;
        std::queue<TreeNode*> breadth_first_traverse;
    };

    iterator begin() const { return iterator(root_); }
    iterator end()   const { return iterator(NULL); }

protected:
    TreeNode* root_;
}; // class Tree

template<class _DATA> inline 
std::ostream& operator<<(std::ostream& out, Tree<_DATA>& tree) {
    tree.ostream_this(out);
    return out;
}

template<class _DATA> inline 
Tree<_DATA>::~Tree() {
    if (!root_) return;
    std::queue<TreeNode*> breadth_first_traverse;
    breadth_first_traverse.push(root_);
    while (!breadth_first_traverse.empty()) {
        TreeNode*    cur      = breadth_first_traverse.front();
        unsigned int cur_n_ch = cur->num_of_children();
        if (cur_n_ch) {
            TreeNode** cur_ch = cur->children();
            for (unsigned int i = 0; i < cur_n_ch; i++) {
                breadth_first_traverse.push(cur_ch[i]);
            }
        }
        breadth_first_traverse.pop();
        delete cur;
    }
}

template<class _DATA>
Tree<_DATA>& Tree<_DATA>::
ostream_this(std::ostream& out) {
    out << "********************************";
    iterator it_end = end();
    for (iterator i = begin(); i != it_end; ++i) {
        out << "\n****MyTreeNode(" << &i << ", depth = " << i->depth() 
            << ", parent = " << i->pparent() << ")\n";
        out << i->content();
        out << "\n****children =";
        unsigned int n_child = i->num_of_children();
        for (unsigned int j = 0; j < n_child; ++j) {
            out << " " << i->pchild(j);
        }
        out << "\n****************";
    }
    return *this;
}

template<class _DATA>
Tree<_DATA>& Tree<_DATA>::
save_this(const char* dir) {
    char path[SIZEOF_PATH], * filename;
    std::strcpy(path, dir);
    std::size_t dir_len = std::strlen(path);
    std::size_t file_len;
    if (dir_len > 0 && (path[dir_len - 1] != '/' && path[dir_len - 1] != '\\'))
        path[dir_len++] = '/';
    filename = path + dir_len;
    file_len = SIZEOF_PATH - dir_len;
    std::strcpy(filename, "Tree");
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << path << std::endl;
        return *this;
    }
    unsigned long node_num = 0;
    file << std::setw(16) << std::setfill('0') << node_num << ".node"
         << "    # root" << std::endl;
    file.close();
    std::queue<TreeNode*>     node_pointer;
    std::queue<unsigned long> node_name;
    std::queue<unsigned long> node_parent_name;
    node_pointer.push(root_);
    node_name.push(node_num);
    node_parent_name.push(0);
    while (!node_pointer.empty()) {
        TreeNode*     node        = node_pointer.front();
        unsigned long name        = node_name.front();
        unsigned long parent_name = node_parent_name.front();
        std::snprintf(filename, file_len, "%016lu.node", name);
        file.open(path);
        if (!file.is_open()) {
            std::cerr << "\nFailed opening file: " << path << std::endl;
            return *this;
        }
        file << std::setw(16) << std::setfill('0') << parent_name
             << ".node    # parent\n"
             << node->depth() << "    # depth\n";
        unsigned int n_child = node->num_of_children();
        file << n_child << "    # number of children\n";
        if (n_child) {
            TreeNode** child = node->children();
            for (unsigned int i = 0; i < n_child; ++i) {
                node_pointer.push(child[i]);
                node_name.push(++node_num);
                node_parent_name.push(name);
                file << std::setw(16) << std::setfill('0') << node_num
                     << ".node    # child " << i << "\n";
            }
        }
        file << node->content();
        file.close();
        node_pointer.pop();
        node_name.pop();
        node_parent_name.pop();
    }
    return *this;
}

template<class _DATA>
Tree<_DATA>& Tree<_DATA>::
read_this(const char* dir) {
    char path[SIZEOF_PATH], * filename, line_str[SIZEOF_PATH], * filename_str;
    std::strcpy(path, dir);
    std::size_t dir_len = std::strlen(path);
    std::size_t file_len;
    if (dir_len > 0 && (path[dir_len - 1] != '/' && path[dir_len - 1] != '\\'))
        path[dir_len++] = '/';
    filename = path + dir_len;
    file_len = SIZEOF_PATH - dir_len;
    std::strcpy(filename, "Tree");
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << path << std::endl;
        return *this;
    }
    while(file.getline(line_str, SIZEOF_LINE) && !line_str[0]);
    if (!line_str[0]) {
        std::cerr << "\nFailed finding file for root." << std::endl;
        return *this;
    }
    file.close();
    filename_str = strtostr(line_str);
    std::strcpy(filename, filename_str);
    file.open(path);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << path << std::endl;
        return *this;
    }
    unsigned long node_num;
    _DATA* data;
    std::queue<unsigned long> node_name;
    std::queue<TreeNode*>     node_parent_pointer;
    file.getline(line_str, SIZEOF_LINE);
    file.getline(line_str, SIZEOF_LINE);
    file.getline(line_str, SIZEOF_LINE);
    unsigned int n_child = strto<unsigned int>(line_str, 10);
    root_ = new TreeNode(NULL, NULL, n_child);
    for (unsigned int i = 0; i < n_child; ++i) {
        file.getline(line_str, SIZEOF_LINE);
        node_num = strto<unsigned long>(line_str, 10);
        node_name.push(node_num);
        node_parent_pointer.push(root_);
    }
    data = new _DATA();
    file >> *data;
    file.close();
    root_->pcontent(data);
    while (!node_name.empty()) {
        unsigned long name   = node_name.front();
        TreeNode*     parent = node_parent_pointer.front();
        std::snprintf(filename, file_len, "%016lu.node", name);
        file.open(path);
        if (!file.is_open()) {
            std::cerr << "\nFailed opening file: " << path << std::endl;
            node_name.pop();
            node_parent_pointer.pop();
            continue;
        }
        file.getline(line_str, SIZEOF_LINE);
        file.getline(line_str, SIZEOF_LINE);
        file.getline(line_str, SIZEOF_LINE);
        unsigned int n_child = strto<unsigned int>(line_str, 10);
        TreeNode* node = new TreeNode(NULL, parent, n_child);
        parent->attach_child(node);
        for (unsigned int i = 0; i < n_child; ++i) {
            file.getline(line_str, SIZEOF_LINE);
            node_num = strto<unsigned long>(line_str, 10);
            node_name.push(node_num);
            node_parent_pointer.push(node);
        }
        data = new _DATA();
        file >> *data;
        file.close();
        node->pcontent(data);
        node_name.pop();
        node_parent_pointer.pop();
    }
    return *this;
}

template<class _DATA> inline 
Tree<_DATA>::iterator::iterator(TreeNode* node): pointer(node) {
    if (pointer) breadth_first_traverse.push(pointer);
}

template<class _DATA> inline 
typename Tree<_DATA>::iterator& 
Tree<_DATA>::iterator::
operator++() {
    unsigned int n_child = pointer->num_of_children();
    if (n_child) {
        TreeNode** child = pointer->children();
        for (unsigned int i = 0; i < n_child; ++i) {
            breadth_first_traverse.push(child[i]);
        }
    }
    breadth_first_traverse.pop();
    if (breadth_first_traverse.empty()) pointer = NULL;
    else pointer = breadth_first_traverse.front();
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::iterator 
Tree<_DATA>::iterator::
operator++(int) {
    iterator temp(*this);
    unsigned int n_child = pointer->num_of_children();
    if (n_child) {
        TreeNode** child = pointer->children();
        for (unsigned int i = 0; i < n_child; ++i) {
            breadth_first_traverse.push(child[i]);
        }
    }
    breadth_first_traverse.pop();
    if (breadth_first_traverse.empty()) pointer = NULL;
    else pointer = breadth_first_traverse.front();
    return temp;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::TreeNode(_DATA&       _data,
                                TreeNode*    _parent,
                                unsigned int _n_branch):
                       n_branch(_n_branch),
                       n_child(0),
                       parent_(_parent),
                       child_(NULL) {
    data = new _DATA(_data);
    if (parent_) depth_ = parent_->depth_ + 1;
    else depth_ = 0;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::TreeNode(_DATA*       _data,
                                TreeNode*    _parent,
                                unsigned int _n_branch):
                       data(_data),
                       n_branch(_n_branch),
                       n_child(0),
                       parent_(_parent),
                       child_(NULL) {
    if (parent_) depth_ = parent_->depth_ + 1;
    else depth_ = 0;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::~TreeNode() {
    delete data;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
content(_DATA& _data) {
    if (data) *data = _data;
    else data = new _DATA(_data);
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
pcontent(_DATA* _data) {
    if (data && data != _data) delete data;
    data = _data;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
clear() {
    delete data;
    data = NULL;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
branches(unsigned int _n_branch) {
    if (_n_branch <= n_branch) {
        return *this;
    }
    if (child_) {
        TreeNode** new_child_ = new TreeNode*[_n_branch];
        for (unsigned int i = 0; i < n_child; ++i) {
            new_child_[i] = child_[i];
        }
        delete[] child_;
        child_ = new_child_;
    }
    n_branch = _n_branch;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
attach_child() {
    if (n_child >= n_branch) {
        std::cerr << "Error attaching a child: Already full branched. "
                  << "Please re-branch the node to eliminate the error."
                  << std::endl;
        exit(1);
    }
    if (!child_) child_ = new TreeNode*[n_branch];
    TreeNode* node = new TreeNode(NULL, this, n_branch);
    child_[n_child++] = node;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
attach_child(TreeNode* node) {
    if (n_child >= n_branch) {
        std::cerr << "Error attaching a child: Already full branched. "
                  << "Please re-branch the node to eliminate the error."
                  << std::endl;
        exit(1);
    }
    if (!child_) child_ = new TreeNode*[n_branch];
    node->parent_     = this;
    node->depth_      = depth_ + 1;
    child_[n_child++] = node;
    return *this;
}

# endif
