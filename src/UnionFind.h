#ifndef UNIONFIND_H
#define UNIONFIND_H

#include <vector>

class UnionFind {

public:
    UnionFind();
    UnionFind(const UnionFind &other);

    bool operator<(const UnionFind &other) const;
    bool operator>(const UnionFind &other) const;

    UnionFind *find();

    int getRank() const;
    void setParent(UnionFind *parent);
    void setRank(const int &rank);

    static UnionFind *makeUnion(UnionFind *uf0, UnionFind *uf1);
    static UnionFind *makeUnion(std::vector<UnionFind *> &sets);

protected:
    int rank_;
    UnionFind *parent_;
};


inline UnionFind::UnionFind() {
    rank_ = 0;
    parent_ = this;
}

inline UnionFind::UnionFind(const UnionFind &other) {
    rank_ = other.rank_;
    parent_ = other.parent_;  // 浅拷贝，确保指针指向同一个父节点
}


inline bool UnionFind::operator<(const UnionFind &other) const {
    return rank_ < other.rank_;
}

inline bool UnionFind::operator>(const UnionFind &other) const {
    return rank_ > other.rank_;
}


inline UnionFind *UnionFind::find() {
    if (parent_ != this) {
        parent_ = parent_->find();  // 路径压缩
    }
    return parent_;
}


inline int UnionFind::getRank() const {
    return rank_;
}

inline void UnionFind::setParent(UnionFind *parent) {
    parent_ = parent;
}

inline void UnionFind::setRank(const int &rank) {
    rank_ = rank;
}


inline UnionFind *UnionFind::makeUnion(UnionFind *uf0, UnionFind *uf1) {
    UnionFind *root0 = uf0->find();
    UnionFind *root1 = uf1->find();
    
    if (root0 == root1) {
        return root0;  // 已经属于同一集合
    }

    if (root0->rank_ > root1->rank_) {
        root1->parent_ = root0;
        return root0;
    } else {
        root0->parent_ = root1;
        if (root0->rank_ == root1->rank_) {
            root1->rank_++;
        }
        return root1;
    }
}


inline UnionFind *UnionFind::makeUnion(std::vector<UnionFind *> &sets) {
    if (sets.empty()) return nullptr;
    UnionFind *result = sets[0];
    for (size_t i = 1; i < sets.size(); ++i) {
        result = makeUnion(result, sets[i]);
    }
    return result;
}

#endif // UNIONFIND_H
