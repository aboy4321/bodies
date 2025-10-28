#include <vector>

struct Node {
    std::vector<size_t> children;
};

struct Octree {
    std::vector<Node> nodes;

    // initializing constructor
    Octree() = default;
    explicit Octree(std::size_t initial_size) : nodes(initial_size) {}
};
