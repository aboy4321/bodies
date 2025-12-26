#include <memory>
#include <vector>
#include <math.h>
using namespace std;

struct Node {
    float topLeft_x, topLeft_y;
    float botRight_x, botRight_y;
    float com_x, com_y;
    float total_mass;
    bool is_leaf;
    int index;
    Node* nw, *ne, *sw, *se;
};

class BHuttTree {
    private:
    vector<Node> node_pool;
    int next_free;
    Node* root;

    const vector<float>& _x , _y, _m;

    public:
    BHuttTree(vector<float>& x, vector<float>& y,
              vector<float>& m)
              : _x(x), _y(y), _m(m) {}

    int get_q(int i, Node* node) {
        if ((node->topLeft_x + node->botRight_x) / 2 >= _x[i]) {
            if ((node->topLeft_y + node->botRight_y) / 2 <= _y[i]) {
                return 2;
            } else return 3;
        } else {
            if((node->topLeft_y + node->botRight_y)/ 2 <= _y[i]) {
                return 1;
            } else return 4;
        }
    }

    void insert(int i, Node* node) {

        if (node == nullptr) return;
        if (!in_bound(i, node)) return;

        float x = _x[i];
        float y = _y[i];
        float m = _m[i];

        if (node->total_mass == 0) {
            node->com_x = x;
            node->com_y = y;
            node->total_mass = m;
        } else {
            float new_mass = node->total_mass + m;
            node->com_x = (node->com_x * node->total_mass + x * m) / new_mass;
            node->com_y = (node->com_y * node->total_mass + y * m) / new_mass;
            node->total_mass = new_mass;
        }
        
        if (node->is_leaf && node->index == -1) {
            node->index = i;
            return;
        }

        if (node->is_leaf && node->index != -1) {
            node->is_leaf = false;

            if (abs(_x[node->index] - x) < 1e-5 && abs(_y[node->index] - y) < 1e-5) {
                 // Edge case: Just skip inserting the duplicate or handle via list
                 // For simple Barnes-Hut, we might just return to avoid stack overflow
                 return; 
            }

            node->is_leaf = false;
            
            // Initialize children (Helper lambda or function recommended to reduce code duplication)
            auto spawn_child = [&](float tx, float ty, float bx, float by) {
                Node* c = &node_pool[next_free++];
                c->topLeft_x = tx; c->topLeft_y = ty;
                c->botRight_x = bx; c->botRight_y = by;
                c->is_leaf = true; 
                c->index = -1;
                c->total_mass = 0; // FIXED: Initialize mass
                c->com_x = 0; c->com_y = 0;
                return c;
            };
            
            float mid_x = (node->topLeft_x + node->botRight_x) / 2;
            float mid_y = (node->topLeft_y + node->botRight_y) / 2;

            if (!node->nw) node->nw = spawn_child(node->topLeft_x, node->topLeft_y, mid_x, mid_y);
            if (!node->ne) node->ne = spawn_child(mid_x, node->topLeft_y, node->botRight_x, mid_y);
            if (!node->sw) node->sw = spawn_child(node->topLeft_x, mid_y, mid_x, node->botRight_y);
            if (!node->se) node->se = spawn_child(mid_x, mid_y, node->botRight_x, node->botRight_y);

            // Push the OLD particle down
            int _prev = node->index;
            node->index = -1;
            int old_q = get_q(_prev, node);
            if (old_q == 1) insert(_prev, node->ne);
            else if (old_q == 2) insert(_prev, node->nw);
            else if (old_q == 3) insert(_prev, node->sw);
            else insert(_prev, node->se);
        }

        // 4. Push the NEW particle (i) down (FIXED: This was missing)
        int q = get_q(i, node);
        if (q == 1) insert(i, node->ne);
        else if (q == 2) insert(i, node->nw);
        else if (q == 3) insert(i, node->sw);
        else insert(i, node->se);
    }

    bool in_bound(int i, Node* node) {
        return _x[i] >= node->topLeft_x && _x[i] <= node->botRight_x
        && _y[i] >= node->topLeft_y && _y[i] <= node->botRight_y;
    }
};

class System {
    public:
    size_t num_particles;

    vector<float> x, y, vx, vy, ax, ay, m;

    const float epsilon = 0.5;
    const float dt = 1.0;
    const float theta = 0.5;
    const float G = 6.6743e-11;

    System(int n) : num_particles(n) {
        x.resize(n);
        y.resize(n);
        vx.resize(n);
        vy.resize(n);
        ax.resize(n);
        ay.resize(n);
        m.resize(n, 1.0f);
    }
};

int main() {
    return 0;
}
