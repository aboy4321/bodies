#include <vector>
#include <algorithm>
#include <queue>
#include <memory>
#include <cmath>
#include <limits>
#include <raylib.h>
#include <iostream>

// defining points in space for bounding box
struct Point {
    float x, y;
    Point(float _x, float _y) : x(_x), y(_y) {}
    Point() : x(0), y(0) {}
};

static constexpr float inf = std::numeric_limits<float>::infinity();

// defining bounding box and its properties 
struct Box {
    Point min{ inf, inf };
    Point max{ -inf, -inf };

    Point middle() const {
        return { (min.x + max.x) / 2.f, (min.y + max.y) / 2.f};
    }
    
    Box& operator |= (Point const& p) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        return *this;
    }
    
    float width() const { return max.x - min.x; }
    float height() const { return max.y - min.y; }
    
    // checking if body is contained in quadtree node/ boundary
    bool contains(const Point& p) const {
        static constexpr float epsilon = 1e-6f;
        return (p.x >= min.x - epsilon && p.x <= max.x + epsilon &&
                p.y >= min.y - epsilon && p.y <= max.y + epsilon);
    }
};

// node for Barnes-Hut quadtree
struct Node {
    bool is_leaf = true;
    float total_mass = 0.f;
    float com_x = 0.f;
    float com_y = 0.f;
    int bodyIndex = -1;

    std::unique_ptr<Node> children[4];

    Box boundary;

    Node(const Box& bounds) : boundary(bounds) {}
    Node() = default;
};


// N body sim system
class System {
    public:
        // creating components of bodies/system
        std::vector<float> x, y;
        std::vector<float> vx, vy;
        std::vector<float> ax, ay;
        std::vector<float> m;

        // adds a body to vectors and increases variables to be computed
        void add_body(float px, float py, float mvx, float mvy, float mass) {
            x.push_back(px); y.push_back(py);
            vx.push_back(mvx); vy.push_back(mvy);
            ax.push_back(0); ay.push_back(0);
            m.push_back(mass);
        }

        Box get_bounds() const {
            Box b;
            for (size_t i= 0; i < x.size(); ++i) {
                b |= {x[i], y[i]};
            }
            if (b.width() < 1.0f) { b.min.x -= 0.5f; b.max.x += 0.5f; }
            if (b.height() < 1.0f) { b.min.y -= 0.5f; b.max.y += 0.5f; }
            return b;
        }
};

class Quadtree {
    private:
        const float theta;
        std::unique_ptr<Node> root;

        int get_quadrant(Node* node, float x, float y) {
            Point center = node->boundary.middle();
            bool right = x >=center.x;
            bool top = y >= center.y;
            if (!right && top) return 0;
            if (right && top) return 1;
            if (!right && !top) return 2;
            return 3;
        }
        
        void subdivide(Node* node) {

            Point center = node->boundary.middle();

            Box b_nw = {
                {node->boundary.min.x, center.y},
                {center.x, node->boundary.max.y}
            };

            Box b_ne = {
                {center.x, center.y},
                {node->boundary.max.x, node->boundary.max.y}
            };

            Box b_sw = {
                {node->boundary.min.x, node->boundary.min.y},
                {center.x, center.y}
            };

            Box b_se = {
                {center.x, node->boundary.min.y},
                {node->boundary.max.x, center.y}
            };

            node->children[0] = std::make_unique<Node>(b_nw);
            node->children[1] = std::make_unique<Node>(b_ne);
            node->children[2] = std::make_unique<Node>(b_sw);
            node->children[3] = std::make_unique<Node>(b_se);

            node->is_leaf = false;
        }

        void insert(int bodyIndex, const System& sys) {
            Node* current = root.get();

            float bx = sys.x[bodyIndex];
            float by = sys.y[bodyIndex];
            float bm = sys.m[bodyIndex];

            while (true) {

                // updating COM before checking if node is leaf or nah
                float prevMass = current->total_mass;
                current->total_mass += bm;

                if (prevMass == 0.0f) {
                    current->com_x = bx;
                    current->com_y = by;
                } else {
                    current->com_x = (current->com_x * prevMass + bx * bm ) / current->total_mass;
                    current->com_y = (current->com_y * prevMass + by * bm ) /  current->total_mass;
                }

                if (current->is_leaf) {
                    if (current->bodyIndex == -1) {
                        current->bodyIndex = bodyIndex;
                        break;
                    }

                    int oldIdx = current->bodyIndex;
                    float oldX = sys.x[oldIdx];
                    float oldY = sys.y[oldIdx];

                    if (std::abs(bx - oldX) < 1e-9 && std::abs(by - oldY) < 1e-9) {
                        break;
                    }

                    subdivide(current);
                    current->bodyIndex =-1;

                    int prevQuad = get_quadrant(current, oldX, oldY);
                    Node* prevChild = current->children[prevQuad].get();
                    prevChild->bodyIndex = oldIdx;
                    prevChild->total_mass = sys.m[oldIdx];
                    prevChild->com_x = oldX;
                    prevChild->com_y = oldY;
                }

                int quad = get_quadrant(current, bx, by);
                current = current->children[quad].get();
            }
        } 

        void calculate_force(int idx, const System& sys, float& out_fx, float& out_fy) {
            static constexpr float G = 100.0f;
            static constexpr float softening = 1e-1f;

            out_fx = 0.0f;
            out_fy = 0.0f;

            static std::vector<Node*> stack;
            stack.clear();
            stack.push_back(root.get());

            while (!stack.empty()) {
                Node* node = stack.back();

                stack.pop_back();

                if (node->total_mass == 0) continue;

                float dx = node->com_x - sys.x[idx];
                float dy = node->com_y - sys.y[idx];
                float distSqr = dx * dx + dy * dy + (softening * softening);

                float dist = std::sqrt(distSqr);

                float s = std::max(node->boundary.width(), node->boundary.height());

                if (node->is_leaf || (s / dist) < theta) {
                    if (node->bodyIndex == idx) continue;
    
                    float force = (G * sys.m[idx] * node->total_mass) / distSqr;

                    out_fx += force * (dx / dist);
                    out_fy += force * (dy / dist);
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (node->children[i] && node->children[i]->total_mass > 0) {
                            stack.push_back(node->children[i].get());
                        }
                    }
                }
            }                        
        }

        void draw_recursive(Node* node) {
            if (!node) return;

            Rectangle rect = { 
                node->boundary.min.x, 
                node->boundary.min.y, 
                node->boundary.width(), 
                node->boundary.height() 
            };

            DrawRectangleLinesEx(rect, 1.0f, Fade(GRAY, 0.15f));

            if (!node->is_leaf) {
                for (int i = 0; i < 4; ++i) {
                    if (node->children[i]) {
                        draw_recursive(node->children[i].get());
                    }
                }
            }
        }

    public:
        Quadtree(const Box& initialBounds, float theta_val) 
            : theta(theta_val) {
                root = std::make_unique<Node>(initialBounds);
        }

        void build(const System& sys) {
            Box current_bounds = sys.get_bounds();
            root = std::make_unique<Node>(current_bounds);

            for (size_t i = 0; i < sys.x.size(); ++i) {
                if (sys.m[i] > 0) insert(i, sys);
            }
        }

        void calculate_forces(System& sys) {
            for (size_t i = 0; i < sys.x.size(); ++i) {
                float fx = 0, fy = 0;
                calculate_force(i, sys, fx, fy);
                sys.ax[i] = fx / sys.m[i];
                sys.ay[i] = fy / sys.m[i];
            }
        }

        void draw() {
            draw_recursive(root.get());
    }
};

int main() {
    const int screenWidth = 1000;
    const int screenHeight = 1000;

    InitWindow(screenWidth, screenHeight, "n bod sim, bhutt ver");

    if (!IsWindowReady()) {
        return 1; 
    }

    SetTargetFPS(144);

    System sys;
    
    float central_mass = 10000.0f;
    sys.add_body(screenWidth/2.0f, screenHeight/2.0f, 0, 0, central_mass);
    
    int num_particles = 1000;
    for(int i = 0; i < num_particles; ++i) {
        float r = 100 + (rand() % 300);  
        float angle = (rand() % 360) * 3.14159f / 180.0f;
        
        float orbital_velocity = sqrtf(100.0f * central_mass / r);
        
        float px = screenWidth/2.0f + cosf(angle) * r;
        float py = screenHeight/2.0f + sinf(angle) * r;
        
        float vx = -sinf(angle) * orbital_velocity;
        float vy = cosf(angle) * orbital_velocity;
        
        sys.add_body(px, py, vx, vy, 1.0f);
    }

    float dt = 0.01f; 
    while (!WindowShouldClose()) {
        Quadtree qt(sys.get_bounds(), 0.25f);
        qt.build(sys);
        
        qt.calculate_forces(sys);
        
        for (size_t i = 0; i < sys.x.size(); ++i) {
            sys.vx[i] += sys.ax[i] * dt;
            sys.vy[i] += sys.ay[i] * dt;
            
            sys.x[i] += sys.vx[i] * dt;
            sys.y[i] += sys.vy[i] * dt;
        }

        BeginDrawing();
        ClearBackground(BLACK);
        
        qt.draw();

        for (size_t i = 0; i < sys.x.size(); ++i) {
            float radius = (i == 0) ? 8.0f : 2.0f;
            Color color = (i == 0) ? YELLOW : WHITE;
            DrawCircle((int)sys.x[i], (int)sys.y[i], radius, color);
        }

        DrawFPS(10, 10);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
