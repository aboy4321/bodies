#include <memory>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <stack>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// creating node struct for octree
struct Node {
    float center_x, center_y, center_z;
    float size;
    float total_mass;
    float com_x, com_y, com_z;
    bool is_leaf;
    std::vector<int> particles;
    std::array<std::unique_ptr<Node>, 8> children;

    // creating constructor to initialize node(s)
    Node(float cx, float cy, float cz, float s)
        : center_x(cx), center_y(cy), center_z(cz), size(s),
        total_mass(0.0f), com_x(0.0f), com_y(0.0f), com_z(0.0f),
        is_leaf(true) {}
};

class System {
    private:
    std::stack<Node> root_node;

    public:
    int num_particles;
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> acc_x, acc_y, acc_z;
    std::vector<float> masses;

    float G;
    float epsilon;

    // supposed to determine accuracy of simulation
    float theta;

    // to utilize cache better
    System(const System &) = default;
    System(System &&) = default;
    System &operator=(const System &) = default;
    System &operator=(System &&) = default;

    // resizing to accomodate for N value
    System(int N, float gravity_const) : num_particles(N), G(gravity_const), epsilon(1e-2f), theta(1.0f) {
        pos_x.resize(N);
        pos_y.resize(N);
        pos_z.resize(N);

        vel_x.resize(N);
        vel_y.resize(N);
        vel_z.resize(N);

        acc_x.resize(N);
        acc_y.resize(N);
        acc_z.resize(N);

        masses.resize(N, 1.0f);
    };

    // computing total mass M for COM
    inline float compute_total_mass() const {
        float M = 0.0;
        for (int i = 0; i < num_particles; ++i) {
            M += masses[i];
        }
        return M;
    }

    inline float compute_pos_cm(const std::vector<float>& pos) const {
        float pos_cm = 0.0f;
        float M = compute_total_mass();
        for (int i = 0; i < num_particles; ++i) {
            pos_cm += masses[i] * pos[i];
        }

        pos_cm /= M;
        return pos_cm;
    }

    inline float compute_v_cm(const std::vector<float>& vel) const {
        float v_cm = 0.0;
        float M = compute_total_mass();
        for (int i = 0; i < num_particles; ++i) {
            v_cm += masses[i] * vel[i];
        }

        v_cm /= M;
        return v_cm;
    }

    // setting position of particles to the center of mass (for now)
    System& shift_to_COM() {
        float x_cm = compute_pos_cm(pos_x);
        float y_cm = compute_pos_cm(pos_y);
        float z_cm = compute_pos_cm(pos_z);

        float vel_x_cm = compute_v_cm(vel_x);
        float vel_y_cm = compute_v_cm(vel_y);
        float vel_z_cm = compute_v_cm(vel_z);

        for (int i = 0; i < num_particles; ++i) {
            pos_x[i] -= x_cm;
            pos_y[i] -= y_cm;
            pos_z[i] -= z_cm;

            vel_x[i] -= vel_x_cm;
            vel_y[i] -= vel_y_cm;
            vel_z[i] -= vel_z_cm;
        }
        return *this;
    }

    // updates pos and velocity per frame to simulate movement for 4 particles at a time, slightly more efficient...
    void update(float dt) {
            int i = 0;
            for (; i <= num_particles - 4; i += 4) {
                vel_x[i]   += acc_x[i]   * dt;
                vel_x[i+1] += acc_x[i+1] * dt;
                vel_x[i+2] += acc_x[i+2] * dt;
                vel_x[i+3] += acc_x[i+3] * dt;

                vel_y[i]   += acc_y[i]   * dt;
                vel_y[i+1] += acc_y[i+1] * dt;
                vel_y[i+2] += acc_y[i+2] * dt;
                vel_y[i+3] += acc_y[i+3] * dt;

                vel_z[i]   += acc_z[i]   * dt;
                vel_z[i+1] += acc_z[i+1] * dt;
                vel_z[i+2] += acc_z[i+2] * dt;
                vel_z[i+3] += acc_z[i+3] * dt;

                pos_x[i]   += vel_x[i]   * dt;
                pos_x[i+1] += vel_x[i+1] * dt;
                pos_x[i+2] += vel_x[i+2] * dt;
                pos_x[i+3] += vel_x[i+3] * dt;

                pos_y[i]   += vel_y[i]   * dt;
                pos_y[i+1] += vel_y[i+1] * dt;
                pos_y[i+2] += vel_y[i+2] * dt;
                pos_y[i+3] += vel_y[i+3] * dt;

                pos_z[i]   += vel_z[i]   * dt;
                pos_z[i+1] += vel_z[i+1] * dt;
                pos_z[i+2] += vel_z[i+2] * dt;
                pos_z[i+3] += vel_z[i+3] * dt;
            }

            // handles remaining N % 4 particles
            for (; i < num_particles; ++i) {
                vel_x[i] += acc_x[i] * dt;
                vel_y[i] += acc_y[i] * dt;
                vel_z[i] += acc_z[i] * dt;

                pos_x[i] += vel_x[i] * dt;
                pos_y[i] += vel_y[i] * dt;
                pos_z[i] += vel_z[i] * dt;
            }
    }

    // refresh acceleration for each time step as to not sum up the forces TOO much...
    void reset_acceleration() {
        std::fill(acc_x.begin(), acc_x.end(), 0.0f);
        std::fill(acc_y.begin(), acc_y.end(), 0.0f);
        std::fill(acc_z.begin(), acc_z.end(), 0.0f);
    }

    private:

    // getting octant of particle in node
    int get_octant(const Node* node, float x, float y, float z) const {
        int octant = 0;
        if (x > node->center_x) octant |= 1;
        if (y > node->center_y) octant |= 2;
        if (z > node->center_z) octant |= 4;
        return octant;
    }

    // adding particle into node in octant
    void insert_particle(Node* node, int particle_id ) {
        if (node-> particles.empty() && node->is_leaf) {
            node->particles.push_back(particle_id);
            return;
        }

        if (node->is_leaf && node->particles.size() ==  1) {
            int existing_id = node->particles[0];
            node->particles.clear();
            node->is_leaf = false;
            insert_particle(node, existing_id);
            insert_particle(node, particle_id);
            return;
        }

        // recursively insert particle into child nodes
        if (node->is_leaf) {
            node->is_leaf = false;
            auto temp_particles = std::move(node->particles);
            for (int pid : temp_particles) {
                insert_particle(node, pid);
            }

            insert_particle(node, particle_id);
            return;
        }

        int octant = get_octant(node, pos_x[particle_id], pos_y[particle_id], pos_z[particle_id]);

        float child_size = node->size * 0.5f;
        float offset = child_size * 0.5f;

        float child_centers[8][3] = {
            {node->center_x - offset, node->center_y - offset, node->center_z - offset}, // 0: left-bottom-back
            {node->center_x + offset, node->center_y - offset, node->center_z - offset}, // 1: right-bottom-back
            {node->center_x - offset, node->center_y + offset, node->center_z - offset}, // 2: left-top-back
            {node->center_x + offset, node->center_y + offset, node->center_z - offset}, // 3: right-top-back
            {node->center_x - offset, node->center_y - offset, node->center_z + offset}, // 4: left-bottom-front
            {node->center_x + offset, node->center_y - offset, node->center_z + offset}, // 5: right-bottom-front
            {node->center_x - offset, node->center_y + offset, node->center_z + offset}, // 6: left-top-front
            {node->center_x + offset, node->center_y + offset, node->center_z + offset}  // 7: right-top-front
        };

        if (!node->children[octant]) {
            node->children[octant] = std::make_unique<Node>(
                child_centers[octant][0],
                child_centers[octant][1],
                child_centers[octant][2],
                child_size
            );
        }

        insert_particle(node->children[octant].get(), particle_id);
    }

    void compute_mass_properties(Node* node) {
        if (node->is_leaf) {
            node->total_mass = 0.0f;
            node->com_x = 0.0f;
            node->com_y = 0.0f;
            node->com_z = 0.0f;

            for (int pid : node->particles) {
                node->total_mass += masses[pid];
                node->com_x += pos_x[pid] * masses[pid];
                node->com_y += pos_y[pid] * masses[pid];
                node->com_z += pos_z[pid] * masses[pid];
            }

            if (node->total_mass > 0.0f) {
                node->com_x /= node->total_mass;
                node->com_y /= node->total_mass;
                node->com_z /= node->total_mass;
            }
        } else {
            node->total_mass = 0.0f;
            node->com_x = 0.0f;
            node->com_y = 0.0f;
            node->com_z = 0.0f;

            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    compute_mass_properties(node->children[i].get());
                    node->total_mass += node->children[i]->total_mass;
                    node->com_x += node->children[i]->com_x * node->children[i]->total_mass;
                    node->com_y += node->children[i]->com_y * node->children[i]->total_mass;
                    node->com_z += node->children[i]->com_z * node->children[i]->total_mass;
                }
            }

            if (node->total_mass > 0.0f) {
                node->com_x /= node->total_mass;
                node->com_y /= node->total_mass;
                node->com_z /= node->total_mass;
            }
        }
    }

    void compute_node_force(const Node* node, int particle_id, float& acc_x, float& acc_y, float& acc_z) {
        if (particle_id <0 || particle_id >= num_particles) return;

        if (!node) return;

        if (node->total_mass == 0.0f) return;

        float dx = node->com_x - pos_x[particle_id];
        float dy = node->com_y - pos_y[particle_id];
        float dz = node->com_z - pos_z[particle_id];

        float r2 = dx*dx + dy*dy + dz*dz + epsilon;
        float r = sqrtf(r2);

        if (node->size / r < theta || node->is_leaf) {
            float rinv = 1.0f / std::sqrt(r2);
            float rinv3 = rinv * rinv * rinv;
            float force = G * rinv3;  // G / r^3

            acc_x += force * dx * node->total_mass;
            acc_y += force * dy * node->total_mass;
            acc_z += force * dz * node->total_mass;
        }

        else {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    compute_node_force(node->children[i].get(), particle_id, acc_x, acc_y, acc_z);
                }
            }
        }
    }

    public:

    void compute_gravity_barnes_hut() {
        reset_acceleration();

        float min_x = pos_x[0], max_x = pos_x[0];
        float min_y = pos_y[0], max_y = pos_y[0];
        float min_z = pos_z[0], max_z = pos_z[0];

        for (int i = 1; i < num_particles; ++i) {
            min_x = std::min(min_x, pos_x[i]); max_x = std::max(max_x, pos_x[i]);
            min_y = std::min(min_y, pos_y[i]); max_y = std::max(max_y, pos_y[i]);
            min_z = std::min(min_z, pos_z[i]); max_z = std::max(max_z, pos_z[i]);
        }

        float padding = 0.1f;
        min_x -= padding; max_x += padding;
        min_y -= padding; max_y += padding;
        min_z -= padding; max_z += padding;

        float center_x = (min_x + max_x) * 0.5f;
        float center_y = (min_y + max_y) * 0.5f;
        float center_z = (min_z + max_z) * 0.5f;
        float size = std::max({max_x - min_x, max_y - min_y, max_z - min_z}) * 0.5f;

        auto root = std::make_unique<Node>(center_x, center_y, center_z, size);

        for (int i = 0; i < num_particles; ++i) {
            insert_particle(root.get(), i);
        }

        compute_mass_properties(root.get());

        for (int i = 0; i < num_particles; ++i) {
            float acc_x_i = 0.0f, acc_y_i = 0.0f, acc_z_i = 0.0f;
            compute_node_force(root.get(), i, acc_x_i, acc_y_i, acc_z_i);
            acc_x[i] = acc_x_i;
            acc_y[i] = acc_y_i;
            acc_z[i] = acc_z_i;
        }
    }
};

// not my OpenGL code...
GLFWwindow* window = nullptr;

void create_window() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    window = glfwCreateWindow(1920, 1080, "N-Body Sim.", NULL, NULL);

    glfwMakeContextCurrent(window);

    glewExperimental = true;
    if(glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return;
    }
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-5, 5, -5, 5, -5, 5);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void render(System& sys) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < sys.num_particles; ++i) {
            float dist = sqrtf(sys.pos_x[i]*sys.pos_x[i] +
                              sys.pos_y[i]*sys.pos_y[i] +
                              sys.pos_z[i]*sys.pos_z[i]);

            float speed = sqrtf(sys.vel_x[i]*sys.vel_x[i] +
                               sys.vel_y[i]*sys.vel_y[i] +
                               sys.vel_z[i]*sys.vel_z[i]);

            // nebula theme gradients w/ respect to speed
            float r = 0.4 + 0.4f * sinf(dist * 0.4f) + 0.2f * fminf(speed * 2.0f, 1.0f);
            float g = 0.1f + 0.2f * sinf(dist * 0.3f + 1.0f) + 0.1f * fminf(speed * 1.5f, 1.0f);
            float b = 0.8f + 0.5f * sinf(dist * 0.5f + 2.0f) + 0.2f * fminf(speed * 2.0f, 1.0f);

            // restricts to valid color range
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);

            glColor3f(r, g, b);
            glVertex3f(sys.pos_x[i], sys.pos_y[i], sys.pos_z[i]);
        }
    glEnd();
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    create_window();

    const int N = 1000;
    System sys(N, 0.001);

    for (int i = 0; i < sys.num_particles; ++i) {
        if (i < sys.num_particles / 2) {
            // First galaxy
            float angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
            float radius = powf((float)rand() / RAND_MAX, 2.0f) * 2.0f;

            sys.pos_x[i] = radius * cosf(angle) - 2.0f;  // Offset from center
            sys.pos_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
            sys.pos_z[i] = radius * sinf(angle);

            float speed = 1.0f / (1.0f + radius);
            sys.vel_x[i] = -speed * sinf(angle) + 0.3f;  // Moving right
            sys.vel_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            sys.vel_z[i] = speed * cosf(angle);
        } else {
            // Second galaxy
            float angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
            float radius = powf((float)rand() / RAND_MAX, 2.0f) * 2.0f;

            sys.pos_x[i] = radius * cosf(angle) + 2.0f;  // Offset from center
            sys.pos_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
            sys.pos_z[i] = radius * sinf(angle);

            float speed = 1.0f / (1.0f + radius);
            sys.vel_x[i] = -speed * sinf(angle) - 0.3f;  // Moving left
            sys.vel_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            sys.vel_z[i] = speed * cosf(angle);
        }
        sys.masses[i] = 0.5f + (float)rand() / RAND_MAX * 2.0f;
    }

    sys.shift_to_COM();
    double last_time = glfwGetTime();
    int frame_count = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        sys.compute_gravity_barnes_hut();
        sys.update(0.001f);

        render(sys);
        glfwSwapBuffers(window);
        frame_count++;
        double current_time = glfwGetTime();
        if (current_time - last_time >= 1.0) {
            double fps = frame_count / (current_time - last_time);
            printf("frames-per-second: %.1f\n", fps);
            frame_count = 0;
            last_time = current_time;
        }
    }

    glfwTerminate();
    return 0;
}
