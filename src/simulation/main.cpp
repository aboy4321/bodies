#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cwchar>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <ctime>
#include <memory>
#include <algorithm>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct Node {
    float center_x, center_y, center_z;
    float size;
    float total_mass;
    float com_x, com_y, com_z;
    bool is_leaf;
    std::vector<int> particles;
    std::array<std::unique_ptr<Node>, 8> children;

    Node(float cx, float cy, float cz, float s)
        : center_x(cx), center_y(cy), center_z(cz), size(s),
        total_mass(0.0f), com_x(0.0f), com_y(0.0f), com_z(0.0f),
        is_leaf(true) {}
};

class System {
    public:
    int num_particles;
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> acc_x, acc_y, acc_z;
    std::vector<float> masses;

    float G;
    float epsilon;
    float theta;

    // to utilize cache better
    System(const System &) = default;
    System(System &&) = default;
    System &operator=(const System &) = default;
    System &operator=(System &&) = default;

    // resizing to accomodate for N value
    System(int N, float gravity_const) : num_particles(N), G(gravity_const), epsilon(1e-4f) {
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

    // using this instead of standard inverse root due to its speed
    inline float quake_root(float n) {
        union {
            float f;
            uint32_t i;
        } conv = {n};

        conv.i = 0x5F375A86 - (conv.i >> 1);
        conv.f *= 1.5f - (n * 0.5f * conv.f * conv.f);
        return conv.f;
    }

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

    // main gravity function currently O(n^2) time
    void compute_gravity() {
        reset_acceleration();

        // allowing parallelization for performance
        #pragma omp parallel
        {
            // each thread has its own acceleration
            std::vector<float> acc_x_private(num_particles, 0.0f);
            std::vector<float> acc_y_private(num_particles, 0.0f);
            std::vector<float> acc_z_private(num_particles, 0.0f);

            #pragma omp for schedule(dynamic, 32) nowait
            for (int i = 0; i < num_particles; ++i) {
                // accessing local variables > accessing vector elements, thus we have the code below:
                float pos_x_i = pos_x[i];
                float pos_y_i = pos_y[i];
                float pos_z_i = pos_z[i];
                float mass_i = masses[i];

                for (int j = i + 1; j < num_particles; ++j) {
                    float dx = pos_x[j] - pos_x_i;
                    float dy = pos_y[j] - pos_y_i;
                    float dz = pos_z[j] - pos_z_i;

                    float r2 = dx * dx + dy * dy + dz * dz + epsilon;
                    float rinv = quake_root(r2);
                    float rinv3 = rinv * rinv * rinv;
                    float force = G * rinv3;

                    // Newton's law!!
                    float fx = force * dx;
                    float fy = force * dy;
                    float fz = force * dz;

                    acc_x_private[i] += fx * masses[j];
                    acc_y_private[i] += fy * masses[j];
                    acc_z_private[i] += fz * masses[j];

                    acc_x_private[j] -= fx * mass_i;
                    acc_y_private[j] -= fy * mass_i;
                    acc_z_private[j] -= fz * mass_i;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < num_particles; ++i) {
                    acc_x[i] += acc_x_private[i];
                    acc_y[i] += acc_y_private[i];
                    acc_z[i] += acc_z_private[i];
                }
            }
        }
    }
    private:

    int get_octant(const Node* node, float x, float y, float z) const {
        int octant = 0;
        if (x > node->center_x) octant |= 1;
        if (y > node->center_y) octant |= 2;
        if (z > node->center_z) octant |= 4;
        return octant;
    }

    void insert_particle(Node* node, int particle_id) {
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

        if (node->is_leaf) {
            node->is_leaf = false;
            auto temp_particles = std::move(node->particles);
            for (int pid : temp_particles) {
                insert_particle(node, pid);
            }

            insert_particle(node, particle_id);
            return;
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

    window = glfwCreateWindow(1440, 1080, "N-Body Sim.", NULL, NULL);

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

    const int N = 12000;
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

    sys.G = 0.001f;

    sys.shift_to_COM();
    double last_time = glfwGetTime();
    int frame_count = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        sys.compute_gravity();
        sys.update(0.01f);

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
