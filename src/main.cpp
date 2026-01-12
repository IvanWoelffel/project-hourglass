#include <SDL3/SDL.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <map>

using namespace ge::gl;

// =====================================================
// Matrix utilities (column-major, course style)
// =====================================================

void matrixIdentity(float* m) {
    for (int i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.f : 0.f;
}

void matrixMultiply(float* O, const float* A, const float* B) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            O[c * 4 + r] = 0;
            for (int i = 0; i < 4; ++i)
                O[c * 4 + r] += A[i * 4 + r] * B[c * 4 + i];
        }
}

void translate(float* m, float x, float y, float z) {
    matrixIdentity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

void rotateX(float* m, float a) {
    matrixIdentity(m);
    m[5] = cos(a);  m[6] = sin(a);
    m[9] = -sin(a); m[10] = cos(a);
}

void rotateY(float* m, float a) {
    matrixIdentity(m);
    m[0] = cos(a);  m[2] = sin(a);
    m[8] = -sin(a); m[10] = cos(a);
}

void frustum(float* m, float l, float r, float b, float t, float n, float f) {
    matrixIdentity(m);
    m[0] = 2 * n / (r - l);
    m[5] = 2 * n / (t - b);
    m[8] = (r + l) / (r - l);
    m[9] = (t + b) / (t - b);
    m[10] = -(f + n) / (f - n);
    m[11] = -1;
    m[14] = -2 * f * n / (f - n);
    m[15] = 0;
}

void perspective(float* m, float aspect, float fovyDeg, float n, float f) {
    float fovy = fovyDeg * 3.1415926f / 180.f;
    float t = n * tan(fovy / 2.f);
    frustum(m, -t * aspect, t * aspect, -t, t, n, f);
}

// =====================================================
// Shader helpers
// =====================================================

GLuint createShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, 4096, nullptr, log);
        std::cerr << log << std::endl;
    }
    return s;
}

GLuint createProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, 4096, nullptr, log);
        std::cerr << log << std::endl;
    }
    return p;
}

// =====================================================
// Materials
// =====================================================

enum class MaterialType { GLASS = 0, METAL = 1, GOLD = 2, DEFAULT = 3 };

struct GPUMesh {
    GLuint vao, vboPos, vboNrm, ebo;
    GLsizei indexCount;
    float baseColor[3];
    MaterialType material;
};

struct SceneData {
    std::vector<GPUMesh> meshes;
};

// =====================================================
// Particle system (sand)
// =====================================================

struct Particle {
    float pos[3];
    float vel[3];
};

struct ParticleSystem {
    std::vector<Particle> particles;
    GLuint vao = 0;
    GLuint vbo = 0;
};

ParticleSystem createSand(size_t count) {
    ParticleSystem ps;
    ps.particles.resize(count);

    for (auto& p : ps.particles) {
        p.pos[0] = ((rand() / float(RAND_MAX)) - 0.5f) * 2.0f;
        p.pos[1] = 6.0f + (rand() / float(RAND_MAX)) * 3.0f;
        p.pos[2] = ((rand() / float(RAND_MAX)) - 0.5f) * 2.0f;

        p.vel[0] = 0.f;
        p.vel[1] = 0.f;
        p.vel[2] = 0.f;
    }

    glCreateVertexArrays(1, &ps.vao);
    glCreateBuffers(1, &ps.vbo);

    glNamedBufferData(ps.vbo,
        ps.particles.size() * sizeof(Particle),
        ps.particles.data(),
        GL_DYNAMIC_DRAW
    );

    glVertexArrayVertexBuffer(ps.vao, 0, ps.vbo, 0, sizeof(Particle));
    glEnableVertexArrayAttrib(ps.vao, 0);
    glVertexArrayAttribFormat(ps.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(ps.vao, 0, 0);

    return ps;
}

void updateSand(ParticleSystem& ps, float dt) {
    const float gravity = -0.001f;
    const float floorY = -20.0f;
    const float neckY = -10.0f;
    const float neckRadius = 0.6f;

    for (auto& p : ps.particles) {
        p.vel[1] += gravity * dt;

        for (int i = 0; i < 3; ++i)
            p.pos[i] += p.vel[i] * dt;

        if (p.pos[1] < floorY) {
            p.pos[1] = floorY;
            p.vel[1] = 0.f;
        }

        if (fabs(p.pos[1] - neckY) < 0.5f) {
            float r = sqrt(p.pos[0] * p.pos[0] + p.pos[2] * p.pos[2]);
            if (r > neckRadius) {
                p.pos[0] *= neckRadius / r;
                p.pos[2] *= neckRadius / r;
            }
        }
    }

    glNamedBufferSubData(ps.vbo, 0,
        ps.particles.size() * sizeof(Particle),
        ps.particles.data()
    );
}


// =====================================================
// Load GLB
// =====================================================

SceneData loadGLBScene(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_PreTransformVertices
    );

    if (!scene || !scene->HasMeshes()) {
        std::cerr << importer.GetErrorString() << std::endl;
        exit(1);
    }

    SceneData out;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* aimesh = scene->mMeshes[m];
        const aiMaterial* mat = scene->mMaterials[aimesh->mMaterialIndex];

        GPUMesh mesh;
        mesh.material = MaterialType::DEFAULT;

        aiColor3D diff(0.8f, 0.8f, 0.8f);
        mat->Get(AI_MATKEY_COLOR_DIFFUSE, diff);
        mesh.baseColor[0] = diff.r;
        mesh.baseColor[1] = diff.g;
        mesh.baseColor[2] = diff.b;

        aiString name;
        mat->Get(AI_MATKEY_NAME, name);
        std::string n = name.C_Str();

        if (n.find("Glass") != std::string::npos) mesh.material = MaterialType::GLASS;
        else if (n.find("Gold") != std::string::npos) mesh.material = MaterialType::GOLD;
        else if (n.find("Metal") != std::string::npos) mesh.material = MaterialType::METAL;

        std::vector<float> pos, nrm;
        std::vector<unsigned int> idx;

        for (unsigned int v = 0; v < aimesh->mNumVertices; ++v) {
            pos.insert(pos.end(), {
                aimesh->mVertices[v].x,
                aimesh->mVertices[v].y,
                aimesh->mVertices[v].z
                });
            nrm.insert(nrm.end(), {
                aimesh->mNormals[v].x,
                aimesh->mNormals[v].y,
                aimesh->mNormals[v].z
                });
        }

        for (unsigned int f = 0; f < aimesh->mNumFaces; ++f)
            idx.insert(idx.end(), {
                aimesh->mFaces[f].mIndices[0],
                aimesh->mFaces[f].mIndices[1],
                aimesh->mFaces[f].mIndices[2]
                });

        mesh.indexCount = (GLsizei)idx.size();

        glCreateBuffers(1, &mesh.vboPos);
        glNamedBufferData(mesh.vboPos, pos.size() * sizeof(float), pos.data(), GL_STATIC_DRAW);

        glCreateBuffers(1, &mesh.vboNrm);
        glNamedBufferData(mesh.vboNrm, nrm.size() * sizeof(float), nrm.data(), GL_STATIC_DRAW);

        glCreateBuffers(1, &mesh.ebo);
        glNamedBufferData(mesh.ebo, idx.size() * sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);

        glCreateVertexArrays(1, &mesh.vao);

        glEnableVertexArrayAttrib(mesh.vao, 0);
        glVertexArrayAttribFormat(mesh.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(mesh.vao, 0, 0);
        glVertexArrayVertexBuffer(mesh.vao, 0, mesh.vboPos, 0, sizeof(float) * 3);

        glEnableVertexArrayAttrib(mesh.vao, 1);
        glVertexArrayAttribFormat(mesh.vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(mesh.vao, 1, 1);
        glVertexArrayVertexBuffer(mesh.vao, 1, mesh.vboNrm, 0, sizeof(float) * 3);

        glVertexArrayElementBuffer(mesh.vao, mesh.ebo);

        out.meshes.push_back(mesh);
    }
    return out;
}

// =====================================================
// MAIN
// =====================================================

int main() {
    SDL_Window* window = SDL_CreateWindow("Hourglass Free Camera",
        1024, 768, SDL_WINDOW_OPENGL);
    SDL_GLContext context = SDL_GL_CreateContext(window);
    ge::gl::init();

    SceneData scene = loadGLBScene("models/hourglass.glb");
    ParticleSystem sand = createSand(4000);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ================= BACKGROUND SHADER =================

    const char* bgVsSrc = R"(#version 460
        const vec2 verts[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2( 3.0, -1.0),
            vec2(-1.0,  3.0)
        );
        out vec2 vUV;
        void main() {
            gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
            vUV = (gl_Position.xy + 1.0) * 0.5;
        }
    )";

    const char* bgFsSrc = R"(#version 460
        in vec2 vUV;
        out vec4 fragColor;
        void main() {
            vec3 topColor    = vec3(0.30, 0.55, 0.95);
            vec3 bottomColor = vec3(0.08, 0.08, 0.12);
            vec3 col = mix(bottomColor, topColor, vUV.y);
            fragColor = vec4(col, 1.0);
        }
    )";

    GLuint bgProg = createProgram(
        createShader(GL_VERTEX_SHADER, bgVsSrc),
        createShader(GL_FRAGMENT_SHADER, bgFsSrc)
    );

    // ================= SAND SHADERS =================

    const char* sandVs = R"(#version 460
    layout(location=0) in vec3 position;
    uniform mat4 viewMatrix;
    uniform mat4 projMatrix;
    void main(){
        gl_Position=projMatrix*viewMatrix*vec4(position,1);
        gl_PointSize=2.5;
    })";

    const char* sandFs = R"(#version 460
    out vec4 fragColor;
    void main(){
        fragColor=vec4(0.85,0.75,0.45,1);
    })";

    GLuint sandProg = createProgram(
        createShader(GL_VERTEX_SHADER, sandVs),
        createShader(GL_FRAGMENT_SHADER, sandFs)
    );

    GLint uSandView = glGetUniformLocation(sandProg, "viewMatrix");
    GLint uSandProj = glGetUniformLocation(sandProg, "projMatrix");


    // ================= HOURGLASS SHADERS =================

    const char* vsSrc = R"(#version 460
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    uniform mat4 viewMatrix;
    uniform mat4 projMatrix;
    out vec3 vPos;
    out vec3 vNormal;
    void main(){
        vPos=position;
        vNormal=normal;
        gl_Position=projMatrix*viewMatrix*vec4(position,1);
    })";

    const char* fsSrc = R"(#version 460
    in vec3 vPos;
    in vec3 vNormal;
    uniform vec3 uColor;
    uniform vec3 viewPos;
    uniform int uMaterialType;
    uniform vec3 lightPos;
    uniform vec3 lightPos2;
    uniform vec3 lightColor2;
    out vec4 fragColor;

    vec3 light(vec3 lp,vec3 lc,vec3 dc,vec3 sc,float s){
        vec3 N=normalize(vNormal);
        vec3 L=normalize(lp-vPos);
        vec3 V=normalize(viewPos-vPos);
        vec3 H=normalize(L+V);
        float d=max(dot(N,L),0);
        float sp=pow(max(dot(N,H),0),s);
        return d*dc*lc + sp*sc*lc;
    }

    void main(){
        float shin=32;
        float a=1;
        vec3 dc=uColor;
        vec3 sc=vec3(1);

        if(uMaterialType==0){ a=0.08; shin=180; dc*=0.2; }
        else if(uMaterialType==1){ shin=70; dc*=0.6; }
        else if(uMaterialType==2){ shin=160; dc=vec3(1,0.75,0.25); sc=vec3(1,0.92,0.45); }

        vec3 col=0.05*dc;
        col+=light(lightPos,vec3(1),dc,sc,shin);
        col+=light(lightPos2,lightColor2,dc,sc,shin*0.6);

        fragColor=vec4(col,a);
    })";

    GLuint prog = createProgram(
        createShader(GL_VERTEX_SHADER, vsSrc),
        createShader(GL_FRAGMENT_SHADER, fsSrc)
    );

    GLint uView = glGetUniformLocation(prog, "viewMatrix");
    GLint uProj = glGetUniformLocation(prog, "projMatrix");
    GLint uCol = glGetUniformLocation(prog, "uColor");
    GLint uMat = glGetUniformLocation(prog, "uMaterialType");
    GLint uViewP = glGetUniformLocation(prog, "viewPos");
    GLint uL1 = glGetUniformLocation(prog, "lightPos");
    GLint uL2 = glGetUniformLocation(prog, "lightPos2");
    GLint uL2C = glGetUniformLocation(prog, "lightColor2");

    // ================= Camera =================

    float camPos[3] = { 0,0,50 };
    float angleX = 0.3f, angleY = 0.6f;
    float speed = 0.05f;
    float zoomSpeed = 2.0f;

    float RX[16], RY[16], R[16], T[16], V[16], P[16];

    std::map<int, bool> keys;

    float light1[3] = { 8,10,8 };
    float light2[3] = { -6,4,-6 };
    float light2Col[3] = { 0.4f,0.45f,0.6f };

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            if (e.type == SDL_EVENT_KEY_DOWN) keys[e.key.key] = true;
            if (e.type == SDL_EVENT_KEY_UP) keys[e.key.key] = false;
            if (e.type == SDL_EVENT_MOUSE_MOTION && (e.motion.state & SDL_BUTTON_LEFT)) {
                angleY -= e.motion.xrel * 0.01f;
                angleX += e.motion.yrel * 0.01f;
            }
            if (e.type == SDL_EVENT_MOUSE_WHEEL) {
                float zoom = e.wheel.y * zoomSpeed;
                camPos[0] -= R[2] * zoom;
                camPos[1] -= R[6] * zoom;
                camPos[2] -= R[10] * zoom;
            }
        }

        rotateX(RX, angleX);
        rotateY(RY, angleY);
        matrixMultiply(R, RX, RY);

        // ----------------- Camera movement -----------------

        float moveUp = ((keys[SDLK_Z] || keys[SDLK_UP]) - (keys[SDLK_S] || keys[SDLK_DOWN])) * speed;
        float moveSide = ((keys[SDLK_D] || keys[SDLK_RIGHT]) - (keys[SDLK_Q] || keys[SDLK_LEFT])) * speed;

        // Apply movements in WORLD space
        camPos[0] += R[0] * moveSide + R[1] * (moveUp);
        camPos[1] += R[4] * moveSide + R[5] * (moveUp);
        camPos[2] += R[8] * moveSide + R[9] * (moveUp);


        translate(T, -camPos[0], -camPos[1], -camPos[2]);
        matrixMultiply(V, R, T);
        perspective(P, 1024.f / 768.f, 60, 0.1f, 500);

        // ---------- Draw background ----------
        glClear(GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glUseProgram(bgProg);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_DEPTH_TEST);

        // ---------- Draw sand ----------------

        updateSand(sand, 0.016f);
        glUseProgram(sandProg);
        glProgramUniformMatrix4fv(sandProg, uSandView, 1, GL_FALSE, V);
        glProgramUniformMatrix4fv(sandProg, uSandProj, 1, GL_FALSE, P);
        glBindVertexArray(sand.vao);
        glDrawArrays(GL_POINTS, 0, sand.particles.size());


        // ---------- Draw hourglass -----------
        glUseProgram(prog);
        glProgramUniformMatrix4fv(prog, uView, 1, GL_FALSE, V);
        glProgramUniformMatrix4fv(prog, uProj, 1, GL_FALSE, P);
        glProgramUniform3fv(prog, uViewP, 1, camPos);
        glProgramUniform3fv(prog, uL1, 1, light1);
        glProgramUniform3fv(prog, uL2, 1, light2);
        glProgramUniform3fv(prog, uL2C, 1, light2Col);

        for (auto& m : scene.meshes) {
            glProgramUniform3fv(prog, uCol, 1, m.baseColor);
            glProgramUniform1i(prog, uMat, (int)m.material);
            glBindVertexArray(m.vao);
            glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, nullptr);
        }

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    return 0;
}
