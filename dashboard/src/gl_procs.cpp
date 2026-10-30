#include "gl_procs.h"

#include <cstdio>

#include <SDL3/SDL.h>

namespace glp {

#define MM_GL_DEF(ret, name, args) ret (*name) args = nullptr;
MM_GL_FUNCS(MM_GL_DEF)
#undef MM_GL_DEF

bool load() {
    bool ok = true;
#define MM_GL_LOAD(ret, name, args)                                                              \
    name = reinterpret_cast<decltype(name)>(SDL_GL_GetProcAddress("gl" #name));                  \
    if (!name) {                                                                                 \
        std::fprintf(stderr, "missing GL entry point gl%s\n", #name);                            \
        ok = false;                                                                              \
    }
    MM_GL_FUNCS(MM_GL_LOAD)
#undef MM_GL_LOAD
    return ok;
}

}  // namespace glp
