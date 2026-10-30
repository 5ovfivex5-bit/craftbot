// Minimal GL 3.3 core entry points, resolved at runtime through SDL_GL_GetProcAddress.
// Only what the dashboard's own GL code (viewport FBO, mesh, textures) needs.
// ImGui's backend keeps using its own bundled loader.
#pragma once

#include <SDL3/SDL_opengl.h>
#include <SDL3/SDL_opengl_glext.h>

namespace glp {

#define MM_GL_FUNCS(X)                                                                            \
    X(void, GenFramebuffers, (GLsizei, GLuint*))                                                  \
    X(void, BindFramebuffer, (GLenum, GLuint))                                                    \
    X(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))                        \
    X(void, FramebufferRenderbuffer, (GLenum, GLenum, GLenum, GLuint))                            \
    X(void, GenRenderbuffers, (GLsizei, GLuint*))                                                 \
    X(void, BindRenderbuffer, (GLenum, GLuint))                                                   \
    X(void, RenderbufferStorage, (GLenum, GLenum, GLsizei, GLsizei))                              \
    X(GLenum, CheckFramebufferStatus, (GLenum))                                                   \
    X(void, DeleteFramebuffers, (GLsizei, const GLuint*))                                         \
    X(void, DeleteRenderbuffers, (GLsizei, const GLuint*))                                        \
    X(void, GenVertexArrays, (GLsizei, GLuint*))                                                  \
    X(void, BindVertexArray, (GLuint))                                                            \
    X(void, DeleteVertexArrays, (GLsizei, const GLuint*))                                         \
    X(void, GenBuffers, (GLsizei, GLuint*))                                                       \
    X(void, BindBuffer, (GLenum, GLuint))                                                         \
    X(void, BufferData, (GLenum, GLsizeiptr, const void*, GLenum))                                \
    X(void, DeleteBuffers, (GLsizei, const GLuint*))                                              \
    X(void, EnableVertexAttribArray, (GLuint))                                                    \
    X(void, VertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*))        \
    X(GLuint, CreateShader, (GLenum))                                                             \
    X(void, ShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*))                  \
    X(void, CompileShader, (GLuint))                                                              \
    X(void, GetShaderiv, (GLuint, GLenum, GLint*))                                                \
    X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                               \
    X(GLuint, CreateProgram, (void))                                                              \
    X(void, AttachShader, (GLuint, GLuint))                                                       \
    X(void, LinkProgram, (GLuint))                                                                \
    X(void, GetProgramiv, (GLuint, GLenum, GLint*))                                               \
    X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                              \
    X(void, UseProgram, (GLuint))                                                                 \
    X(GLint, GetUniformLocation, (GLuint, const GLchar*))                                         \
    X(void, UniformMatrix4fv, (GLint, GLsizei, GLboolean, const GLfloat*))                        \
    X(void, Uniform3f, (GLint, GLfloat, GLfloat, GLfloat))                                        \
    X(void, DeleteShader, (GLuint))                                                               \
    X(void, DeleteProgram, (GLuint))

#define MM_GL_DECL(ret, name, args) extern ret (*name) args;
MM_GL_FUNCS(MM_GL_DECL)
#undef MM_GL_DECL

// Returns false (and logs to stderr) if any required entry point is missing.
bool load();

}  // namespace glp
