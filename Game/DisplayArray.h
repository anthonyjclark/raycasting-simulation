#if !defined(_DISPLAY_ARRAY_H_)
#define _DISPLAY_ARRAY_H_

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>

const char *vertexShaderSource = R"vert(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	TexCoord = aTexCoord;
}
)vert";

const char *fragmentShaderSource = R"frag(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main()
{
    FragColor = texture(texture1, TexCoord);
}
)frag";

void errorCallback(int error, const char *message)
{
    std::cerr << "Error: " << message << " (" << error << ")\n";
}

void resizeCallback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

struct DisplayArray
{
    bool ready;
    GLFWwindow *window;
    uint32_t width, height;

    DisplayArray(uint32_t width, uint32_t height, GLFWkeyfun keyCallback);

    ~DisplayArray()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool running()
    {
        return !glfwWindowShouldClose(window);
    }

    void pre()
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void render(const void *array)
    {
        // Draw the current texture using the current program and vertex array
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, array);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    void post()
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
};

DisplayArray::DisplayArray(uint32_t width, uint32_t height, GLFWkeyfun keyCallback)
    : ready(true), width(width), height(height)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        // Initialization failed
        std::cerr << "Failed to initialize GLFW\n";
        ready = false;
        return;
    }

    // Configure GLFW for my mac
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(width, height, "Display Array", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        ready = false;
        return;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, resizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    // Prevent tearing
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD\n";
        ready = false;
        return;
    }

    //
    // OpenGL setup
    //

    GLint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int shaderSuccess;
    char shaderLog[512];

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &shaderSuccess);
    if (!shaderSuccess)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, shaderLog);
        std::cerr << "Error: vertex shader compilation failed\n"
                  << vertexShaderSource << "\n"
                  << shaderLog << std::endl;
        ready = false;
        return;
    }

    GLint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &shaderSuccess);
    if (!shaderSuccess)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, shaderLog);
        std::cerr << "Error: fragment shader compilation failed\n"
                  << fragmentShaderSource << "\n"
                  << shaderLog << std::endl;
        ready = false;
        return;
    }

    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &shaderSuccess);
    if (!shaderSuccess)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, shaderLog);
        std::cout << "Error: shader program linking failed\n"
                  << shaderLog << std::endl;
        ready = false;
        return;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float vertices[] = {
        // (x, y, z, u, v)
        +1.0f, +1.0f, 0.0f, 1.0f, 0.0f, // top right
        +1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // bottom left
        -1.0f, +1.0f, 0.0f, 0.0f, 0.0f  // top left
    };

    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    // Vertex array (the following buffers as assigned to this object)
    GLuint vertexArrayID;
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    // Vertex buffer
    GLuint vertexBufferID;
    glGenBuffers(1, &vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Indices buffer
    GLuint indexBufferID;
    glGenBuffers(1, &indexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    // set the texture wrapping parameters
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // // set texture filtering parameters
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);
}

#endif // _DISPLAY_ARRAY_H_
