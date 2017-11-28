#ifndef WINDOW_H
#define WINDOW_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include "Camera.h"
#include <iostream>

void errorCallback(int error, const char* description)
{
    puts(description);
}
// An interactive game window
class Window {
private:
  GLFWwindow* window;
  Camera* camera;

  // quad to render texture to screen with
  GLuint quadVAO, quadVBO;

  // input
  bool keys[1024];
  bool firstMouse;
  GLfloat lastX, lastY;
  GLfloat deltaTime;
  GLfloat lastFrame;

public:
  int width, height;

  Window(int width, int height, Camera* camera) {
    this->width = width;
    this->height = height;
    this->camera = camera;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwSetErrorCallback(errorCallback);

    // create window
    window = glfwCreateWindow(width, height, "cuda-pathtrace", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetWindowPos(window, 100, 100);

    // turn on GLEW
    glewExperimental = GL_TRUE;
    glewInit();
    //glUseProgram(0);
    // tell OpenGL size of rendering window
    glViewport(0, 0, width, height);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width, 0.0, height);
    // disable cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // set user pointer to this object
    glfwSetWindowUserPointer(window, (void*)this);
    
    // register callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    // set up textured quad
    //InitQuad();

    // set up input
    firstMouse = true;
    lastX = width / 2;
    lastY = height / 2;
    deltaTime = 0.0f;
    lastFrame = 0.0f;
    for (int i = 0; i < 1024; i++)
      keys[i] = false;
  }

  void DrawToScreen(GLPixelBuffer& denoisedBuffer) {
    // bidn buffer
    denoisedBuffer.BindBuffer();
    glClear(GL_COLOR_BUFFER_BIT);
    glVertexPointer(2, GL_FLOAT, 12, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);
    //RenderQuad();
    glfwSwapBuffers(window);
  }

  bool ShouldClose() {
    glfwPollEvents();
    return glfwWindowShouldClose(window);
  }

  // Renders a quad that fills the screen
  void RenderQuad() {
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
  }

  // Set ups the quad
  void InitQuad() {
    GLfloat quadVertices[] = {
			// Positions        // Texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// Setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
  }

  void DoMovement() {
    GLfloat currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // Camera controls
    if (keys[GLFW_KEY_W])
      camera->ProcessKeyboard(FORWARD, deltaTime);
    if (keys[GLFW_KEY_S])
      camera->ProcessKeyboard(BACKWARD, deltaTime);
    if (keys[GLFW_KEY_A])
      camera->ProcessKeyboard(LEFT, deltaTime);
    if (keys[GLFW_KEY_D])
      camera->ProcessKeyboard(RIGHT, deltaTime);
  }

  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    // When a user presses the escape key, we set the WindowShouldClose property to true, 
    // closing the application
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GL_TRUE);

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
      Window* w = (Window*)glfwGetWindowUserPointer(window);
      std::cout << w->camera->Position.x << " " << w->camera->Position.y << " " << w->camera->Position.z << " " << w->camera->Yaw << " " << w->camera->Pitch << " " << std::endl;
    }

    if (key >= 0 && key < 1024) {
      Window* w = (Window*)glfwGetWindowUserPointer(window);
      if (action == GLFW_PRESS)
        w->keys[key] = true;
      else if (action == GLFW_RELEASE)
        w->keys[key] = false;
    }
  }

  static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w->firstMouse) {
      w->lastX = xpos;
      w->lastY = ypos;
      w->firstMouse = false;
    }

    GLfloat xoffset = xpos - w->lastX;
    GLfloat yoffset = w->lastY - ypos; // Reversed since y-coordinates range from bottom to top
    w->lastX = xpos;
    w->lastY = ypos;

    GLfloat sensitivity = 0.05f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    w->camera->ProcessMouseMovement(xoffset, yoffset);
  }

};

#endif