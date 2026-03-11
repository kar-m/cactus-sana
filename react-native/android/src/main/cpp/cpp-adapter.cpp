#include <jni.h>
#include "cactusOnLoad.hpp"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return margelo::nitro::cactus::initialize(vm);
}
