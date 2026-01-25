#include "rpihub75.h"

#ifndef SCENE_H
#define SCENE_H

#define macro_var(name) concat(name, __LINE__)
#define defer(start, end) for (int macro_var(_i_) = (start, 0); !macro_var(_i_); (macro_var(_i_)++, end))

#endif