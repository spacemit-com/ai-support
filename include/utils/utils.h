#ifndef _INCLUDE_UTILS_UTILS_H_
#define _INCLUDE_UTILS_UTILS_H_

#include <sys/stat.h>
#include <string>

static bool exists_check(const std::string& name) 
{
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

#endif