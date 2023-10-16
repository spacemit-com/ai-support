#ifndef _TASK_API_FACTORY_H_
#define _TASK_API_FACTORY_H_

#include <memory>

#include "base_task_api.h"
#include "engine.h"

template <typename T>
using EnableIfBaseUntypedTaskApiSubclass = typename std::enable_if<
    std::is_base_of<BaseUntypedTaskApi, T>::value>::type*;

// Template creator for all subclasses of BaseTaskApi
class TaskAPIFactory {
    public:
    TaskAPIFactory() = delete;
    template <typename T, EnableIfBaseUntypedTaskApiSubclass<T> = nullptr>
    static std::unique_ptr<T> Create()
    {
        auto engine = std::unique_ptr<Engine>(new Engine());
        return CreateFromEngine<T>(std::move(engine));
    }
    private:
    template <typename T, EnableIfBaseUntypedTaskApiSubclass<T> = nullptr>
    static std::unique_ptr<T> CreateFromEngine(std::unique_ptr<Engine> engine){
        return std::unique_ptr<T>(new T(std::move(engine)));
    }
};
#endif