#ifndef _PROCESSOR_H_
#define _PROCESSOR_H_

class Processor{
    public:
    Processor() = default;
    virtual ~Processor() = default;

    // Processor is neither copyable nor movable.
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;
};

class Preprocessor : public Processor{
    protected:
    using Processor::Processor;
};

class Postprocessor : public Processor{
    protected:
    using Processor::Processor;
};

#endif