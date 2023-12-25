#ifndef SUPPORT_SRC_PROCESSOR_PROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_PROCESSOR_H_

class Processor {
 public:
  Processor() = default;
  virtual ~Processor() = default;

  // Processor is neither copyable nor movable.
  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;
};

class Preprocessor : public Processor {
 protected:
  using Processor::Processor;
};

class Postprocessor : public Processor {
 protected:
  using Processor::Processor;
};

#endif  // SUPPORT_SRC_PROCESSOR_PROCESSOR_H_
