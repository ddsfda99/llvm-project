//===- bolt/Passes/Outliner.h -----------------------------------*- C++ -*-===//
//
// This file declares the Outliner class which implements code outlining
// optimization for BOLT.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_OUTLINER_H
#define BOLT_PASSES_OUTLINER_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Passes/BinaryPasses.h"
#include <string>

namespace llvm {
namespace bolt {

class BinaryContext;
class BinaryFunction;

class Outliner : public BinaryFunctionPass {
private:
  unsigned OutlinedFuncCounter = 0;

  /// Generate a unique name for an outlined function.
  std::string generateOutlinedFunctionName();

  /// Create a new outlined function containing instructions [Begin, End)
  /// from the given basic block. Returns the new function, or nullptr on failure.
  /// \param NeedsPrologueEpilogue Whether the outlined function needs prologue/epilogue
  /// \param EndsWithCall Whether the sequence ends with a call instruction
  BinaryFunction *createOutlinedFunction(BinaryContext &BC,
                                         BinaryFunction &Caller,
                                         BinaryBasicBlock &BB,
                                         BinaryBasicBlock::iterator Begin,
                                         BinaryBasicBlock::iterator End,
                                         bool NeedsPrologueEpilogue,
                                         bool EndsWithCall);

public:
  explicit Outliner(const cl::opt<bool> &PrintPass);

  Error runOnFunctions(BinaryContext &BC) override;

  const char *getName() const override { return "Outliner"; }
};

} // namespace bolt
} // namespace llvm

#endif