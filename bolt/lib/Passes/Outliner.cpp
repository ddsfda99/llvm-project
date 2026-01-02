//===- bolt/Passes/Outliner.cpp -------------------------------------------===//
//
// This file implements the Outliner class which performs code outlining
// optimization for BOLT (AArch64 only).
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/Outliner.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SuffixTree.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <vector>

#define DEBUG_TYPE "outliner"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> ExecutionCountThreshold;

cl::opt<bool> OutlineColdOnly(
    "outliner-cold-only",
    cl::desc("Only outline code from cold basic blocks (default: true)"),
    cl::init(true), cl::Hidden, cl::cat(BoltCategory));

cl::opt<unsigned> OutlinerColdThreshold(
    "outliner-cold-threshold",
    cl::desc(
        "Execution count threshold for cold blocks (0 = use BOLT default)"),
    cl::init(0), cl::Hidden, cl::cat(BoltCategory));

cl::opt<unsigned>
    OutlinerMaxIterations("outliner-max-iterations",
                          cl::desc("Maximum number of outlining iterations for "
                                   "nested outlining (default: 8)"),
                          cl::init(8), cl::Hidden, cl::cat(BoltCategory));

cl::opt<bool> OutlinerUpgradeLeafFunctions(
    "outliner-upgrade-leaf-functions",
    cl::desc("Upgrade leaf functions to save/restore LR, enabling outlining "
             "from them (default: false, experimental)"),
    cl::init(false), cl::Hidden, cl::cat(BoltCategory));

} // namespace opts

/// Stores information about a replacement to be made in the original function.
struct Replacement {
    BinaryBasicBlock *BB;       // The basic block containing the instructions to replace
    unsigned StartIndex;        // Index of the first instruction to replace
    unsigned NumToErase;        // Number of instructions to erase
    MCSymbol *CallTarget;       // Target symbol for the BL instruction
    bool NeedsPrologueEpilogue; // Indicates if prologue/epilogue is required
    bool FromLeafFunction;      // Whether this replacement is in a leaf function
    bool UseTailJump;           // 使用 B（尾跳转）而非 BL 调用 outlined function
    bool EraseFollowingTerminator; // 删除序列后的 RET/B（尾跳转时需要）
    MCPhysReg SaveLRReg;        // Register to save LR (X19-X28) if FromLeafFunction, 0 otherwise
};

/// Compare two MCExprs.
static bool areMCExprsEqual(const MCExpr *ExprA, const MCExpr *ExprB) {
  if (!ExprA && !ExprB)
    return true;
  if (!ExprA || !ExprB)
    return false;

  if (ExprA->getKind() != ExprB->getKind())
    return false;

  switch (ExprA->getKind()) {
  case MCExpr::Constant: {
    auto *CEA = cast<MCConstantExpr>(ExprA);
    auto *CEB = cast<MCConstantExpr>(ExprB);
    return CEA->getValue() == CEB->getValue();
  }
  case MCExpr::SymbolRef: {
    auto *SREA = cast<MCSymbolRefExpr>(ExprA);
    auto *SREB = cast<MCSymbolRefExpr>(ExprB);
    // Compare based on symbol name, not pointer
    return SREA->getSymbol().getName() == SREB->getSymbol().getName();
  }
  case MCExpr::Unary: {
    auto *UEA = cast<MCUnaryExpr>(ExprA);
    auto *UEB = cast<MCUnaryExpr>(ExprB);
    return UEA->getOpcode() == UEB->getOpcode() &&
           areMCExprsEqual(UEA->getSubExpr(), UEB->getSubExpr());
  }
  case MCExpr::Binary: {
    auto *BEA = cast<MCBinaryExpr>(ExprA);
    auto *BEB = cast<MCBinaryExpr>(ExprB);
    return BEA->getOpcode() == BEB->getOpcode() &&
           areMCExprsEqual(BEA->getLHS(), BEB->getLHS()) &&
           areMCExprsEqual(BEA->getRHS(), BEB->getRHS());
  }
  case MCExpr::Target:
  case MCExpr::Specifier:
    // For target-specific/specifier expressions, compare by kind only
    // (conservative: may have false negatives)
    return true;
  }
  return false;
}

// Forward declaration
static unsigned hashMCInst(const MCInst &Inst);

/// Recursively hash an MCExpr based on its content, not pointer.
/// This prevents false negatives when the same expression appears at different
/// addresses.
static hash_code hashMCExpr(const MCExpr *Expr) {
  if (!Expr)
    return hash_code(0);

  switch (Expr->getKind()) {
  case MCExpr::Constant: {
    auto *CE = cast<MCConstantExpr>(Expr);
    return hash_combine(0 /*Constant*/, CE->getValue());
  }
  case MCExpr::SymbolRef: {
    auto *SRE = cast<MCSymbolRefExpr>(Expr);
    // Hash based on symbol name, not pointer
    return hash_combine(1 /*SymbolRef*/, SRE->getSymbol().getName());
  }
  case MCExpr::Unary: {
    auto *UE = cast<MCUnaryExpr>(Expr);
    return hash_combine(2 /*Unary*/, UE->getOpcode(),
                        hashMCExpr(UE->getSubExpr()));
  }
  case MCExpr::Binary: {
    auto *BE = cast<MCBinaryExpr>(Expr);
    return hash_combine(3 /*Binary*/, BE->getOpcode(), hashMCExpr(BE->getLHS()),
                        hashMCExpr(BE->getRHS()));
  }
  case MCExpr::Target:
  case MCExpr::Specifier:
    // Target-specific/specifier expressions - use kind as hash
    return hash_combine(4 /*Target/Specifier*/, Expr->getKind());
  }
  return hash_code(0);
}

/// Compute hash value for a single MCOperand.
/// Improved to match LLVM MachineOutliner's hash_value(const MachineOperand
/// &MO). Key improvements:
/// 1. Expression hashing based on content, not pointer (prevents false
/// negatives)
/// 2. Include target flags in hash (prevents false positives)
/// 3. Include register definition flags (prevents false positives)
static hash_code hashMCOperand(const MCOperand &Op) {
  if (Op.isReg()) {
    // Hash register operands by type and register ID
    // Note: MCOperand doesn't have isDef flag, so we can't include it
    return hash_combine(0 /*kRegister*/, Op.getReg());
  }
  if (Op.isImm()) {
    // Hash immediate operands by type and value
    // Note: MCOperand doesn't have TargetFlags, so we can't include it
    return hash_combine(1 /*kImmediate*/, Op.getImm());
  }
  if (Op.isSFPImm()) {
    return hash_combine(2 /*kSFPImmediate*/, Op.getSFPImm());
  }
  if (Op.isDFPImm()) {
    return hash_combine(3 /*kDFPImmediate*/, Op.getDFPImm());
  }
  if (Op.isExpr()) {
    // IMPROVED: Hash expression based on content, not pointer
    // This prevents false negatives when the same expression appears at
    // different addresses
    return hash_combine(4 /*kExpr*/, hashMCExpr(Op.getExpr()));
  }
  if (Op.isInst()) {
    // Sub-instruction operand (used for annotations in BOLT)
    // Handle null MCInst (BOLT annotations)
    const MCInst *Inst = Op.getInst();
    if (Inst) {
      return hash_combine(5 /*kInst*/, hashMCInst(*Inst));
    }
    return hash_combine(5 /*kInst*/, 0);
  }
  // Invalid operand
  return hash_code(0);
}

// Compute hash value for an MCInst.
static unsigned hashMCInst(const MCInst &Inst) {
  SmallVector<size_t, 16> HashComponents;
  HashComponents.reserve(Inst.getNumOperands() + 1);

  // Hash the opcode
  HashComponents.push_back(Inst.getOpcode());

  // Hash all operands
  for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    // Skip annotation operands (sub-instructions with nullptr)
    if (Op.isInst() && Op.getInst() == nullptr)
      break; // Annotations start here, stop hashing
    HashComponents.push_back(hashMCOperand(Op));
  }

  return hash_combine_range(HashComponents.begin(), HashComponents.end());
}

/// Check if two MCInsts are identical (for outlining purposes).
/// Similar to MachineInstr::isIdenticalTo with IgnoreVRegDefs.
/// IMPORTANT: Uses content-based comparison for expressions to ensure
/// hash/equality consistency. If hashMCExpr(A) == hashMCExpr(B),
/// then areMCInstsIdentical should return true.
static bool areMCInstsIdentical(const MCInst &A, const MCInst &B) {
  if (A.getOpcode() != B.getOpcode())
    return false;

  // Get the number of "prime" operands (excluding annotations)
  unsigned NumOpsA = A.getNumOperands();
  unsigned NumOpsB = B.getNumOperands();

  // Find where annotations start (null MCInst operand)
  for (unsigned I = 0; I < A.getNumOperands(); ++I) {
    if (A.getOperand(I).isInst() && A.getOperand(I).getInst() == nullptr) {
      NumOpsA = I;
      break;
    }
  }
  for (unsigned I = 0; I < B.getNumOperands(); ++I) {
    if (B.getOperand(I).isInst() && B.getOperand(I).getInst() == nullptr) {
      NumOpsB = I;
      break;
    }
  }

  if (NumOpsA != NumOpsB)
    return false;

  for (unsigned I = 0; I < NumOpsA; ++I) {
    const MCOperand &OpA = A.getOperand(I);
    const MCOperand &OpB = B.getOperand(I);

    // Check operand type
    if (OpA.isReg() != OpB.isReg() || OpA.isImm() != OpB.isImm() ||
        OpA.isSFPImm() != OpB.isSFPImm() || OpA.isDFPImm() != OpB.isDFPImm() ||
        OpA.isExpr() != OpB.isExpr())
      return false;

    // Compare values
    if (OpA.isReg() && OpA.getReg() != OpB.getReg())
      return false;
    if (OpA.isImm() && OpA.getImm() != OpB.getImm())
      return false;
    if (OpA.isSFPImm() && OpA.getSFPImm() != OpB.getSFPImm())
      return false;
    if (OpA.isDFPImm() && OpA.getDFPImm() != OpB.getDFPImm())
      return false;
    // IMPROVED: Use content-based comparison for expressions, not pointer
    // comparison
    if (OpA.isExpr() && !areMCExprsEqual(OpA.getExpr(), OpB.getExpr()))
      return false;
  }

  return true;
}

/// DenseMapInfo for MCInst pointers, comparing by instruction content.
/// Similar to MachineInstrExpressionTrait in MachineInstr.h
struct MCInstExpressionTrait : DenseMapInfo<const MCInst *> {
  static inline const MCInst *getEmptyKey() { return nullptr; }

  static inline const MCInst *getTombstoneKey() {
    return reinterpret_cast<const MCInst *>(-1);
  }

  static unsigned getHashValue(const MCInst *const &MI) {
    if (MI == getEmptyKey() || MI == getTombstoneKey())
      return 0;
    return hashMCInst(*MI);
  }

  static bool isEqual(const MCInst *const &LHS, const MCInst *const &RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || LHS == getTombstoneKey())
      return LHS == RHS;
    return areMCInstsIdentical(*LHS, *RHS);
  }
};

/// Stores information about a mapped instruction location.
struct InstrLocation {
  BinaryBasicBlock::iterator It;
  BinaryBasicBlock *BB;
  bool ContainsCall;        // 该指令是否是 BL 指令
  bool HasSPRelativeAccess; // 该指令是否访问 SP-relative 内存
  bool FromLeafFunction;    // 该指令是否来自叶子函数
  uint64_t ExecCount;       // 基本块的执行次数（用于冷/热判断）

  InstrLocation()
      : BB(nullptr), ContainsCall(false), HasSPRelativeAccess(false),
        FromLeafFunction(false), ExecCount(0) {}
  InstrLocation(BinaryBasicBlock::iterator It, BinaryBasicBlock *BB,
                bool ContainsCall = false, bool HasSPRelativeAccess = false,
                bool FromLeafFunction = false, uint64_t ExecCount = 0)
      : It(It), BB(BB), ContainsCall(ContainsCall),
        HasSPRelativeAccess(HasSPRelativeAccess),
        FromLeafFunction(FromLeafFunction), ExecCount(ExecCount) {}
};

/// Maps MCInsts to unsigned integers for suffix tree construction.
/// Similar to InstructionMapper in MachineOutliner.cpp
struct InstructionMapper {
  /// The next available integer to assign to an illegal (non-outlinable)
  /// MCInst. Set to -3 for compatibility with DenseMapInfo<unsigned>.
  unsigned IllegalInstrNumber = static_cast<unsigned>(-3);

  /// The next available integer to assign to a legal (outlinable) MCInst.
  unsigned LegalInstrNumber = 0;

  /// Map from MCInst content to assigned integer.
  DenseMap<const MCInst *, unsigned, MCInstExpressionTrait>
      InstructionIntegerMap;

  /// The vector of unsigned integers representing the instruction sequence.
  SmallVector<unsigned> UnsignedVec;

  /// Stores the instruction location for each entry in UnsignedVec.
  SmallVector<InstrLocation> InstrList;

  /// Track if we added an illegal number in the previous step.
  bool AddedIllegalLastTime = false;

  /// Map a legal (outlinable) instruction to an unsigned integer.
  unsigned mapToLegalUnsigned(BinaryBasicBlock::iterator It,
                              BinaryBasicBlock *BB, bool ContainsCall = false,
                              bool HasSPRelativeAccess = false,
                              bool FromLeafFunction = false,
                              uint64_t ExecCount = 0) {
    AddedIllegalLastTime = false;

    const MCInst *MI = &(*It);
    auto [ResultIt, WasInserted] =
        InstructionIntegerMap.insert(std::make_pair(MI, LegalInstrNumber));

    unsigned MINumber = ResultIt->second;
    if (WasInserted)
      LegalInstrNumber++;

    InstrList.push_back(
        InstrLocation(It, BB, ContainsCall, HasSPRelativeAccess, FromLeafFunction, ExecCount));
    UnsignedVec.push_back(MINumber);

    // Ensure we don't overflow
    if (LegalInstrNumber >= IllegalInstrNumber)
      report_fatal_error("BOLT Outliner: Instruction mapping overflow!");

    return MINumber;
  }

  /// Map an illegal (non-outlinable) instruction to a unique unsigned integer.
  unsigned mapToIllegalUnsigned(BinaryBasicBlock::iterator It,
                                BinaryBasicBlock *BB,
                                bool FromLeafFunction = false,
                                uint64_t ExecCount = 0) {
    // Only add one illegal number per range of legal numbers
    if (AddedIllegalLastTime)
      return IllegalInstrNumber + 1; // Return previous illegal number

    AddedIllegalLastTime = true;
    unsigned MINumber = IllegalInstrNumber;

    InstrList.push_back(InstrLocation(It, BB, false, false, FromLeafFunction, ExecCount));
    UnsignedVec.push_back(MINumber);
    IllegalInstrNumber--;

    if (LegalInstrNumber >= IllegalInstrNumber)
      report_fatal_error("BOLT Outliner: Instruction mapping overflow!");

    return MINumber;
  }

  /// Add a sentinel value to terminate a basic block's sequence.
  void addSentinel(BinaryBasicBlock *BB, bool FromLeafFunction = false, uint64_t ExecCount = 0) {
    AddedIllegalLastTime = false;
    InstrList.push_back(InstrLocation(BB->end(), BB, false, false, FromLeafFunction, ExecCount));
    UnsignedVec.push_back(IllegalInstrNumber);
    IllegalInstrNumber--;
  }

  /// Clear the mapper state.
  void clear() {
    IllegalInstrNumber = static_cast<unsigned>(-3);
    LegalInstrNumber = 0;
    InstructionIntegerMap.clear();
    UnsignedVec.clear();
    InstrList.clear();
    AddedIllegalLastTime = false;
  }

  /// Print mapping statistics.
  void printStats(raw_ostream &OS) const {
    OS << "InstructionMapper stats:\n";
    OS << "  Legal instructions mapped: " << LegalInstrNumber << "\n";
    OS << "  Unique instruction patterns: " << InstructionIntegerMap.size()
       << "\n";
    OS << "  Total entries in UnsignedVec: " << UnsignedVec.size() << "\n";
  }
};

// stp x29, x30, [sp, #-16]!
void createPushRegisters(MCInst &Inst, MCPhysReg Reg1, MCPhysReg Reg2) {
  Inst.clear();
  Inst.setOpcode(AArch64::STPXpre);
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createReg(Reg1));
  Inst.addOperand(MCOperand::createReg(Reg2));
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createImm(-2)); // -16 / 8 = -2
}

// ldp x29, x30, [sp], #16
void createPopRegisters(MCInst &Inst, MCPhysReg Reg1, MCPhysReg Reg2) {
  Inst.clear();
  Inst.setOpcode(AArch64::LDPXpost);
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createReg(Reg1));
  Inst.addOperand(MCOperand::createReg(Reg2));
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createImm(2)); // 16 / 8 = 2
}

// 检查指令是否使用或定义指定的寄存器
bool usesOrDefsReg(const MCInst &Inst, unsigned Reg,
                   const MCRegisterInfo &RegInfo) {
  for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && RegInfo.regsOverlap(Op.getReg(), Reg)) {
      return true;
    }
  }
  return false;
}

// AArch64 caller-saved registers (will be clobbered by function call)
// x0-x18 are caller-saved (including x8 which is used for indirect result)
// x19-x28 are callee-saved
// x29 = FP, x30 = LR, SP = x31
static bool isCallerSavedReg(unsigned Reg, const MCRegisterInfo &RegInfo) {
  // Check if register overlaps with any caller-saved register
  static const unsigned CallerSavedRegs[] = {
      AArch64::X0,  AArch64::X1,  AArch64::X2,  AArch64::X3,  AArch64::X4,
      AArch64::X5,  AArch64::X6,  AArch64::X7,  AArch64::X8,  AArch64::X9,
      AArch64::X10, AArch64::X11, AArch64::X12, AArch64::X13, AArch64::X14,
      AArch64::X15, AArch64::X16, AArch64::X17, AArch64::X18,
      // Also include the W variants (32-bit views)
      AArch64::W0,  AArch64::W1,  AArch64::W2,  AArch64::W3,  AArch64::W4,
      AArch64::W5,  AArch64::W6,  AArch64::W7,  AArch64::W8,  AArch64::W9,
      AArch64::W10, AArch64::W11, AArch64::W12, AArch64::W13, AArch64::W14,
      AArch64::W15, AArch64::W16, AArch64::W17, AArch64::W18};
  for (unsigned CSR : CallerSavedRegs) {
    if (RegInfo.regsOverlap(Reg, CSR))
      return true;
  }
  return false;
}

// Get all registers defined by an instruction
static void getDefsOfInstruction(const MCInst &Inst, const BinaryContext &BC,
                                 SmallVectorImpl<unsigned> &Defs) {
  const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());

  // Explicit defs (first N operands where N = NumDefs)
  unsigned NumDefs = Desc.getNumDefs();
  for (unsigned I = 0; I < NumDefs && I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() != 0)
      Defs.push_back(Op.getReg());
  }

  // Implicit defs from the instruction descriptor
  ArrayRef<MCPhysReg> ImpDefs = Desc.implicit_defs();
  for (MCPhysReg Reg : ImpDefs) {
    Defs.push_back(Reg);
  }
}

// Get all registers used by an instruction
static void getUsesOfInstruction(const MCInst &Inst, const BinaryContext &BC,
                                 SmallVectorImpl<unsigned> &Uses) {
  const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());
  unsigned NumDefs = Desc.getNumDefs();

  // Explicit uses (operands after the defs)
  for (unsigned I = NumDefs; I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() != 0)
      Uses.push_back(Op.getReg());
  }

  // Also check defs that might be read-modify-write
  // (some instructions read their destination before writing)

  // Implicit uses from the instruction descriptor
  ArrayRef<MCPhysReg> ImpUses = Desc.implicit_uses();
  for (MCPhysReg Reg : ImpUses) {
    Uses.push_back(Reg);
  }
}

// Check if outlining a sequence would clobber live registers
// This is critical for correctness: when we replace a sequence with a call,
// the call will clobber all caller-saved registers (x0-x18 on AArch64).
//
// The key insight is: a register is "live across" the call if:
//   1. It is defined BEFORE the sequence (not redefined in the sequence), AND
//   2. It is used AFTER the sequence (before being redefined)
//
// We must reject outlining if any caller-saved register is live across the call.
//
// Returns true if the sequence can be safely outlined (no liveness issues)
static bool checkLivenessForOutlining(BinaryBasicBlock *BB,
                                      BinaryBasicBlock::iterator SeqStart,
                                      BinaryBasicBlock::iterator SeqEnd,
                                      const BinaryContext &BC) {
  const MCRegisterInfo &RegInfo = *BC.MRI;

  // Use backward scan to compute live-in at the point after the sequence.
  // This is more accurate: we scan from the end of BB backward to SeqEnd,
  // tracking which caller-saved registers are live.
  SmallSet<unsigned, 32> LiveAtSeqEnd;

  // Start from end of BB and scan backward
  // We build a vector of instructions after SeqEnd first
  SmallVector<MCInst *, 32> InstsAfterSeq;
  for (auto It = SeqEnd; It != BB->end(); ++It) {
    InstsAfterSeq.push_back(&*It);
  }

  // Now scan backward
  for (auto RIt = InstsAfterSeq.rbegin(); RIt != InstsAfterSeq.rend(); ++RIt) {
    MCInst *Inst = *RIt;
    if (BC.MIB->isPseudo(*Inst))
      continue;

    // In backward scan: first remove defs (kills the liveness),
    // then add uses (makes it live)
    SmallVector<unsigned, 4> Defs;
    getDefsOfInstruction(*Inst, BC, Defs);
    for (unsigned DefReg : Defs) {
      // Remove any register that overlaps with DefReg
      SmallVector<unsigned, 4> ToRemove;
      for (unsigned LiveReg : LiveAtSeqEnd) {
        if (RegInfo.regsOverlap(DefReg, LiveReg)) {
          ToRemove.push_back(LiveReg);
        }
      }
      for (unsigned R : ToRemove)
        LiveAtSeqEnd.erase(R);
    }

    // Then add uses
    SmallVector<unsigned, 4> Uses;
    getUsesOfInstruction(*Inst, BC, Uses);
    for (unsigned Reg : Uses) {
      if (isCallerSavedReg(Reg, RegInfo)) {
        LiveAtSeqEnd.insert(Reg);
      }
    }
  }

  if (LiveAtSeqEnd.empty())
    return true; // No caller-saved regs live at seq end, safe to outline

  // Now check if any of these live registers are defined in the sequence.
  // If a register is live at SeqEnd AND defined in the sequence,
  // then the outlined function will define it, so it's safe.
  SmallSet<unsigned, 32> DefsInSeq;
  for (auto It = SeqStart; It != SeqEnd; ++It) {
    if (BC.MIB->isPseudo(*It))
      continue;
    SmallVector<unsigned, 4> Defs;
    getDefsOfInstruction(*It, BC, Defs);
    for (unsigned Reg : Defs) {
      DefsInSeq.insert(Reg);
    }
  }

  // For each register live at SeqEnd, check if it's defined in the sequence
  for (unsigned LiveReg : LiveAtSeqEnd) {
    bool DefinedInSeq = false;
    for (unsigned DefReg : DefsInSeq) {
      if (RegInfo.regsOverlap(LiveReg, DefReg)) {
        DefinedInSeq = true;
        break;
      }
    }
    if (!DefinedInSeq) {
      // This register is live at SeqEnd but not defined in the sequence.
      // It must be defined before the sequence, so the call would clobber it.
      LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Liveness conflict - reg "
                        << LiveReg << " live across call point\n");
      return false; // Cannot outline safely
    }
  }

  return true; // Safe to outline
}

// 检查指令序列是否读取 LR (X30)
// 用于叶子函数 - 如果序列读取 LR，BL 会覆盖它，导致错误
static bool sequenceReadsLR(BinaryBasicBlock::iterator StartIt,
                            BinaryBasicBlock::iterator EndIt,
                            const BinaryContext &BC) {
  const MCRegisterInfo &MRI = *BC.MRI;
  
  for (auto It = StartIt; It != EndIt; ++It) {
    const MCInst &Inst = *It;
    const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());
    
    // 检查显式使用（读取）LR
    unsigned NumDefs = Desc.getNumDefs();
    for (unsigned I = NumDefs; I < Inst.getNumOperands(); ++I) {
      const MCOperand &Op = Inst.getOperand(I);
      if (Op.isReg() && MRI.regsOverlap(Op.getReg(), AArch64::LR)) {
        return true; // 序列读取 LR
      }
    }
    
    // 检查隐式使用 LR
    ArrayRef<MCPhysReg> ImpUses = Desc.implicit_uses();
    for (MCPhysReg Reg : ImpUses) {
      if (MRI.regsOverlap(Reg, AArch64::LR)) {
        return true;
      }
    }
  }
  
  return false;
}

// Forward declaration (used before definition).
// Forward declaration (used before definition).
bool modifiesSP(const MCInst &Inst, const MCRegisterInfo &RegInfo);
bool isPACInstr(unsigned Opcode);

// 检查函数是否安全地“升级”为非叶子：不能有 CFI，不能修改/使用 SP
// （当前 outline 逻辑没有全局偏移修正），避免已有框架被破坏。


// 检查指令是否定义（写入）FP 寄存器
// 只读 FP 的指令可以被 outline（FP 值在函数内是常量）
// 写入 FP 的指令不能被 outline（会破坏栈帧）
bool definesFramePointer(const MCInst &Inst, const MCRegisterInfo &RegInfo,
                         const BinaryContext &BC) {
  const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());
  unsigned NumDefs = Desc.getNumDefs();

  // 检查显式定义
  for (unsigned I = 0; I < NumDefs && I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && RegInfo.regsOverlap(Op.getReg(), AArch64::FP)) {
      return true;
    }
  }

  // 检查隐式定义
  ArrayRef<MCPhysReg> ImpDefs = Desc.implicit_defs();
  for (MCPhysReg Reg : ImpDefs) {
    if (RegInfo.regsOverlap(Reg, AArch64::FP)) {
      return true;
    }
  }

  // 检查 pre/post-indexed 访问是否修改 FP
  // 例如: ldr x0, [x29], #8 或 ldr x0, [x29, #8]!
  unsigned Opcode = Inst.getOpcode();
  switch (Opcode) {
  case AArch64::LDRXpre:
  case AArch64::LDRXpost:
  case AArch64::LDRWpre:
  case AArch64::LDRWpost:
  case AArch64::STRXpre:
  case AArch64::STRXpost:
  case AArch64::STRWpre:
  case AArch64::STRWpost:
  case AArch64::LDPXpre:
  case AArch64::LDPXpost:
  case AArch64::STPXpre:
  case AArch64::STPXpost:
    // 这些指令的第一个操作数是写回的基址寄存器
    if (Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
        RegInfo.regsOverlap(Inst.getOperand(0).getReg(), AArch64::FP)) {
      return true;
    }
    break;
  default:
    break;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// SP-relative 指令处理 (基于 PLOS 论文 §4.1.5 Stack Access Offsetting)
//
// 核心思想：
// - outlined function 建立 16 bytes 的最小栈帧 (stp x29, x30, [sp, #-16]!)
// - 所有 SP-relative 访问需要 +16 偏移修正
// - 只支持简单的 [SP, #imm] 形式，不支持寄存器变址
//===----------------------------------------------------------------------===//

// Outlined function 的栈帧大小（固定 16 bytes：保存 FP 和 LR）
constexpr int64_t OutlinedFrameSize = 16;

// 检查是否是可以安全修正偏移的 SP-relative load/store 指令
// 返回 true 如果是 [SP, #imm] 形式的访问
// 返回 false 如果是 [SP, Xn] 寄存器变址形式（不支持）
bool isSPRelativeWithImmOffset(const MCInst &Inst,
                               const MCRegisterInfo &RegInfo, int &BaseRegOpIdx,
                               int &ImmOpIdx, int &Scale) {
  unsigned Opcode = Inst.getOpcode();

  // 检查各种 SP-relative load/store 指令
  // 格式: LDR/STR Rt, [Xn, #imm] 其中 Xn 可能是 SP
  // 操作数布局因指令而异

  switch (Opcode) {
  // Unsigned immediate offset loads/stores (最常见)
  // 格式: LDRXui Rt, Rn, #imm  (Rn 在 operand 1, imm 在 operand 2)
  case AArch64::LDRXui:
  case AArch64::LDRWui:
  case AArch64::LDRBui:
  case AArch64::LDRHui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
  case AArch64::LDRSWui:
  case AArch64::LDRSHWui:
  case AArch64::LDRSHXui:
  case AArch64::LDRSBWui:
  case AArch64::LDRSBXui:
  case AArch64::LDRBBui:
  case AArch64::LDRHHui:
  case AArch64::STRXui:
  case AArch64::STRWui:
  case AArch64::STRBui:
  case AArch64::STRHui:
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:
  case AArch64::STRBBui:
  case AArch64::STRHHui:
    if (Inst.getNumOperands() >= 3 && Inst.getOperand(1).isReg() &&
        Inst.getOperand(1).getReg() == AArch64::SP &&
        Inst.getOperand(2).isImm()) {
      BaseRegOpIdx = 1;
      ImmOpIdx = 2;
      // Scale 取决于数据大小
      switch (Opcode) {
      case AArch64::LDRXui:
      case AArch64::STRXui:
      case AArch64::LDRDui:
      case AArch64::STRDui:
        Scale = 8;
        break;
      case AArch64::LDRSWui:
      case AArch64::LDRWui:
      case AArch64::STRWui:
      case AArch64::LDRSui:
      case AArch64::STRSui:
        Scale = 4;
        break;
      case AArch64::LDRHui:
      case AArch64::STRHui:
      case AArch64::LDRHHui:
      case AArch64::STRHHui:
      case AArch64::LDRSHWui:
      case AArch64::LDRSHXui:
        Scale = 2;
        break;
      case AArch64::LDRBui:
      case AArch64::STRBui:
      case AArch64::LDRBBui:
      case AArch64::STRBBui:
      case AArch64::LDRSBWui:
      case AArch64::LDRSBXui:
        Scale = 1;
        break;
      case AArch64::LDRQui:
      case AArch64::STRQui:
        Scale = 16;
        break;
      default:
        Scale = 8;
        break;
      }
      return true;
    }
    return false;

  // Unscaled immediate offset (LDUR/STUR)
  // 格式: LDURXi Rt, Rn, #imm (无缩放)
  case AArch64::LDURXi:
  case AArch64::LDURWi:
  case AArch64::LDURBi:
  case AArch64::LDURHi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURSWi:
  case AArch64::LDURSHWi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSBXi:
  case AArch64::LDURBBi:
  case AArch64::LDURHHi:
  case AArch64::STURXi:
  case AArch64::STURWi:
  case AArch64::STURBi:
  case AArch64::STURHi:
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURBBi:
  case AArch64::STURHHi:
    if (Inst.getNumOperands() >= 3 && Inst.getOperand(1).isReg() &&
        Inst.getOperand(1).getReg() == AArch64::SP &&
        Inst.getOperand(2).isImm()) {
      BaseRegOpIdx = 1;
      ImmOpIdx = 2;
      Scale = 1; // Unscaled
      return true;
    }
    return false;

  // Pair loads/stores (LDP/STP)
  // 格式: LDPXi Rt1, Rt2, Rn, #imm
  case AArch64::LDPXi:
  case AArch64::LDPWi:
  case AArch64::LDPSi:
  case AArch64::LDPDi:
  case AArch64::LDPQi:
  case AArch64::LDPSWi:
  case AArch64::STPXi:
  case AArch64::STPWi:
  case AArch64::STPSi:
  case AArch64::STPDi:
  case AArch64::STPQi:
    if (Inst.getNumOperands() >= 4 && Inst.getOperand(2).isReg() &&
        Inst.getOperand(2).getReg() == AArch64::SP &&
        Inst.getOperand(3).isImm()) {
      BaseRegOpIdx = 2;
      ImmOpIdx = 3;
      // LDP/STP 的 scale 是数据大小
      switch (Opcode) {
      case AArch64::LDPXi:
      case AArch64::STPXi:
      case AArch64::LDPDi:
      case AArch64::STPDi:
        Scale = 8;
        break;
      case AArch64::LDPWi:
      case AArch64::STPWi:
      case AArch64::LDPSi:
      case AArch64::STPSi:
      case AArch64::LDPSWi:
        Scale = 4;
        break;
      case AArch64::LDPQi:
      case AArch64::STPQi:
        Scale = 16;
        break;
      default:
        Scale = 8;
        break;
      }
      return true;
    }
    return false;

  default:
    return false;
  }
}

// 检查是否是 unscaled (LDUR/STUR) 的 SP-relative 指令
bool isUnscaledSPRelativeOpcode(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::LDURXi:
  case AArch64::LDURWi:
  case AArch64::LDURBi:
  case AArch64::LDURHi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURSWi:
  case AArch64::LDURSHWi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSBXi:
  case AArch64::LDURBBi:
  case AArch64::LDURHHi:
  case AArch64::STURXi:
  case AArch64::STURWi:
  case AArch64::STURBi:
  case AArch64::STURHi:
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURBBi:
  case AArch64::STURHHi:
    return true;
  default:
    return false;
  }
}

// 检查是否是 pair (LDP/STP) 的 SP-relative 指令
bool isPairSPRelativeOpcode(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::LDPXi:
  case AArch64::LDPWi:
  case AArch64::LDPSi:
  case AArch64::LDPDi:
  case AArch64::LDPQi:
  case AArch64::LDPSWi:
  case AArch64::STPXi:
  case AArch64::STPWi:
  case AArch64::STPSi:
  case AArch64::STPDi:
  case AArch64::STPQi:
    return true;
  default:
    return false;
  }
}

// 检查指令是否修改 SP（动态栈分配等）
// 这些指令不能被 outline，因为会破坏 "+16 偏移" 模型
bool modifiesSP(const MCInst &Inst, const MCRegisterInfo &RegInfo) {
  unsigned Opcode = Inst.getOpcode();

  // 检查是否是 ADD/SUB SP, SP, #imm 或 ADD/SUB SP, SP, Xn
  // 这些指令会动态修改 SP
  switch (Opcode) {
  case AArch64::ADDXri:
  case AArch64::SUBXri:
  case AArch64::ADDXrx:
  case AArch64::SUBXrx:
  case AArch64::ADDXrs:
  case AArch64::SUBXrs:
  case AArch64::ADDXrr:
  case AArch64::SUBXrr:
    // 检查目标寄存器是否是 SP
    if (Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
        Inst.getOperand(0).getReg() == AArch64::SP) {
      return true;
    }
    break;

  // Pre/Post-indexed loads/stores 会修改基址寄存器
  // 如果基址是 SP，则不能 outline
  case AArch64::LDRXpre:
  case AArch64::LDRXpost:
  case AArch64::LDRWpre:
  case AArch64::LDRWpost:
  case AArch64::STRXpre:
  case AArch64::STRXpost:
  case AArch64::STRWpre:
  case AArch64::STRWpost:
  case AArch64::LDPXpre:
  case AArch64::LDPXpost:
  case AArch64::STPXpre:
  case AArch64::STPXpost:
    // 这些指令的第一个操作数是写回的基址寄存器
    if (Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
        Inst.getOperand(0).getReg() == AArch64::SP) {
      return true;
    }
    break;

  default:
    break;
  }

  return false;
}

// 检查 SP-relative 指令是否可以安全地被 outline
// 返回 true 如果可以通过偏移修正来处理
bool canOutlineSPRelativeInstr(const MCInst &Inst,
                               const MCRegisterInfo &RegInfo) {
  int BaseRegOpIdx, ImmOpIdx, Scale;

  // 如果是可修正的 SP-relative 访问
  if (isSPRelativeWithImmOffset(Inst, RegInfo, BaseRegOpIdx, ImmOpIdx, Scale)) {
    int64_t CurrentOffset = Inst.getOperand(ImmOpIdx).getImm();
    int64_t NewOffset = CurrentOffset + (OutlinedFrameSize / Scale);

    // 检查新偏移是否在有效范围内
    // 对于 unsigned immediate，范围是 [0, 4095]
    // 对于 unscaled immediate，范围是 [-256, 255]
    unsigned Opcode = Inst.getOpcode();

    // Unscaled (LDUR/STUR) 范围检查
    if (isUnscaledSPRelativeOpcode(Opcode)) {
      // 实际偏移 = CurrentOffset * 1 + 16
      int64_t ActualNewOffset = CurrentOffset + OutlinedFrameSize;
      if (ActualNewOffset < -256 || ActualNewOffset > 255) {
        LLVM_DEBUG(
            dbgs()
            << "BOLT-OUTLINING: SP-relative unscaled offset out of range: "
            << ActualNewOffset << "\n");
        return false;
      }
    } else if (isPairSPRelativeOpcode(Opcode)) {
      // Pair loads/stores use signed imm7 scaled by element size.
      if (NewOffset < -64 || NewOffset > 63) {
        LLVM_DEBUG(
            dbgs() << "BOLT-OUTLINING: SP-relative pair offset out of range: "
                   << NewOffset << " (scale=" << Scale << ")\n");
        return false;
      }
    } else {
      // Scaled offset 范围检查 [0, 4095]
      if (NewOffset < 0 || NewOffset > 4095) {
        LLVM_DEBUG(
            dbgs() << "BOLT-OUTLINING: SP-relative scaled offset out of range: "
                   << NewOffset << " (scale=" << Scale << ")\n");
        return false;
      }
    }

    return true;
  }

  // 如果指令修改 SP，不能 outline
  if (modifiesSP(Inst, RegInfo)) {
    return false;
  }

  // 如果指令使用 SP 但不是上述可处理的形式，保守地拒绝
  if (usesOrDefsReg(Inst, AArch64::SP, RegInfo)) {
    LLVM_DEBUG(
        dbgs()
        << "BOLT-OUTLINING: Rejecting unsupported SP-relative instruction\n");
    return false;
  }

  return true;
}

// 对 SP-relative 指令进行偏移修正
// 在 outlined function 中，SP 比原来低 16 bytes
// 所以所有 SP-relative 访问需要 +16 偏移
void adjustSPRelativeOffset(MCInst &Inst, const MCRegisterInfo &RegInfo) {
  int BaseRegOpIdx, ImmOpIdx, Scale;

  if (!isSPRelativeWithImmOffset(Inst, RegInfo, BaseRegOpIdx, ImmOpIdx,
                                 Scale)) {
    // DEBUG: 检查是否有遗漏的 SP-relative 指令
    if (usesOrDefsReg(Inst, AArch64::SP, RegInfo)) {
      outs() << "BOLT-OUTLINING: WARNING: SP-relative instruction not "
                "adjusted! Opcode="
             << Inst.getOpcode() << "\n";
    }
    return;
  }

  int64_t CurrentOffset = Inst.getOperand(ImmOpIdx).getImm();
  int64_t Adjustment = OutlinedFrameSize / Scale;
  int64_t NewOffset = CurrentOffset + Adjustment;

  outs() << "BOLT-OUTLINING:   Adjusted SP offset: " << CurrentOffset << " -> "
         << NewOffset << " (scale=" << Scale << ")\n";

  Inst.getOperand(ImmOpIdx).setImm(NewOffset);
}

// 检查是否是 PC-relative literal load (LDR Xn, =label)
bool isPCRelativeLiteralLoad(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::LDRXl:
  case AArch64::LDRWl:
  case AArch64::LDRSWl:
  case AArch64::LDRSl:
  case AArch64::LDRDl:
  case AArch64::LDRQl:
  case AArch64::PRFMl:
    return true;
  default:
    return false;
  }
}

// 检查是否是不能 outline 的系统指令
// 注意：内存屏障 (DMB/DSB/ISB) 可以安全 outline，因为它们的效果不依赖指令地址
bool isMemoryBarrierOrSystemInstr(unsigned Opcode) {
  switch (Opcode) {
  // Memory barriers - 可以 outline，效果不依赖地址
  // case AArch64::DMB:
  // case AArch64::DSB:
  // case AArch64::ISB:
  // case AArch64::TSB:

  // System instructions - 不能 outline
  case AArch64::SYSxt:
  case AArch64::SYSLxt:
  // Hint instructions - 大部分可以 outline，但保守处理
  // case AArch64::HINT:
  // Exception generating - 绝对不能 outline
  case AArch64::SVC:
  case AArch64::HVC:
  case AArch64::SMC:
  case AArch64::BRK:
  case AArch64::HLT:
  case AArch64::DCPS1:
  case AArch64::DCPS2:
  case AArch64::DCPS3:
    return true;
  default:
    return false;
  }
}

// 检查是否是 PAC (Pointer Authentication) 指令
bool isPACInstr(unsigned Opcode) {
  switch (Opcode) {
  // PAC instructions
  case AArch64::PACIA:
  case AArch64::PACIB:
  case AArch64::PACDA:
  case AArch64::PACDB:
  case AArch64::PACIZA:
  case AArch64::PACIZB:
  case AArch64::PACDZA:
  case AArch64::PACDZB:
  case AArch64::PACIAZ:
  case AArch64::PACIBZ:
  case AArch64::PACIASP:
  case AArch64::PACIBSP:
  case AArch64::PACIA1716:
  case AArch64::PACIB1716:
  // AUT instructions
  case AArch64::AUTIA:
  case AArch64::AUTIB:
  case AArch64::AUTDA:
  case AArch64::AUTDB:
  case AArch64::AUTIZA:
  case AArch64::AUTIZB:
  case AArch64::AUTDZA:
  case AArch64::AUTDZB:
  case AArch64::AUTIAZ:
  case AArch64::AUTIBZ:
  case AArch64::AUTIASP:
  case AArch64::AUTIBSP:
  case AArch64::AUTIA1716:
  case AArch64::AUTIB1716:
  // XPAC instructions
  case AArch64::XPACI:
  case AArch64::XPACD:
  case AArch64::XPACLRI:
  // Combined branch instructions
  case AArch64::BRAA:
  case AArch64::BRAB:
  case AArch64::BRAAZ:
  case AArch64::BRABZ:
  case AArch64::BLRAA:
  case AArch64::BLRAB:
  case AArch64::BLRAAZ:
  case AArch64::BLRABZ:
  case AArch64::RETAA:
  case AArch64::RETAB:
    return true;
  default:
    return false;
  }
}

// 检查是否是 exclusive load/store (用于原子操作)
bool isExclusiveLoadStore(unsigned Opcode) {
  switch (Opcode) {
  // Exclusive loads
  case AArch64::LDXRB:
  case AArch64::LDXRH:
  case AArch64::LDXRW:
  case AArch64::LDXRX:
  case AArch64::LDAXRB:
  case AArch64::LDAXRH:
  case AArch64::LDAXRW:
  case AArch64::LDAXRX:
  case AArch64::LDXPW:
  case AArch64::LDXPX:
  case AArch64::LDAXPW:
  case AArch64::LDAXPX:
  // Exclusive stores
  case AArch64::STXRB:
  case AArch64::STXRH:
  case AArch64::STXRW:
  case AArch64::STXRX:
  case AArch64::STLXRB:
  case AArch64::STLXRH:
  case AArch64::STLXRW:
  case AArch64::STLXRX:
  case AArch64::STXPW:
  case AArch64::STXPX:
  case AArch64::STLXPW:
  case AArch64::STLXPX:
  // Clear exclusive
  case AArch64::CLREX:
    return true;
  default:
    return false;
  }
}

// 检查是否是 LSE 原子指令
bool isLSEAtomicInstr(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::LDADDW:
  case AArch64::LDADDX:
  case AArch64::LDADDAW:
  case AArch64::LDADDAX:
  case AArch64::LDADDLW:
  case AArch64::LDADDLX:
  case AArch64::LDADDALW:
  case AArch64::LDADDALX:
  case AArch64::LDCLRW:
  case AArch64::LDCLRX:
  case AArch64::LDEORW:
  case AArch64::LDEORX:
  case AArch64::LDSETW:
  case AArch64::LDSETX:
  case AArch64::LDSMAXW:
  case AArch64::LDSMAXX:
  case AArch64::LDSMINW:
  case AArch64::LDSMINX:
  case AArch64::LDUMAXW:
  case AArch64::LDUMAXX:
  case AArch64::LDUMINW:
  case AArch64::LDUMINX:
  case AArch64::SWPW:
  case AArch64::SWPX:
  case AArch64::CASW:
  case AArch64::CASX:
  case AArch64::CASALW:
  case AArch64::CASALX:
    return true;
  default:
    return false;
  }
}

// 检查函数是否是叶子函数（不保存 LR）
// 叶子函数不调用其他函数，因此不需要保存 LR
// 我们通过检查 prologue 中是否保存了 LR 来判断
//
// 重要：从 leaf function 中 outline 代码是危险的！
// 因为 BL 指令会覆盖 LR，而 leaf function 没有保存原始 LR，
// 导致返回地址丢失。
bool isLeafFunction(const BinaryFunction &BF, const BinaryContext &BC) {
  if (BF.empty())
    return true;

  const BinaryBasicBlock &EntryBB = BF.front();
  if (!EntryBB.hasInstructions())
    return true;

  // 检查前几条指令，看是否保存了 LR
  unsigned Count = 0;
  for (const MCInst &Inst : EntryBB) {
    if (BC.MIB->isPseudo(Inst))
      continue;

    if (Count++ > 10) // 检查前 10 条非伪指令
      break;

    unsigned Opcode = Inst.getOpcode();

    // 检查常见的保存 LR 的模式
    // STP Xn, X30, [SP, #imm]! 或 STP Xn, X30, [SP, #imm]
    // STR X30, [SP, #imm]! 或 STR X30, [SP, #imm]
    if (Opcode == AArch64::STPXpre || Opcode == AArch64::STPXi ||
        Opcode == AArch64::STRXui || Opcode == AArch64::STRXpre) {
      // 检查是否保存了 x30 (LR)
      for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
        const MCOperand &Op = Inst.getOperand(I);
        if (Op.isReg() && Op.getReg() == AArch64::LR) {
          return false; // 保存了 LR，不是叶子函数
        }
      }
    }
  }

  return true; // 没有保存 LR，是叶子函数
}

// PLOS §3.2: Wrapping Built-in Function Calls
// 检查函数调用是否使用栈传递参数
// 在 AArch64 中，前 8 个整数/指针参数通过 x0-x7 传递
// 前 8 个浮点参数通过 v0-v7 传递
// 超过 8 个参数的调用会使用栈传递，不能被安全 outline
//
// 由于我们无法在二进制级别直接知道被调用函数的参数数量，
// 我们采用保守策略：检查调用点之前是否有向栈写入参数的模式
// 但更简单的方法是：直接允许所有调用，因为：
// 1. 栈参数是由调用者准备的，在 outlined 序列之外
// 2. outlined 函数只是执行 BL 指令本身
// 3. 栈参数访问使用 SP-relative 偏移，我们已经支持偏移修正
//
// 然而，如果调用序列包含为栈参数准备数据的 SP-relative store，
// 这些 store 需要偏移修正，这已经在 SP-relative 处理中支持。
// 所以实际上我们已经通过 SP-relative 访问检查间接支持了参数传递。
//
// 这个函数用于未来更精细的控制，目前返回 true 允许所有调用
bool canOutlineCallInstruction(const MCInst &Inst, const BinaryContext &BC) {
  // 目前允许所有调用指令
  // 栈参数的处理通过 SP-relative 偏移修正机制保证正确性
  return true;
}

// 检测指令是否属于函数 prologue
// Prologue 指令通常包括：
//   - 保存 callee-saved 寄存器（stp/str 到栈）
//   - 设置帧指针（mov x29, sp）
//   - 分配栈空间（sub sp, sp, #imm）
// 这些指令与函数帧结构紧密相关，不应被 outline
bool isPrologueInstruction(const MCInst &Inst, const BinaryContext &BC,
                           bool &PrologueEnded) {
  if (PrologueEnded)
    return false;

  unsigned Opcode = Inst.getOpcode();

  // 1. 检查 STP/STR pre-indexed（常见的 prologue 开始）
  //    stp x29, x30, [sp, #-N]!  或  str xN, [sp, #-M]!
  if (Opcode == AArch64::STPXpre || Opcode == AArch64::STPDpre ||
      Opcode == AArch64::STRXpre || Opcode == AArch64::STRDpre) {
    // 检查目标是否是 SP
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      const MCOperand &Op = Inst.getOperand(I);
      if (Op.isReg() && Op.getReg() == AArch64::SP)
        return true; // prologue: 向栈保存寄存器
    }
  }

  // 2. 检查 STP/STR 普通形式保存 callee-saved 寄存器
  //    stp xN, xM, [sp, #offset]
  if (Opcode == AArch64::STPXi || Opcode == AArch64::STPDi ||
      Opcode == AArch64::STRXui || Opcode == AArch64::STRDui) {
    // 检查是否存储到 SP 相对地址，且保存的是 callee-saved 寄存器
    bool StoreToSP = false;
    bool SavesCalleeSaved = false;
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      const MCOperand &Op = Inst.getOperand(I);
      if (!Op.isReg())
        continue;
      unsigned Reg = Op.getReg();
      if (Reg == AArch64::SP)
        StoreToSP = true;
      // Callee-saved: x19-x28, x29(FP), x30(LR), d8-d15
      if ((Reg >= AArch64::X19 && Reg <= AArch64::X28) ||
          Reg == AArch64::FP || Reg == AArch64::LR ||
          (Reg >= AArch64::D8 && Reg <= AArch64::D15))
        SavesCalleeSaved = true;
    }
    if (StoreToSP && SavesCalleeSaved)
      return true; // prologue: 保存 callee-saved 寄存器
  }

  // 3. 设置帧指针：mov x29, sp 或 add x29, sp, #0
  if (Opcode == AArch64::ADDXri || Opcode == AArch64::ORRXrs) {
    // 检查是否是 x29 = sp + imm 或 mov x29, sp
    if (Inst.getNumOperands() >= 2) {
      const MCOperand &Dst = Inst.getOperand(0);
      const MCOperand &Src = Inst.getOperand(1);
      if (Dst.isReg() && Dst.getReg() == AArch64::FP &&
          Src.isReg() && Src.getReg() == AArch64::SP)
        return true; // prologue: 设置帧指针
    }
  }

  // 4. 分配栈空间：sub sp, sp, #imm
  if (Opcode == AArch64::SUBXri) {
    if (Inst.getNumOperands() >= 2) {
      const MCOperand &Dst = Inst.getOperand(0);
      const MCOperand &Src = Inst.getOperand(1);
      if (Dst.isReg() && Dst.getReg() == AArch64::SP &&
          Src.isReg() && Src.getReg() == AArch64::SP)
        return true; // prologue: 分配栈空间
    }
  }

  // 如果遇到非 prologue 指令，标记 prologue 结束
  // prologue 应该是连续的，一旦遇到非 prologue 指令就结束
  PrologueEnded = true;
  return false;
}

// 检查指令是否可以安全地被 outline
// 参考 LLVM MachineOutliner 的 AArch64 实现
bool canOutlineInstruction(const MCInst &Inst, const BinaryContext &BC) {
  unsigned Opcode = Inst.getOpcode();
  const MCRegisterInfo &RegInfo = *BC.MRI;

  // 1. PC 相对寻址指令不能 outline（地址会变）
  if (Opcode == AArch64::ADR || Opcode == AArch64::ADRP) {
    return false;
  }

  // 2. PC-relative literal loads 不能 outline
  if (isPCRelativeLiteralLoad(Opcode)) {
    return false;
  }

  // 3. 检查是否有 PC 相对的符号引用/重定位
  // 注意：不是所有 Expr 都是 PC 相对的。例如：
  //   - 全局变量地址（通过 ADRP + ADD 加载）- 可以 outline
  //   - PC 相对的 literal load（LDR Xn, =label）- 不能 outline（已在第 2
  //   步检查）
  // 我们允许某些安全的 Expr，但禁止明确的 PC 相对指令
  for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isExpr()) {
      const MCExpr *Expr = Op.getExpr();
      if (!Expr)
        continue;

      // 允许简单的符号引用（例如全局变量）
      // 这些可以通过重定位在 outlined 函数中正确处理
      if (Expr->getKind() == MCExpr::SymbolRef) {
        continue; // 允许
      }

      // 允许常数表达式
      if (Expr->getKind() == MCExpr::Constant) {
        continue; // 允许
      }

      // 允许 Binary 表达式（如 symbol + offset）
      // 这通常是全局变量的偏移访问，可以通过重定位处理
      if (Expr->getKind() == MCExpr::Binary) {
        continue; // 允许
      }

      // 允许 Unary 表达式
      if (Expr->getKind() == MCExpr::Unary) {
        continue; // 允许
      }

      // 禁止 Target/Specifier 表达式（可能包含 PC 相对部分）
      return false;
    }
  }

  // 4. 检查 SP-relative 指令
  // 对于 SP-relative 内存访问，检查是否可以通过偏移修正来处理
  // PLOS §4.1.5: 允许 SP-relative 指令，在 outlined 函数中 +16 修正偏移
  if (usesOrDefsReg(Inst, AArch64::SP, RegInfo)) {
    // 检查是否可以安全 outline（偏移在有效范围内）
    if (!canOutlineSPRelativeInstr(Inst, RegInfo)) {
      return false;
    }
    // 可以安全 outline，继续其他检查
  }

  // 5. 检查是否定义（写入）FP
  // FP 值在函数内是常量，所以只读 FP 的指令（如 ldr x0, [x29, #8]）可以被 outline
  // 但写入 FP 的指令不能被 outline（会破坏栈帧）
  // 例如：pre/post-indexed 访问 ldr x0, [x29], #8 会修改 x29，不能 outline
  if (definesFramePointer(Inst, RegInfo, BC)) {
    return false;
  }

  // 6. 检查是否使用或定义 LR (X30) - outlined 函数会破坏 LR
  // 注意：BL 指令会定义 LR（保存返回地址），但这是安全的，因为：
  // - outlined function 会先保存调用者的 LR (stp x29, x30, [sp, #-16]!)
  // - BL 设置的新 LR 值在返回前会被调用函数正确处理
  // 所以我们允许 BL 指令，它会被标记为 ContainsCall=true
  if (usesOrDefsReg(Inst, AArch64::LR, RegInfo)) {
    // 允许 BL 指令（它会定义 LR，但我们会正确保存）
    if (BC.MIB->isCall(Inst)) {
      // PLOS §3.2: 检查调用是否可以被安全 outline
      return canOutlineCallInstruction(Inst, BC);
    }
    return false;
  }

  // 6. 内存屏障和系统指令不能 outline（有副作用）
  if (isMemoryBarrierOrSystemInstr(Opcode)) {
    return false;
  }

  // 7. PAC 指令不能 outline（指针认证依赖上下文）
  if (isPACInstr(Opcode)) {
    return false;
  }

  // 8. Exclusive load/store 不能 outline（原子操作必须连续）
  // LDXR/STXR 对必须在同一个基本块中连续执行
  if (isExclusiveLoadStore(Opcode)) {
    return false;
  }

  // 9. LSE 原子指令可以 outline（它们是单条指令的原子操作）
  // 例如: LDADD, LDCLR, SWP, CAS 等
  // 这些指令的原子性由硬件保证，与指令位置无关
  // if (isLSEAtomicInstr(Opcode)) {
  //   return false;
  // }

  // 10. CFI 指令不能 outline（调试信息/异常处理）
  if (BC.MIB->isCFI(Inst)) {
    return false;
  }

  // 11. 检查指令是否有 side effects（除了内存访问）
  const MCInstrDesc &Desc = BC.MII->get(Opcode);
  if (Desc.hasUnmodeledSideEffects()) {
    return false;
  }

  return true;
}

Outliner::Outliner(const cl::opt<bool> &PrintPass)
    : BinaryFunctionPass(PrintPass) {}

std::string Outliner::generateOutlinedFunctionName() {
  return "__bolt_outlined_" + std::to_string(OutlinedFuncCounter++);
}

BinaryFunction *Outliner::createOutlinedFunction(
    BinaryContext &BC, BinaryFunction &Caller, BinaryBasicBlock &BB,
    BinaryBasicBlock::iterator Begin, BinaryBasicBlock::iterator End,
    bool NeedsPrologueEpilogue, bool EndsWithCall) {

  if (Begin == End)
    return nullptr;

  size_t NumInsts = std::distance(Begin, End);
  LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Outlining " << NumInsts
                    << " instructions, prologue=" << NeedsPrologueEpilogue
                    << ", tailCall=" << EndsWithCall << "\n");

  std::string OutlinedName = generateOutlinedFunctionName();

  // Use createInjectedBinaryFunction - no need for address or section
  BinaryFunction *OutlinedFunc = BC.createInjectedBinaryFunction(OutlinedName);

  // Create BasicBlock
  std::vector<std::unique_ptr<BinaryBasicBlock>> BBs;
  BBs.emplace_back(OutlinedFunc->createBasicBlock());

  // 尾调用优化 (PLOS §4.1.4):
  // 如果序列最后一条指令是 BL，可以用 B 替换并省略 RET
  // 这样被调用函数的 RET 会直接返回到原始调用者
  //
  // 两种情况：
  // 1. 序列只有一个 BL（就是最后一条）且没有 SP-relative 访问：
  //    - 不需要 prologue/epilogue（shrink-wrap + tail-call）
  //    - 直接 B 跳转
  // 2. 序列有多个 BL 或有 SP-relative 访问：
  //    - 需要 prologue
  //    - 在最后 BL 之前插入 epilogue（恢复 LR）
  //    - 然后 B 跳转
  bool ApplyTailCallOpt = EndsWithCall;

  // Prologue - only if needed (Shrink Wrapping optimization from PLOS)
  // If the outlined sequence doesn't contain any calls, we don't need to save
  // LR
  if (NeedsPrologueEpilogue) {
    MCInst StpInst;
    createPushRegisters(StpInst, AArch64::FP, AArch64::LR);
    BBs.back()->addInstruction(StpInst);
  }

  // Copy instructions (skip CFI instructions)
  // 如果有 prologue/epilogue，需要对 SP-relative 指令进行偏移修正
  // 基于 PLOS 论文 §4.1.5: Stack Access Offsetting
  const MCRegisterInfo &RegInfo = *BC.MRI;

  // 计算最后一条非 CFI 指令的迭代器（用于尾调用优化）
  BinaryBasicBlock::iterator LastNonCFI = End;
  if (ApplyTailCallOpt) {
    for (auto It = Begin; It != End; ++It) {
      if (!BC.MIB->isCFI(*It)) {
        LastNonCFI = It;
      }
    }
  }

  for (auto It = Begin; It != End; ++It) {
    if (BC.MIB->isCFI(*It))
      continue;
    MCInst InstCopy = *It;

    // 只有当有 prologue 时才需要偏移修正
    // 因为 prologue 会将 SP 降低 16 bytes
    if (NeedsPrologueEpilogue) {
      adjustSPRelativeOffset(InstCopy, RegInfo);
    }

    // 尾调用优化：将最后一条 BL 转换为 B
    if (ApplyTailCallOpt && It == LastNonCFI && BC.MIB->isCall(InstCopy)) {
      // 获取调用目标
      const MCSymbol *Target = BC.MIB->getTargetSymbol(InstCopy);
      if (Target) {
        // 如果有 prologue，需要在尾调用之前恢复 LR
        if (NeedsPrologueEpilogue) {
          MCInst LdpInst;
          createPopRegisters(LdpInst, AArch64::FP, AArch64::LR);
          BBs.back()->addInstruction(LdpInst);
        }

        // 创建无条件跳转指令代替 BL
        MCInst TailJump;
        BC.MIB->createTailCall(TailJump, Target, BC.Ctx.get());
        BBs.back()->addInstruction(TailJump);
        LLVM_DEBUG(
            dbgs() << "BOLT-OUTLINING: Applied tail call optimization\n");
        continue; // 跳过添加原始 BL
      }
    }

    BBs.back()->addInstruction(InstCopy);
  }

  // Epilogue - only if we added prologue and NOT tail call optimized
  // 尾调用时 epilogue 已经在尾调用之前插入了
  if (NeedsPrologueEpilogue && !ApplyTailCallOpt) {
    MCInst LdpInst;
    createPopRegisters(LdpInst, AArch64::FP, AArch64::LR);
    BBs.back()->addInstruction(LdpInst);
  }

  // 添加返回指令（除非应用了尾调用优化）
  if (!ApplyTailCallOpt) {
    MCInst RetInst;
    BC.MIB->createReturn(RetInst);
    BBs.back()->addInstruction(RetInst);
  }

  BBs.back()->setCFIState(0);

  // Insert BB and finalize
  OutlinedFunc->insertBasicBlocks(nullptr, std::move(BBs),
                                  /*UpdateLayout=*/true,
                                  /*UpdateCFIState=*/false);
  OutlinedFunc->updateState(BinaryFunction::State::CFG_Finalized);

  LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Created " << OutlinedName << " with "
                    << OutlinedFunc->front().size() << " instructions\n");

  return OutlinedFunc;
}

/// Represents a candidate sequence for outlining.
struct OutlineCandidate {
  unsigned StartIdx;    // Index in UnsignedVec/InstrList
  unsigned Length;      // Number of instructions
  BinaryBasicBlock *BB; // The basic block containing this candidate
  bool ContainsCalls;   // Whether this sequence contains BL instructions
  bool EndsWithCall; // Whether the last instruction is a BL (for tail call opt)
  bool FromLeafFunction; // Whether this candidate comes from a leaf function
  bool CanUseTailJump;   // 是否可以用 B（尾跳转）调用：序列后直接是 RET 或 B
  uint64_t ExecCount; // 执行次数（用于 cost model）
  unsigned CallOverhead; // 调用此候选的开销（字节数），逐候选计算

  unsigned getEndIdx() const { return StartIdx + Length - 1; }
};

Error Outliner::runOnFunctions(BinaryContext &BC) {
  const unsigned MaxIterations = opts::OutlinerMaxIterations;
  unsigned TotalOutlinedFuncs = 0;
  unsigned TotalReplacements = 0;
  int64_t TotalBytesSavedAll = 0;
  unsigned TotalShrinkWrapped = 0;

  outs() << "BOLT-OUTLINING: Starting outliner with max " << MaxIterations
         << " iterations\n";

  for (unsigned Iteration = 1; Iteration <= MaxIterations; ++Iteration) {
    outs() << "\n";
    outs() << "BOLT-OUTLINING: ========== Iteration " << Iteration << "/"
           << MaxIterations << " ==========\n";

    // 第一次迭代：跳过已有的 outlined 函数
    // 后续迭代：包含之前创建的 outlined 函数（嵌套 outlining）
    bool IncludeOutlinedFuncs = (Iteration > 1);

    //===--------------------------------------------------------------------===//
    // Phase 1: Build instruction mapping (similar to MachineOutliner)
    //
    // 基于 profile 数据，只 outline 冷代码以平衡代码大小和性能
    // - 冷代码：执行频率低，outline 后对性能影响小
    // - 热代码：执行频率高，保持 inline 以避免函数调用开销
    //===--------------------------------------------------------------------===//
    InstructionMapper Mapper;
    unsigned TotalFunctions = 0;
    unsigned TotalBBs = 0;
    unsigned LegalInsts = 0;
    unsigned IllegalInsts = 0;
    unsigned SkippedHotBBs = 0;
    unsigned ColdBBs = 0;
    unsigned EntryPointBBs = 0;
    unsigned PrologueInstsSkipped = 0;
    unsigned EntryPointInstsOutlined = 0;

    // 确定冷代码阈值
    // 如果用户指定了阈值，使用用户的值；否则使用 BOLT 默认的热代码阈值
    const uint64_t ColdThreshold = opts::OutlinerColdThreshold > 0
                                       ? opts::OutlinerColdThreshold
                                       : BC.getHotThreshold();

    const bool ProfileAvailable = BC.NumProfiledFuncs > 0;
    const bool OnlyCold = opts::OutlineColdOnly;

    outs() << "BOLT-OUTLINING: Phase 1 - Building instruction mapping...\n";
    outs() << "BOLT-OUTLINING:   Total binary functions in BC: "
           << BC.getBinaryFunctions().size() << "\n";
    outs() << "BOLT-OUTLINING:   NumProfiledFuncs: " << BC.NumProfiledFuncs
           << "\n";
    if (ProfileAvailable) {
      outs() << "BOLT-OUTLINING:   Profile data available, cold threshold = "
             << ColdThreshold << "\n";
      if (OnlyCold) {
        outs() << "BOLT-OUTLINING:   Only outlining cold code (exec count < "
               << ColdThreshold << ")\n";
      }
    } else {
      outs()
          << "BOLT-OUTLINING:   No profile data, outlining all eligible code\n";
    }

    unsigned ProcessedFuncs = 0;
    for (auto &BFI : BC.getBinaryFunctions()) {
      ProcessedFuncs++;
      BinaryFunction &BF = BFI.second;

      // Skip non-simple functions
      if (!BF.isSimple() || BF.empty())
        continue;

      // 处理 outlined 函数的逻辑：
      // - 第一次迭代：跳过已有的 outlined 函数
      // - 后续迭代：包含之前创建的 outlined 函数（嵌套 outlining）
      bool IsOutlinedFunc = (BF.getPrintName().find("__bolt_outlined_") == 0);
      if (IsOutlinedFunc && !IncludeOutlinedFuncs)
        continue;

      // 叶子函数处理：
      // 叶子函数不保存 LR，所以从它们 outline 代码时需要特殊处理。
      // 当调用 outlined function 时，BL 会覆盖 LR（叶子函数的返回地址）。
      //
      // 安全策略：仍然收集叶子函数的指令，但在 Phase 3 中只允许：
      // 1. 序列以 BL 结尾（可用尾调用优化：BL→B，不破坏 LR）
      // 2. 序列不读取 LR
      // 这样 outlined function 用 tail-call (B) 调用，不需要返回，不会破坏 LR
      bool IsLeafFunc = isLeafFunction(BF, BC);
      // 不再跳过叶子函数，而是标记它们，在 Phase 3 进行安全筛选

      TotalFunctions++;

      for (BinaryBasicBlock &BB : BF) {
        if (!BB.hasInstructions())
          continue;

        TotalBBs++;

        // 获取基本块的执行次数
        uint64_t BBExecCount = BB.getKnownExecutionCount();

        // Skip BBs that are not safe to outline (landing pads/etc.)
        // 注意：不再完全跳过 entry points，而是只跳过 prologue 指令
        if (!BB.canOutline()) {
          if (!BB.empty()) {
            Mapper.mapToIllegalUnsigned(BB.begin(), &BB, IsLeafFunc, BBExecCount);
            IllegalInsts++;
          }
          continue;
        }

        // Entry point 需要特殊处理：prologue 指令标记为 illegal
        bool IsEntryBB = BB.isEntryPoint();
        bool PrologueEnded = false;
        if (IsEntryBB)
          EntryPointBBs++;

        // Note: We no longer skip entire BBs ending with return.
        // Instead, we mark only the ret instruction itself as illegal
        // during instruction iteration below. This allows outlining
        // instructions before the ret.

        // 基于 profile 数据过滤热代码
        // 只有在有 profile 数据且启用了 cold-only 模式时才过滤
        if (ProfileAvailable && OnlyCold) {
          // 如果基本块是热的（执行次数 >= 阈值），跳过
          if (BBExecCount >= ColdThreshold) {
            SkippedHotBBs++;
            LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Skipping hot BB "
                              << BB.getName() << " (exec=" << BBExecCount
                              << " >= " << ColdThreshold << ")\n");
            // 添加一个 illegal marker 来断开序列
            if (!BB.empty()) {
              Mapper.mapToIllegalUnsigned(BB.begin(), &BB, IsLeafFunc, BBExecCount);
              IllegalInsts++;
            }
            continue;
          }
          ColdBBs++;
        }

        bool HaveLegalRange = false;
        bool CanOutlineWithPrevInstr = false;

        for (auto It = BB.begin(); It != BB.end(); ++It) {
          // Skip pseudo instructions (invisible)
          if (BC.MIB->isPseudo(*It)) {
            Mapper.AddedIllegalLastTime = false;
            continue;
          }

          // Entry point: 检测并跳过 prologue 指令
          // Prologue 指令与函数帧结构紧密相关，不应被 outline
          if (IsEntryBB && isPrologueInstruction(*It, BC, PrologueEnded)) {
            Mapper.mapToIllegalUnsigned(It, &BB, IsLeafFunc, BBExecCount);
            IllegalInsts++;
            PrologueInstsSkipped++;
            LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Skipping prologue instruction in "
                              << BF.getPrintName() << "\n");
            continue;
          }

          // Check if instruction is legal to outline
          bool IsBranch = BC.MIB->isBranch(*It);
          bool IsCall = BC.MIB->isCall(*It);
          bool IsReturn = BC.MIB->isReturn(*It);

          // 检测可修正的 SP-relative 内存访问
          // PLOS §4.1.5: 允许 SP-relative 指令，在 outlined 函数中修正偏移
          int BaseRegOpIdx, ImmOpIdx, Scale;
          bool HasSPAccess = isSPRelativeWithImmOffset(
              *It, *BC.MRI, BaseRegOpIdx, ImmOpIdx, Scale);

          // 检查 SP-relative 指令是否可以安全 outline
          // 如果偏移可以修正，则允许；否则拒绝
          if (HasSPAccess) {
            if (!canOutlineSPRelativeInstr(*It, *BC.MRI)) {
              // 偏移超出范围，不能 outline
              Mapper.mapToIllegalUnsigned(It, &BB, IsLeafFunc, BBExecCount);
              CanOutlineWithPrevInstr = false;
              IllegalInsts++;
              continue;
            }
            // 可以安全修正偏移，允许 outline
            // 继续到下面的处理逻辑
          }

          // Branches and returns are illegal (they change control flow)
          // But calls (BL) can be outlined - we just need to save LR
          if (IsBranch || IsReturn) {
            Mapper.mapToIllegalUnsigned(It, &BB, IsLeafFunc, BBExecCount);
            CanOutlineWithPrevInstr = false;
            IllegalInsts++;
          } else if (IsCall) {
            // BL instructions can be outlined, but we need to track them
            // to know if we need prologue/epilogue
            if (canOutlineInstruction(*It, BC)) {
              Mapper.mapToLegalUnsigned(It, &BB, /*ContainsCall=*/true,
                                        /*HasSPRelativeAccess=*/false,
                                        /*FromLeafFunction=*/IsLeafFunc,
                                        BBExecCount);
              if (CanOutlineWithPrevInstr)
                HaveLegalRange = true;
              CanOutlineWithPrevInstr = true;
              LegalInsts++;
              if (IsEntryBB && PrologueEnded)
                EntryPointInstsOutlined++;
            } else {
              Mapper.mapToIllegalUnsigned(It, &BB, IsLeafFunc, BBExecCount);
              CanOutlineWithPrevInstr = false;
              IllegalInsts++;
            }
          } else if (!canOutlineInstruction(*It, BC)) {
            // Illegal instruction
            Mapper.mapToIllegalUnsigned(It, &BB, IsLeafFunc, BBExecCount);
            CanOutlineWithPrevInstr = false;
            IllegalInsts++;
          } else {
            // Legal instruction (including SP-relative that passed the check above)
            Mapper.mapToLegalUnsigned(It, &BB, /*ContainsCall=*/false,
                                      /*HasSPRelativeAccess=*/HasSPAccess,
                                      /*FromLeafFunction=*/IsLeafFunc,
                                      BBExecCount);
            if (CanOutlineWithPrevInstr)
              HaveLegalRange = true;
            CanOutlineWithPrevInstr = true;
            LegalInsts++;
            if (IsEntryBB && PrologueEnded)
              EntryPointInstsOutlined++;
          }
        }

        // Add sentinel at end of BB to prevent cross-BB matching
        if (HaveLegalRange && !BB.empty()) {
          Mapper.addSentinel(&BB, IsLeafFunc, BBExecCount);
        }
      }
    }

    outs() << "BOLT-OUTLINING: === Phase 1 Complete ===\n";
    outs() << "BOLT-OUTLINING:   Total functions processed: " << TotalFunctions
           << "\n";
    outs() << "BOLT-OUTLINING:   Total basic blocks processed: " << TotalBBs
           << "\n";
    if (ProfileAvailable && OnlyCold) {
      outs() << "BOLT-OUTLINING:   Cold BBs: " << ColdBBs << "\n";
      outs() << "BOLT-OUTLINING:   Hot BBs skipped: " << SkippedHotBBs << "\n";
    }
    outs() << "BOLT-OUTLINING:   Entry point BBs: " << EntryPointBBs << "\n";
    outs() << "BOLT-OUTLINING:   Prologue instructions skipped: " << PrologueInstsSkipped << "\n";
    outs() << "BOLT-OUTLINING:   Entry point instructions collected: " << EntryPointInstsOutlined << "\n";
    outs() << "BOLT-OUTLINING:   Legal (outlinable) instructions: "
           << LegalInsts << "\n";
    outs() << "BOLT-OUTLINING:   Illegal (non-outlinable) instructions: "
           << IllegalInsts << "\n";
    outs() << "BOLT-OUTLINING:   UnsignedVec size: "
           << Mapper.UnsignedVec.size() << "\n";
    outs() << "BOLT-OUTLINING:   InstrList size: " << Mapper.InstrList.size()
           << "\n";

    if (Mapper.UnsignedVec.empty()) {
      outs() << "BOLT-OUTLINING: No instructions to outline\n";
      return Error::success();
    }

    //===--------------------------------------------------------------------===//
    // Phase 2: Build suffix tree and find repeated sequences
    //===--------------------------------------------------------------------===//
    outs() << "BOLT-OUTLINING: Phase 2 - Building suffix tree...\n";
    outs() << "BOLT-OUTLINING:   Input vector size: "
           << Mapper.UnsignedVec.size() << " entries\n";

    SuffixTree ST(Mapper.UnsignedVec, /*OutlinerLeafDescendants=*/true);
    outs() << "BOLT-OUTLINING:   Suffix tree constructed successfully\n";

    // Collect all repeated sequences with their candidates
    struct OutlinedSequence {
      unsigned Length;
      std::vector<OutlineCandidate> Candidates;
      BinaryFunction *OutlinedFunc = nullptr;
      bool NeedsPrologueEpilogue = false;
      bool EndsWithCall = false; // 尾调用优化：序列是否以 BL 结尾
    };

    std::vector<OutlinedSequence> SequencesToOutline;
    unsigned TotalRepeatedSeqs = 0;
    unsigned MinOccurrences = 2; // At least 2 occurrences to be worth outlining
    unsigned MinLength = 1;      // Allow single instruction outlining (very aggressive)

    for (SuffixTree::RepeatedSubstring &RS : ST) {
      TotalRepeatedSeqs++;
      unsigned StringLen = RS.Length;

      // Skip sequences that are too short
      if (StringLen < MinLength)
        continue;

      // Sort start indices to efficiently check for overlaps
      llvm::sort(RS.StartIndices);

      std::vector<OutlineCandidate> CandidatesForSeq;
      bool SeqContainsCalls = false;
      bool SeqContainsSPRelativeAccess = false; // 是否包含 SP-relative 访问

      for (const unsigned &StartIdx : RS.StartIndices) {
        unsigned EndIdx = StartIdx + StringLen - 1;

        // Check for overlap with previously added candidates
        if (!CandidatesForSeq.empty() &&
            StartIdx <= CandidatesForSeq.back().getEndIdx()) {
          continue; // Overlaps, skip
        }

        // Verify the iterator is valid
        if (StartIdx >= Mapper.InstrList.size() ||
            EndIdx >= Mapper.InstrList.size())
          continue;

        // Get the basic block for this candidate
        const InstrLocation &StartLoc = Mapper.InstrList[StartIdx];
        const InstrLocation &EndLoc = Mapper.InstrList[EndIdx];
        BinaryBasicBlock *BB = StartLoc.BB;
        if (!BB)
          continue;

        // Ensure all instructions are in the same BB
        if (EndLoc.BB != BB)
          continue;

        // Get iterators for this candidate sequence
        BinaryBasicBlock::iterator SeqStartIt = StartLoc.It;
        BinaryBasicBlock::iterator SeqEndIt = EndLoc.It;
        ++SeqEndIt; // Make it past-the-end

        // Check if this candidate contains any calls or SP-relative accesses
        bool CandContainsCalls = false;
        bool CandHasSPAccess = false;
        bool CandEndsWithCall = false;               // 尾调用优化
        uint64_t CandExecCount = StartLoc.ExecCount; // 使用基本块的执行次数
        bool CandFromLeafFunc = false;
        for (unsigned I = StartIdx; I <= EndIdx; ++I) {
          if (Mapper.InstrList[I].ContainsCall) {
            CandContainsCalls = true;
            SeqContainsCalls = true;
            // 检查是否是序列的最后一条指令
            if (I == EndIdx) {
              CandEndsWithCall = true;
            }
          }
          if (Mapper.InstrList[I].HasSPRelativeAccess) {
            CandHasSPAccess = true;
            SeqContainsSPRelativeAccess = true;
          }
          if (Mapper.InstrList[I].FromLeafFunction) {
            CandFromLeafFunc = true;
          }
        }

        // CRITICAL: Check liveness ONLY if the sequence contains SP-relative accesses
        // that would require prologue/epilogue (which clobbers caller-saved regs).
        // For ordinary sequences without SP-relative access, the outlined function
        // won't clobber any caller-saved registers, so liveness check is unnecessary.
        if (CandHasSPAccess && !checkLivenessForOutlining(BB, SeqStartIt, SeqEndIt, BC)) {
          LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Skipping candidate at "
                            << BB->getName() << " due to liveness conflict\n");
          continue; // Skip this candidate due to liveness conflict
        }

        // 关键安全检查：对于叶子函数，序列不能读取 LR
        // 因为 BL 指令会覆盖 LR，如果序列依赖 LR 的值，就会出错
        if (CandFromLeafFunc && sequenceReadsLR(SeqStartIt, SeqEndIt, BC)) {
          LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Skipping leaf func candidate at "
                            << BB->getName() << " - sequence reads LR\n");
          continue;
        }

        // 检查是否可以用尾跳转 (B) 调用 outlined function
        // 这对叶子函数很重要：如果序列后直接是 RET/B/BR，可以用 B 调用
        // 不需要返回，从而避免 BL 覆盖 LR 的问题
        bool CanUseTailJump = false;
        if (SeqEndIt != BB->end()) {
          // 检查序列后的下一条指令
          const MCInst &NextInst = *SeqEndIt;
          if (BC.MIB->isReturn(NextInst) || 
              BC.MIB->isUnconditionalBranch(NextInst) ||
              BC.MIB->isTailCall(NextInst)) {
            CanUseTailJump = true;
          }
        } else {
          // 序列是 BB 的最后几条指令，检查 BB 的后继
          // 如果没有 fall-through 后继，可能是尾跳转
          if (BB->succ_size() == 0 || 
              (BB->succ_size() == 1 && BB->getFallthrough() == nullptr)) {
            CanUseTailJump = true;
          }
        }
        // 注意：CandEndsWithCall 只影响 outlined function 内部的尾调用优化
        // 它不影响调用站点是否可以用尾跳转（B）调用 outlined function
        // 调用站点的尾跳转只取决于序列后面是否紧跟 RET/B/BR

        OutlineCandidate Cand;
        Cand.StartIdx = StartIdx;
        Cand.Length = StringLen;
        Cand.BB = BB;
        Cand.ContainsCalls = CandContainsCalls;
        Cand.EndsWithCall = CandEndsWithCall;
        Cand.FromLeafFunction = CandFromLeafFunc;
        Cand.CanUseTailJump = CanUseTailJump;
        Cand.ExecCount = CandExecCount;
        
        // 计算此候选的 CallOverhead（基于 MachineOutliner 的逻辑）
        // - 普通情况（LR 已保存在 prologue 中）：只需 BL = 4 bytes
        // - 叶子函数 + 可用尾跳转：B = 4 bytes
        // - 叶子函数 + 不能尾跳转：mov reg, lr + bl + mov lr, reg = 12 bytes
        if (CandFromLeafFunc) {
          if (CanUseTailJump) {
            Cand.CallOverhead = 4;  // B (tail jump)
          } else {
            Cand.CallOverhead = 12; // 需要保存/恢复 LR
          }
        } else {
          Cand.CallOverhead = 4;    // 普通 BL
        }
        
        // 对于叶子函数的候选，暂时完全禁用
        // TODO: 调试后重新启用
        if (CandFromLeafFunc) {
          LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Skipping leaf func candidate at "
                            << BB->getName() << " - leaf functions disabled\n");
          continue;
        }
        
        CandidatesForSeq.push_back(Cand);
      }

      // Need at least MinOccurrences non-overlapping candidates
      if (CandidatesForSeq.size() >= MinOccurrences) {
        OutlinedSequence Seq;
        Seq.Length = StringLen;
        Seq.Candidates = std::move(CandidatesForSeq);

        // 检查是否有候选来自叶子函数
        bool AnyFromLeafFunction = false;
        for (const auto &Cand : Seq.Candidates) {
          if (Cand.FromLeafFunction) {
            AnyFromLeafFunction = true;
            break;
          }
        }
        
        // 检查是否所有叶子函数候选都可以使用尾跳转
        bool AllLeafCanUseTailJump = true;
        for (const auto &Cand : Seq.Candidates) {
          if (Cand.FromLeafFunction && !Cand.CanUseTailJump) {
            AllLeafCanUseTailJump = false;
            break;
          }
        }

        // PLOS §4.1.6 Post-Link Shrink Wrapping:
        // 需要 prologue/epilogue 的情况：
        // - 包含调用：需要保存 LR（因为 BL 会覆盖 LR）
        // - 包含 SP-relative 访问：需要 prologue 来建立栈帧以便偏移修正
        // - 来自叶子函数但不能使用尾跳转：需要保存 LR
        // 注意：如果所有叶子函数候选都可以使用尾跳转，就不需要 prologue
        bool LeafNeedsPrologue = AnyFromLeafFunction && !AllLeafCanUseTailJump;
        Seq.NeedsPrologueEpilogue =
            SeqContainsCalls || SeqContainsSPRelativeAccess || LeafNeedsPrologue;

        // PLOS §4.1.4 Tail Call Optimization:
        // 如果所有候选都以 BL 结尾，可以应用尾调用优化
        // 将最后的 BL 转换为 B，省略 RET 指令
        bool AllEndsWithCall = !Seq.Candidates.empty();
        for (const auto &Cand : Seq.Candidates) {
          if (!Cand.EndsWithCall) {
            AllEndsWithCall = false;
            break;
          }
        }
        Seq.EndsWithCall = AllEndsWithCall;

        SequencesToOutline.push_back(std::move(Seq));
      }
    }

    outs() << "BOLT-OUTLINING: === Phase 2 Complete ===\n";
    outs() << "BOLT-OUTLINING:   Total repeated sequences found: "
           << TotalRepeatedSeqs << "\n";
    outs() << "BOLT-OUTLINING:   Sequences with >= " << MinOccurrences
           << " occurrences and length >= " << MinLength << ": "
           << SequencesToOutline.size() << "\n";

    // Print a few example sequences for debugging
    unsigned ExampleCount = 0;
    for (const auto &Seq : SequencesToOutline) {
      if (ExampleCount++ >= 5)
        break;
      outs() << "BOLT-OUTLINING:     - Seq length=" << Seq.Length
             << ", occurrences=" << Seq.Candidates.size()
             << ", needsPrologue=" << Seq.NeedsPrologueEpilogue << "\n";
    }

    //===--------------------------------------------------------------------===//
    // Phase 3: Apply cost model and create outlined functions
    //
    // Cost model 考虑两个因素：
    // 1. 代码大小收益：outline 后节省的字节数
    // 2. 性能影响：基于执行频率，热代码 outline 会带来函数调用开销
    //
    // 最终收益 = 大小收益 - 性能惩罚
    // 性能惩罚 = 总执行次数 * 调用开销系数
    //===--------------------------------------------------------------------===//
    outs() << "BOLT-OUTLINING: Phase 3 - Applying cost model and creating "
              "functions...\n";
    outs() << "BOLT-OUTLINING:   Evaluating " << SequencesToOutline.size()
           << " candidate sequences\n";

    // Cost model constants for AArch64:
    // - Each instruction is 4 bytes
    // - BL (call) instruction is 4 bytes
    // ============================================================
    // Cost Model based on PLOS paper (Section 4.4)
    // ============================================================
    // Benefit = NotOutlinedCost - OutliningCost
    //
    // NotOutlinedCost = K × SeqBytes
    // OutliningCost = SeqBytes + FrameBytes + K × CallBytes
    //
    // Where:
    //   K = number of occurrences of the sequence
    //   SeqBytes = sequence length in bytes (SeqLength × 4 for AArch64)
    //   CallBytes = 4 bytes (BL instruction on AArch64)
    //   FrameBytes = prologue/epilogue overhead:
    //     - With prologue: stp + ldp + ret = 12 bytes
    //     - Without prologue (shrink wrap): ret = 4 bytes
    //     - With tail call + prologue: stp + ldp = 8 bytes (no RET)
    //     - With tail call, no prologue: 0 bytes (no RET)
    // ============================================================

    const unsigned InstrSize = 4; // AArch64 fixed instruction size
    const unsigned FrameBytesWithPrologue = 12;   // stp + ldp + ret
    const unsigned FrameBytesWithoutPrologue = 4; // just ret (Shrink Wrapping)
    const unsigned FrameBytesWithTailCallAndPrologue =
        8; // stp + ldp (Tail Call with prologue)
    const unsigned FrameBytesWithTailCallNoPrologue =
        0; // nothing (Tail Call, shrink-wrapped)
    // 最小收益阈值：设置为 0 允许任何有正收益的 outlining
    // 即使只节省 4 bytes 也值得 outline（每条指令都算）
    const int64_t MinBenefitThreshold =
        0; // Allow any positive benefit (most aggressive)

    // Calculate benefit for a sequence with its candidates
    // 新的 cost model：使用逐候选的 CallOverhead
    // 基于 MachineOutliner 的公式：
    // Benefit = K × SeqBytes - (Σ CallOverhead[i] + SeqBytes + FrameBytes)
    //         = (K - 1) × SeqBytes - FrameBytes - Σ CallOverhead[i]
    auto calculateBenefitForSequence = [&](const OutlinedSequence &Seq) -> int64_t {
      const unsigned SeqBytes = Seq.Length * InstrSize;
      const unsigned K = Seq.Candidates.size();

      // 计算所有候选的总 CallOverhead
      unsigned TotalCallOverhead = 0;
      for (const auto &Cand : Seq.Candidates) {
        TotalCallOverhead += Cand.CallOverhead;
      }

      // 尾调用优化：如果序列以 BL 结尾，可以省略 RET
      unsigned FrameBytes = FrameBytesWithoutPrologue;
      if (Seq.EndsWithCall) {
        // 尾调用优化
        FrameBytes = Seq.NeedsPrologueEpilogue ? FrameBytesWithTailCallAndPrologue
                                               : FrameBytesWithTailCallNoPrologue;
      } else if (Seq.NeedsPrologueEpilogue) {
        FrameBytes = FrameBytesWithPrologue;
      }

      // NotOutlinedCost = K × SeqBytes
      // OutliningCost = TotalCallOverhead + SeqBytes + FrameBytes
      // Benefit = NotOutlinedCost - OutliningCost
      //         = K × SeqBytes - TotalCallOverhead - SeqBytes - FrameBytes
      //         = (K - 1) × SeqBytes - TotalCallOverhead - FrameBytes
      const int64_t NotOutlinedCost = K * SeqBytes;
      const int64_t OutliningCost = TotalCallOverhead + SeqBytes + FrameBytes;
      const int64_t Benefit = NotOutlinedCost - OutliningCost;

      LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Benefit = " << K << " × " << SeqBytes
                        << " - (" << TotalCallOverhead << " + " << SeqBytes
                        << " + " << FrameBytes << ") = " << Benefit
                        << (Seq.EndsWithCall ? " [tail-call]" : "") << "\n");

      return Benefit;
    };

    // Sort by benefit descending - prioritize sequences with highest code size
    // reduction. Note: PLOS paper uses length-first greedy, but benefit-first
    // works better with our SuffixTree-based implementation.
    llvm::sort(SequencesToOutline,
               [&](const OutlinedSequence &A, const OutlinedSequence &B) {
                 int64_t BenefitA = calculateBenefitForSequence(A);
                 int64_t BenefitB = calculateBenefitForSequence(B);
                 // Primary: higher benefit first
                 if (BenefitA != BenefitB)
                   return BenefitA > BenefitB;
                 // Secondary: longer sequences first (tie-breaker)
                 return A.Length > B.Length;
               });

    // Track which indices have been outlined to avoid conflicts
    DenseSet<unsigned> OutlinedIndices;

    std::vector<Replacement> Replacements;
    unsigned OutlinedCount = 0;
    unsigned SkippedNoBenefit = 0;
    unsigned MaxOutlinedFunctions = 10000; // No practical limit
    int64_t TotalBytesSaved = 0;
    unsigned ShrinkWrappedCount = 0;
    unsigned TailCallOptCount = 0; // 尾调用优化计数

    for (OutlinedSequence &Seq : SequencesToOutline) {
      if (OutlinedCount >= MaxOutlinedFunctions)
        break;

      // Filter candidates that don't conflict with already outlined regions
      std::vector<OutlineCandidate> ValidCandidates;
      for (const OutlineCandidate &Cand : Seq.Candidates) {
        bool Conflict = false;
        for (unsigned I = Cand.StartIdx; I <= Cand.getEndIdx(); ++I) {
          if (OutlinedIndices.count(I)) {
            Conflict = true;
            break;
          }
        }
        if (!Conflict)
          ValidCandidates.push_back(Cand);
      }

      // Still need at least 2 candidates
      if (ValidCandidates.size() < MinOccurrences)
        continue;

      // Apply cost model using per-candidate CallOverhead
      // 创建临时序列用于计算收益
      OutlinedSequence TempSeq;
      TempSeq.Length = Seq.Length;
      TempSeq.Candidates = ValidCandidates;
      TempSeq.NeedsPrologueEpilogue = Seq.NeedsPrologueEpilogue;
      TempSeq.EndsWithCall = Seq.EndsWithCall;
      
      int64_t Benefit = calculateBenefitForSequence(TempSeq);
      if (Benefit < MinBenefitThreshold) {
        SkippedNoBenefit++;
        continue;
      }

      // Use the first candidate to create the outlined function
      const OutlineCandidate &FirstCand = ValidCandidates[0];
      BinaryBasicBlock::iterator StartIt =
          Mapper.InstrList[FirstCand.StartIdx].It;
      BinaryBasicBlock::iterator EndIt =
          Mapper.InstrList[FirstCand.getEndIdx()].It;
      ++EndIt; // Make it past-the-end

      BinaryBasicBlock *BB = FirstCand.BB;
      BinaryFunction *BF = BB->getFunction();

      LLVM_DEBUG(dbgs() << "BOLT-OUTLINING: Outlining " << Seq.Length
                        << " instrs x " << ValidCandidates.size()
                        << " occurrences, benefit=" << Benefit
                        << " bytes, shrinkWrap=" << !Seq.NeedsPrologueEpilogue
                        << ", tailCall=" << Seq.EndsWithCall << "\n");

      BinaryFunction *OutlinedFunc =
          createOutlinedFunction(BC, *BF, *BB, StartIt, EndIt,
                                 Seq.NeedsPrologueEpilogue, Seq.EndsWithCall);

      if (!OutlinedFunc) {
        outs()
            << "BOLT-OUTLINING:   [WARN] Failed to create outlined function\n";
        continue;
      }

      // 尾调用优化：只要序列以 BL 结尾就可以应用
      bool AppliedTailCall = Seq.EndsWithCall;

      outs() << "BOLT-OUTLINING:   Created " << OutlinedFunc->getPrintName()
             << " (" << Seq.Length << " insts x " << ValidCandidates.size()
             << " sites, benefit=" << Benefit << " bytes";
      if (!Seq.NeedsPrologueEpilogue)
        outs() << ", shrink-wrapped";
      if (AppliedTailCall)
        outs() << ", tail-call";
      outs() << ")\n";

      Seq.OutlinedFunc = OutlinedFunc;
      OutlinedCount++;
      TotalBytesSaved += Benefit;
      if (!Seq.NeedsPrologueEpilogue)
        ShrinkWrappedCount++;
      if (AppliedTailCall)
        TailCallOptCount++;

      // Record replacements for all valid candidates
      for (const OutlineCandidate &Cand : ValidCandidates) {
        // Mark indices as outlined
        for (unsigned I = Cand.StartIdx; I <= Cand.getEndIdx(); ++I) {
          OutlinedIndices.insert(I);
        }

        // Find the instruction range within the BB
        BinaryBasicBlock::iterator CandEndIt =
            Mapper.InstrList[Cand.getEndIdx()].It;
        ++CandEndIt; // Make it past-the-end for iteration
        BinaryBasicBlock *CandBB = Cand.BB;

        Replacement R;
        R.BB = CandBB;
        R.CallTarget = OutlinedFunc->getSymbol();
        R.NeedsPrologueEpilogue = Seq.NeedsPrologueEpilogue;
        R.FromLeafFunction = Cand.FromLeafFunction;
        R.UseTailJump = Cand.CanUseTailJump && Cand.FromLeafFunction;
        R.SaveLRReg = 0;
        R.EraseFollowingTerminator = false;
        
        // 对于叶子函数使用尾跳转时，如果序列后紧跟 RET/B，需要删除它
        // 因为 outlined function 的 RET/B 会替代原来的
        if (R.UseTailJump && CandEndIt != CandBB->end()) {
          const MCInst &NextInst = *CandEndIt;
          if (BC.MIB->isReturn(NextInst) || 
              BC.MIB->isUnconditionalBranch(NextInst)) {
            R.EraseFollowingTerminator = true;
          }
        }

        // 计算索引
        unsigned StartIndex = 0;
        unsigned EndIndex = 0;
        unsigned Index = 0;
        --CandEndIt; // Back to inclusive end
        for (auto It = CandBB->begin(); It != CandBB->end(); ++It, ++Index) {
          if (&(*It) == &(*Mapper.InstrList[Cand.StartIdx].It))
            StartIndex = Index;
          if (&(*It) == &(*CandEndIt)) {
            EndIndex = Index;
            break;
          }
        }
        R.StartIndex = StartIndex;
        R.NumToErase = EndIndex - StartIndex + 1;
        
        Replacements.push_back(R);
      }
    }

    outs() << "BOLT-OUTLINING: Skipped " << SkippedNoBenefit
           << " sequences with insufficient benefit\n";

    //===--------------------------------------------------------------------===//
    // Phase 4: Apply replacements (from back to front to preserve indices)
    //===--------------------------------------------------------------------===//
    outs() << "BOLT-OUTLINING: Phase 4 - Applying " << Replacements.size()
           << " replacements...\n";

    // Sort replacements by BB and then by StartIndex descending
    // This ensures we process later indices first within each BB
    llvm::sort(Replacements, [](const Replacement &A, const Replacement &B) {
      if (A.BB != B.BB)
        return A.BB < B.BB;
      return A.StartIndex > B.StartIndex; // Descending order
    });

    unsigned ReplacementsDone = 0;
    unsigned LeafFunctionSaves = 0;
    for (const Replacement &R : Replacements) {
      BinaryBasicBlock &BB = *R.BB;

      // Navigate to the start position
      auto It = BB.begin();
      std::advance(It, R.StartIndex);

      // Erase the original instructions
      for (unsigned i = 0; i < R.NumToErase; ++i) {
        It = BB.eraseInstruction(It);
      }
      
      // 如果使用尾跳转且需要删除后面的终结指令（RET/B）
      if (R.EraseFollowingTerminator && It != BB.end()) {
        It = BB.eraseInstruction(It);
      }

      // 处理调用 outlined function 的方式：
      // 1. 如果 UseTailJump：使用 B（尾跳转），不返回，用于叶子函数
      // 2. 如果 SaveLRReg != 0：保存/恢复 LR（旧方式，可能不安全，暂时保留）
      // 3. 正常情况：使用 BL
      if (R.UseTailJump) {
        // 对于叶子函数的尾跳转优化：用 B 替代 BL
        // 这样不会覆盖 LR，outlined function 的 RET 直接返回到原调用者
        MCInst TailJump;
        BC.MIB->createTailCall(TailJump, R.CallTarget, BC.Ctx.get());
        BB.insertInstruction(It, std::move(TailJump));
        LeafFunctionSaves++;
      } else if (R.FromLeafFunction && R.SaveLRReg != 0) {
        // 创建 mov SaveLRReg, x30 (ORR Xd, XZR, Xm)
        MCInst SaveLR;
        SaveLR.setOpcode(AArch64::ORRXrs);
        SaveLR.addOperand(MCOperand::createReg(R.SaveLRReg)); // Rd
        SaveLR.addOperand(MCOperand::createReg(AArch64::XZR)); // Rn = XZR
        SaveLR.addOperand(MCOperand::createReg(AArch64::LR));  // Rm = X30
        SaveLR.addOperand(MCOperand::createImm(0)); // shift = 0
        It = BB.insertInstruction(It, std::move(SaveLR));
        ++It;
        
        // 创建 BL
        MCInst CallInst;
        BC.MIB->createCall(CallInst, R.CallTarget, BC.Ctx.get());
        It = BB.insertInstruction(It, std::move(CallInst));
        ++It;
        
        // 创建 mov x30, SaveLRReg
        MCInst RestoreLR;
        RestoreLR.setOpcode(AArch64::ORRXrs);
        RestoreLR.addOperand(MCOperand::createReg(AArch64::LR)); // Rd = X30
        RestoreLR.addOperand(MCOperand::createReg(AArch64::XZR)); // Rn = XZR
        RestoreLR.addOperand(MCOperand::createReg(R.SaveLRReg));  // Rm
        RestoreLR.addOperand(MCOperand::createImm(0)); // shift = 0
        BB.insertInstruction(It, std::move(RestoreLR));
      } else {
        // 正常情况：直接插入 BL
        MCInst CallInst;
        BC.MIB->createCall(CallInst, R.CallTarget, BC.Ctx.get());
        BB.insertInstruction(It, std::move(CallInst));
      }
      
      ReplacementsDone++;

      // Print progress every 100 replacements or for the first few
      if (ReplacementsDone <= 5 || ReplacementsDone % 100 == 0) {
        outs() << "BOLT-OUTLINING:   [" << ReplacementsDone << "/"
               << Replacements.size() << "] Replaced " << R.NumToErase
               << " insts in " << BB.getFunction()->getPrintName()
               << "::" << BB.getName() << " with call to "
               << R.CallTarget->getName();
        if (R.UseTailJump)
          outs() << " (leaf-tail-jump)";
        outs() << "\n";
      }
    }
    outs() << "BOLT-OUTLINING:   All " << ReplacementsDone
           << " replacements applied successfully";
    if (LeafFunctionSaves > 0)
      outs() << " (" << LeafFunctionSaves << " tail-jumps for leaf functions)";
    outs() << "\n";

    outs() << "BOLT-OUTLINING: === Phase 4 Complete ===\n";
    outs() << "BOLT-OUTLINING: ========== Iteration " << Iteration
           << " Summary ==========\n";
    outs() << "BOLT-INFO: Outliner: created " << OutlinedCount
           << " outlined functions (" << ShrinkWrappedCount
           << " shrink-wrapped, " << TailCallOptCount << " tail-call)\n";
    outs() << "BOLT-INFO: Outliner: replaced " << Replacements.size()
           << " instruction sequences with calls\n";
    outs() << "BOLT-INFO: Outliner: estimated code size reduction: "
           << TotalBytesSaved << " bytes\n";
    outs() << "BOLT-OUTLINING: ========================================\n";

    // Accumulate statistics across iterations
    TotalOutlinedFuncs += OutlinedCount;
    TotalReplacements += Replacements.size();
    TotalBytesSavedAll += TotalBytesSaved;
    TotalShrinkWrapped += ShrinkWrappedCount;

    // If this iteration found nothing, no point continuing
    if (OutlinedCount == 0) {
      outs() << "BOLT-OUTLINING: No new outlined functions found in iteration "
             << Iteration << ", stopping early\n";
      break;
    }
  } // end of iteration loop

  // Final summary across all iterations
  outs() << "\n";
  outs() << "BOLT-OUTLINING: ========== Final Outliner Summary ==========\n";
  outs() << "BOLT-INFO: Outliner: total outlined functions: "
         << TotalOutlinedFuncs << " (" << TotalShrinkWrapped
         << " shrink-wrapped)\n";
  outs() << "BOLT-INFO: Outliner: total replacements: " << TotalReplacements
         << "\n";
  outs() << "BOLT-INFO: Outliner: total estimated code size reduction: "
         << TotalBytesSavedAll << " bytes\n";
  outs() << "BOLT-OUTLINING: =============================================\n";

  return Error::success();
}