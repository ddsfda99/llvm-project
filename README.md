# Post-Link Outlining Pass for LLVM BOLT

基于 LLVM BOLT 构建的后链接代码提取（Post-Link Outlining）优化 Pass，通过将重复的指令序列抽取为共享函数来显著减小二进制代码体积。

## 概述

Outliner Pass 工作在后链接（post-link）阶段，对最终链接好的二进制文件进行分析，识别出重复出现的指令序列。这些重复序列会被抽取成独立的"提取函数"（outlined functions），原位置的代码则被替换为函数调用或直接跳转。

### 核心特性

- **基于后缀树的重复序列检测**：使用 LLVM SuffixTree，O(n) 时间复杂度
- **Profile 引导的冷代码提取**：有 Profile 时只对冷代码提取，保护热路径性能
- **Shrink Wrapping 优化**：不含函数调用的序列省略 prologue/epilogue
- **尾调用优化**：BL 结尾的序列转换为 B（尾跳转）
- **SP-relative 偏移修正**：自动处理栈访问偏移 +16
- **迭代式嵌套提取**：支持多轮迭代发现嵌套重复模式

## 测试结果

4 个测试案例均通过正确性验证：

| 测试用例 | Outlined 函数数 | 替换次数 | 代码体积节省 | 验证 |
|---------|----------------|---------|-------------|------|
| blowfish | 4 (2 shrink-wrapped) | 10 | 24 bytes | ✅ MD5 一致 |
| basicmath | 10 (5 shrink-wrapped) | 43 | 296 bytes | ✅ 输出一致 |
| lame | 166 (94 shrink-wrapped) | 765 | 2,960 bytes | ✅ MD5 一致 |
| typeset | 1,129 (717 shrink-wrapped) | 8,014 | 31,264 bytes | ✅ MD5 一致 |

**总计节省：约 34.5 KB 代码空间**

## 使用方法

### 基本用法

```bash
llvm-bolt <输入二进制> -o <输出二进制> --enable-outliner
```

### 完整优化命令

```bash
llvm-bolt binary -o binary.plos \
  --enable-outliner \
  --eliminate-unreachable \
  --simplify-conditional-tail-calls \
  --peepholes=all \
  --icf=all \
  --use-old-text \
  --align-text=4 \
  --align-functions=4
```

### Profile 引导用法

```bash
# 1. 生成插桩二进制
llvm-bolt binary -o binary.instrumented --instrument

# 2. 运行收集数据
./binary.instrumented <负载>

# 3. 使用 Profile 优化
llvm-bolt binary -o binary.outlined \
  --data=/tmp/prof.fdata \
  --enable-outliner
```

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--enable-outliner` | 开启 Outliner Pass | false |
| `--outliner-cold-only` | 只提取冷代码 | true |
| `--outliner-cold-threshold=N` | 冷代码阈值 | 0 |
| `--outliner-max-iterations=N` | 最大迭代次数 | 8 |

## 构建方法

```bash
# 克隆仓库
git clone https://github.com/ddsfda99/llvm-project.git
cd llvm-project

# 配置构建
mkdir build_bolt && cd build_bolt
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="bolt;clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="AArch64;X86"

# 编译
ninja bolt
```

## 相关文件

| 路径 | 说明 |
|------|------|
| `bolt/lib/Passes/Outliner.cpp` | Outliner Pass 实现（约 2570 行） |
| `bolt/include/bolt/Passes/Outliner.h` | Outliner Pass 接口 |
| `bolt/docs/Outliner-README-zh.md` | 详细中文文档 |
| `bolt/docs/BOLT_Outliner_Technical_Document.md` | 技术文档 |

## 当前限制

- 仅支持 **AArch64** 架构

## 参考文献

1. [PLOS: Post-Link Outlining for Code Size Reduction](https://dl.acm.org/doi/abs/10.1145/3708493.3712692)
2. [LLVM MachineOutliner](https://llvm.org/doxygen/MachineOutliner_8cpp_source.html)
3. [LLVM BOLT](https://github.com/llvm/llvm-project/tree/main/bolt)

## License

Apache 2.0 License with LLVM Exceptions
