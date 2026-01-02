# Post-Link Outlining Pass for LLVM BOLT
基于 LLVM BOLT 构建的 Post Link Outlining Pass，通过将重复的指令序列提取为函数来减小二进制代码体积。
具体效果请参考[README.pdf](https://github.com/ddsfda99/llvm-project/blob/main/README.pdf)。
编译好的llvm-bolt在本项目的build_bolt/bin/llvm-bolt。
## 概述
Outliner Pass 识别出重复出现的指令序列，提取为独立的 outlined functions，原位置的代码则被替换为函数调用或无条件跳转。
## 优化技术
1. 基于后缀树的重复序列检测，O(n) 时间复杂度。
2. 有 Profile 时只对冷代码提取，保护热路径性能。
3. 不含函数调用的序列省略 prologue/epilogue。
4. 尾调用优化，BL 结尾的序列转换为 B。
5. SP-relative 偏移修正，自动处理栈访问偏移 +16。
6. 支持多轮迭代发现嵌套重复模式。
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
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON

# 编译
ninja bolt
```
## Post Link Outlining Pass 相关文件
| 路径 | 说明 |
|------|------|
| `bolt/lib/Passes/Outliner.cpp` | Outliner Pass 实现 |
| `bolt/include/bolt/Passes/Outliner.h` | Outliner Pass 接口 |

## 参考资料
1. [PLOS: Post-Link Outlining for Code Size Reduction](https://dl.acm.org/doi/abs/10.1145/3708493.3712692)
2. [LLVM MachineOutliner](https://llvm.org/doxygen/MachineOutliner_8cpp_source.html)



