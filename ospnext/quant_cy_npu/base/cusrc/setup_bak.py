"""
setup_bak.py — 自动探测 Ascend 库路径的健壮版本

与原 setup.py 的主要区别：
  1. 通过环境变量 + 多候选目录自动定位 CANN toolkit 根路径
  2. 分别为 libruntime.so / libascendcl.so 搜索多个候选目录，找到即停止
  3. ccec 阶段：设备侧内核 .so 默认不链接 host-side runtime/acl 库（最干净方案），
     若用户设置 CCEC_LINK_RUNTIME=1 才启用链接（兼容旧行为）
  4. 检查 ccec 返回码，失败立即中止并给出诊断信息，不再让错误被后续 setup() 冲淡
  5. 打印完整的路径探测过程，便于在服务器上直接定位问题

用法：
  python setup_bak.py build_ext --inplace

诊断模式（只探测路径，不编译）：
  python setup_bak.py --probe-only
"""

import os
import sys
import glob

# ─────────────────────────────────────────────────────────────────────────────
# 0. 诊断模式
# ─────────────────────────────────────────────────────────────────────────────
PROBE_ONLY = '--probe-only' in sys.argv
if PROBE_ONLY:
    sys.argv.remove('--probe-only')


def info(msg):
    print(f'[setup] {msg}')

def warn(msg):
    print(f'[setup][WARN] {msg}')

def die(msg):
    print(f'[setup][ERROR] {msg}', file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 确定 Ascend toolkit 根路径
#    优先级：ASCEND_TOOLKIT_HOME > DDK_PATH > /usr/local/Ascend/.../latest
# ─────────────────────────────────────────────────────────────────────────────

def resolve_toolkit_root():
    candidates = []

    # 1a. 标准环境变量（source set_env.sh 后会设置）
    for var in ('ASCEND_TOOLKIT_HOME', 'DDK_PATH', 'ASCEND_HOME_PATH'):
        v = os.environ.get(var)
        if v:
            candidates.append((var, os.path.realpath(v)))

    # 1b. 常见硬编码路径（覆盖不同 CANN 安装形态）
    fallbacks = [
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/latest',
        '/usr/local/Ascend/ascend-toolkit',
    ]
    for p in fallbacks:
        rp = os.path.realpath(p)
        if os.path.isdir(rp):
            candidates.append(('hardcoded', rp))

    info('=== Ascend toolkit root 候选路径 ===')
    for src, p in candidates:
        info(f'  [{src}] {p}')

    if not candidates:
        die('找不到 Ascend toolkit 根目录。\n'
            '请先执行：source /usr/local/Ascend/ascend-toolkit/set_env.sh\n'
            '或手动设置环境变量 ASCEND_TOOLKIT_HOME=<toolkit路径>')

    root = candidates[0][1]
    info(f'使用 toolkit root: {root}')
    return root


TOOLKIT_ROOT = resolve_toolkit_root()

# 实际指向的真实路径（展开软链）
info(f'  realpath → {os.path.realpath(TOOLKIT_ROOT)}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. 探测 arch 前缀（aarch64-linux / x86_64-linux）
# ─────────────────────────────────────────────────────────────────────────────

import platform
_machine = platform.machine()
if _machine == 'aarch64':
    ARCH_PREFIX = 'aarch64-linux'
elif _machine == 'x86_64':
    ARCH_PREFIX = 'x86_64-linux'
else:
    ARCH_PREFIX = 'aarch64-linux'
    warn(f'未知架构 {_machine}，默认使用 aarch64-linux')

info(f'架构前缀: {ARCH_PREFIX}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. 自动搜索 so 文件
#    为每个库名单独搜索，找到即停止，并输出完整搜索日志
# ─────────────────────────────────────────────────────────────────────────────

# 候选库目录（按优先级排列）
LIB_SEARCH_DIRS = [
    # 来自环境变量 NPU_HOST_LIB（source set_env.sh 后设置）
    os.environ.get('NPU_HOST_LIB', ''),

    # stub 目录（编译期链接用，不含实际实现，最推荐用于编译期 -L）
    f'{TOOLKIT_ROOT}/runtime/lib64/stub',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/lib64/stub',

    # 正常 lib64 目录
    f'{TOOLKIT_ROOT}/runtime/lib64',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/lib64',
    f'{TOOLKIT_ROOT}/lib64',

    # acllib（部分 CANN 版本把 ascendcl 单独放这里）
    f'{TOOLKIT_ROOT}/acllib/lib64',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/acllib/lib64',

    # 更老的安装形态
    '/usr/local/Ascend/runtime/lib64',
    '/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64',
]
LIB_SEARCH_DIRS = [d for d in LIB_SEARCH_DIRS if d]  # 去掉空字符串


def find_lib(libname):
    """在候选目录列表中搜索 lib<libname>.so，返回找到的第一个目录，未找到返回 None。"""
    filename = f'lib{libname}.so'
    info(f'--- 搜索 {filename} ---')
    for d in LIB_SEARCH_DIRS:
        full = os.path.join(d, filename)
        # 也匹配带版本号的 so（libfoo.so.x）
        matches = glob.glob(full) + glob.glob(full + '.*')
        exists = os.path.exists(full) or bool(matches)
        status = 'FOUND' if exists else '-----'
        info(f'  [{status}] {d}')
        if exists:
            return d
    return None


info('\n=== 搜索 Ascend 运行时库 ===')
runtime_lib_dir  = find_lib('runtime')
ascendcl_lib_dir = find_lib('ascendcl')

info('\n=== 搜索结果 ===')
info(f'  libruntime.so  → {runtime_lib_dir  or "NOT FOUND"}')
info(f'  libascendcl.so → {ascendcl_lib_dir or "NOT FOUND"}')

if PROBE_ONLY:
    info('\n[诊断模式] 只探测路径，不执行编译。退出。')
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ccec 阶段：编译设备侧内核 .so
#
#    设计决策：
#      - 设备侧内核（AICore 代码）在运行时由 ACL/torch_npu 动态加载，
#        编译期通常不需要链接 libruntime / libascendcl（host-side 库）。
#      - 默认行为：不链接，避免"找不到库"导致的链接失败。
#      - 如果你的 ccec 版本或 toolkit 强制要求，设置环境变量：
#          export CCEC_LINK_RUNTIME=1
#        再重新编译即可恢复链接行为。
# ─────────────────────────────────────────────────────────────────────────────

CCEC_LINK_RUNTIME = os.environ.get('CCEC_LINK_RUNTIME', '0') == '1'

ascend_op_extension_name = 'npu_quant_op'
ascend_op_src_files = ['npu_quantop_base.cpp']

ascend_include_paths = [
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/tikcpp/tikcfw/',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/tikcpp/tikcfw/interface',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/tikcpp/tikcfw/impl',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/tikcpp/tikcfw/kernel_tiling',
    f'{TOOLKIT_ROOT}/{ARCH_PREFIX}/tikcpp/tikcfw/op_frame',
]

# 验证 include 路径是否存在
info('\n=== 检查 include 路径 ===')
missing_includes = []
for p in ascend_include_paths:
    exists = os.path.isdir(p)
    info(f'  [{"OK  " if exists else "MISS"}] {p}')
    if not exists:
        missing_includes.append(p)
if missing_includes:
    warn(f'{len(missing_includes)} 个 include 目录不存在，编译可能失败（见上方 MISS 条目）')

# 构建 ccec 命令
ccec_lib_dirs = []
ccec_libs = []

if CCEC_LINK_RUNTIME:
    info('\nCCEC_LINK_RUNTIME=1，将链接 runtime/ascendcl')
    for libname, found_dir in [('runtime', runtime_lib_dir), ('ascendcl', ascendcl_lib_dir)]:
        if found_dir is None:
            die(f'CCEC_LINK_RUNTIME=1 但未找到 lib{libname}.so，无法链接。\n'
                f'请先 source set_env.sh 或手动设置 NPU_HOST_LIB。')
        if found_dir not in ccec_lib_dirs:
            ccec_lib_dirs.append(found_dir)
        ccec_libs.append(libname)
else:
    info('\n[默认] ccec 阶段不链接 host-side runtime/ascendcl（推荐）')

cmd_parts = [
    'ccec -O3 -xcce',
    ' '.join(ascend_op_src_files),
    '--cce-aicore-arch=dav-c220-vec',
    ' '.join(f'-L{p}' for p in ccec_lib_dirs),
    ' '.join(f'-I{p}' for p in ascend_include_paths),
    ' '.join(f'-l{l}' for l in ccec_libs),
    f'-shared -fPIC -o lib{ascend_op_extension_name}.so -std=c++17',
]
cmd = ' '.join(p for p in cmd_parts if p.strip())

info(f'\n=== ccec 命令 ===\n{cmd}\n')

ret = os.system(cmd)
if ret != 0:
    die(
        f'ccec 编译失败，返回码 {ret}。\n'
        f'\n排查建议：\n'
        f'  1. 确认已 source set_env.sh：\n'
        f'       source /usr/local/Ascend/ascend-toolkit/set_env.sh\n'
        f'  2. 检查 ARCH_PREFIX 是否正确（当前：{ARCH_PREFIX}）\n'
        f'  3. 用诊断模式确认路径：\n'
        f'       python setup_bak.py --probe-only\n'
        f'  4. 确认 ccec 可执行：which ccec\n'
        f'  5. 如果仍然提示找不到库，尝试：\n'
        f'       export CCEC_LINK_RUNTIME=1\n'
        f'       python setup_bak.py build_ext --inplace\n'
        f'     并提供 libruntime.so / libascendcl.so 的实际路径（见上方探测结果）'
    )

info(f'ccec 成功生成 lib{ascend_op_extension_name}.so')


# ─────────────────────────────────────────────────────────────────────────────
# 5. 构建 PyTorch pybind11 扩展
# ─────────────────────────────────────────────────────────────────────────────

from setuptools import setup
from torch.utils import cpp_extension
import torch
import torch_npu

pytorch_extension_name = 'npu_quant'
pytorch_interface_cpp_files = ['npu_quant.cpp']

torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
info(f'\n=== PyTorch 扩展 ===')
info(f'TORCH_NPU path: {torch_npu_path}')
info(f'torch version:  {torch.__version__}')

# rpath：运行时在这两个目录里找 .so，不依赖 LD_LIBRARY_PATH
extra_link_args = [
    f'-Wl,-rpath={torch_npu_path}/lib,-rpath={os.getcwd()}'
]

setup(
    name=pytorch_extension_name,
    ext_modules=[
        cpp_extension.CppExtension(
            pytorch_extension_name,
            pytorch_interface_cpp_files,
            include_dirs=[
                torch_npu_path + '/include',
            ],
            library_dirs=[
                torch_npu_path + '/lib',
                os.getcwd(),            # libnpu_quant_op.so 就在当前目录
            ],
            libraries=[ascend_op_extension_name, 'torch_npu'],
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
