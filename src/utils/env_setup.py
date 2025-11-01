"""环境变量设置模块

统一管理系统环境变量，避免代码重复
"""
import os
import platform
from pathlib import Path


def setup_ghostscript_env():
    """设置Ghostscript环境变量

    根据不同操作系统设置对应的库路径：
    - macOS: 设置 DYLD_LIBRARY_PATH
    - Linux: 设置 LD_LIBRARY_PATH
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        # Homebrew安装的Ghostscript路径
        gs_paths = [
            '/opt/homebrew/lib',  # Apple Silicon
            '/usr/local/lib',      # Intel Mac
        ]

        # 找到存在的路径
        existing_path = None
        for path in gs_paths:
            if Path(path).exists():
                existing_path = path
                break

        if existing_path:
            env_var = 'DYLD_LIBRARY_PATH'
            current_path = os.environ.get(env_var, '')

            # 避免重复添加
            if existing_path not in current_path:
                new_path = f"{existing_path}:{current_path}" if current_path else existing_path
                os.environ[env_var] = new_path

    elif system == "Linux":
        # Linux系统设置
        gs_path = '/usr/lib'
        if Path(gs_path).exists():
            env_var = 'LD_LIBRARY_PATH'
            current_path = os.environ.get(env_var, '')

            if gs_path not in current_path:
                new_path = f"{gs_path}:{current_path}" if current_path else gs_path
                os.environ[env_var] = new_path


def ensure_env_in_subprocess():
    """确保子进程中环境变量被正确设置

    在使用multiprocessing时，需要在每个子进程中调用此函数
    """
    setup_ghostscript_env()
