import subprocess
import sys
import os
import argparse
import shutil


def pack_script(
    script_path,
    exe_name=None,
    one_file=True,
    no_console=False,
    icon_path=None,
    clean=True,
):
    """
    使用 PyInstaller 将指定的 Python 脚本打包成 EXE。

    :param script_path: 目标 Python 脚本的路径。
    :param exe_name: 生成的 EXE 文件名 (不含扩展名)。如果为 None, 则使用脚本名。
    :param one_file: 是否打包成单个 EXE 文件。
    :param no_console: 是否隐藏控制台窗口 (适用于 GUI 应用)。
    :param icon_path: 要使用的图标文件 (.ico) 的路径。
    :param clean: 是否在打包后清理临时文件 (build 目录和 .spec 文件)。
    """
    # 0. 检查这是否在 Windows 上运行
    if sys.platform != "win32":
        print("错误：此脚本只能在 Windows 操作系统上创建 .exe 文件。")
        sys.exit(1)

    # 1. 验证输入路径是否存在
    if not os.path.exists(script_path):
        print(f"错误：找不到指定的脚本文件 '{script_path}'")
        sys.exit(1)

    if icon_path and not os.path.exists(icon_path):
        print(f"错误：找不到指定的图标文件 '{icon_path}'")
        sys.exit(1)

    # 2. 构建 PyInstaller 命令
    command = ["pyinstaller", "--noconfirm", script_path]

    if one_file:
        command.append("--onefile")
    if no_console:
        command.append("--noconsole")
    if exe_name:
        command.extend(["--name", exe_name])
    if icon_path:
        command.extend(["--icon", icon_path])

    print("-" * 50)
    print(f"准备打包脚本: {os.path.basename(script_path)}")
    print(f"即将执行命令: \n{' '.join(command)}")
    print("-" * 50)

    # 3. 执行打包命令并实时显示输出
    try:
        subprocess.run(
            command,
            check=True,  # 如果命令返回非零退出码，则引发 CalledProcessError
            text=True,  # 将 stdout/stderr 解码为文本
            encoding="utf-8",  # 明确指定编码以避免乱码
            stdout=sys.stdout,  # 将子进程的标准输出重定向到当前脚本的标准输出
            stderr=sys.stderr,  # 将子进程的标准错误重定向到当前脚本的标准错误
        )
        print("\n[SUCCESS] PyInstaller 打包成功！")
    except FileNotFoundError:
        print("\n[ERROR] 找不到 'pyinstaller' 命令。")
        print(
            "请确保您已经通过 'pip install pyinstaller' 安装了它，并且 Python 的 Scripts 目录已在系统的 PATH 环境变量中。"
        )
        sys.exit(1)
    except subprocess.CalledProcessError:
        # 当 check=True 且进程返回错误码时，会抛出此异常
        print("\n[ERROR] PyInstaller 在打包过程中遇到错误。")
        print("请检查上面显示的 PyInstaller 输出日志以了解详细信息。")
        sys.exit(1)

    # 4. 清理临时文件 (已修正 spec 文件名逻辑)
    if clean:
        print("正在清理临时文件...")
        base_name = (
            exe_name if exe_name else os.path.splitext(os.path.basename(script_path))[0]
        )
        spec_file = f"{base_name}.spec"

        try:
            if os.path.exists("build"):
                shutil.rmtree("build")
                print("- 已删除 'build' 目录")
            if os.path.exists(spec_file):
                os.remove(spec_file)
                print(f"- 已删除 '{spec_file}' 文件")
            else:
                # 添加一个检查，以防 spec 文件因某些原因找不到
                print(f"- 未找到 spec 文件 '{spec_file}'，跳过删除。")

        except OSError as e:
            print(f"清理临时文件时出错: {e}")

    # 5. 指示输出位置
    output_dir = os.path.join(os.getcwd(), "dist")
    print("\n打包完成！")
    print(f"您的 .exe 文件位于: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一个使用 PyInstaller 将 Python 脚本打包为 .exe 的工具。",
        formatter_class=argparse.RawTextHelpFormatter,  # 保持帮助文本格式
    )

    # (已改进) 将 script_path 作为位置参数
    parser.add_argument("script_path", help="要打包的 Python 脚本的路径。")
    parser.add_argument("-n", "--name", help="指定输出的 EXE 文件名 (不含 .exe)。")
    parser.add_argument("-i", "--icon", help="为 EXE 指定一个 .ico 图标文件路径。")
    parser.add_argument(
        "--one-dir",
        action="store_false",
        dest="one_file",
        help="打包成一个包含所有依赖的文件夹 (默认是单个文件)。",
    )
    parser.add_argument(
        "-w",
        "--no-console",
        action="store_true",
        help="运行时不显示命令行窗口 (适用于 GUI 程序)。",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="打包后不清理 build 目录和 .spec 文件。",
    )

    args = parser.parse_args()

    pack_script(
        script_path=args.script_path,
        exe_name=args.name,
        one_file=args.one_file,
        no_console=args.no_console,
        icon_path=args.icon,
        clean=args.clean,
    )
