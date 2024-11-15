# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import collections
import contextlib
import datetime
import signal
import subprocess
import time
import typing
from pathlib import Path

from ..logger import get_logger
from ..platform_helpers import is_linux, is_windows
from ..run import run

_log = get_logger("util.android")

SdkToolPaths = collections.namedtuple("SdkToolPaths",
                                      ["emulator", "adb", "sdkmanager", "avdmanager"])


def get_sdk_tool_paths(sdk_root: str):
    def filename(name, windows_extension):
        if is_windows():
            return f"{name}.{windows_extension}"
        else:
            return name

    sdk_root = Path(sdk_root).resolve(strict=True)

    return SdkToolPaths(
        # do not use sdk_root/tools/emulator as that is superseded by sdk_root/emulator/emulator
        emulator=str((sdk_root / "emulator" / filename("emulator", "exe")).resolve(strict=True)),
        adb=str((sdk_root / "platform-tools" / filename("adb", "exe")).resolve(strict=True)),
        sdkmanager=str(
            (sdk_root / "cmdline-tools" / "latest" / "bin" / filename("sdkmanager", "bat")).resolve(
                strict=True)
        ),
        avdmanager=str(
            (sdk_root / "cmdline-tools" / "latest" / "bin" / filename("avdmanager", "bat")).resolve(
                strict=True)
        ),
    )


def create_virtual_device(sdk_tool_paths: SdkToolPaths, system_image_package_name: str,
                          avd_name: str):
    run(sdk_tool_paths.sdkmanager, "--install", system_image_package_name, input=b"y")

    run(
        sdk_tool_paths.avdmanager,
        "create",
        "avd",
        "--name",
        avd_name,
        "--package",
        system_image_package_name,
        "--force",
        input=b"no",
    )


_process_creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0


def _start_process(*args) -> subprocess.Popen:
    _log.debug(f"Starting process - args: {[*args]}")
    return subprocess.Popen([*args], creationflags=_process_creationflags)


_stop_signal = signal.CTRL_BREAK_EVENT if is_windows() else signal.SIGTERM


def _stop_process(proc: subprocess.Popen):
    if proc.returncode is not None:
        # process has exited
        return

    _log.debug(f"Stopping process - args: {proc.args}")
    proc.send_signal(_stop_signal)

    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        _log.warning("Timeout expired, forcibly stopping process...")
        proc.kill()


def _stop_process_with_pid(pid: int):
    # minimize scope of external module usage
    import psutil

    if psutil.pid_exists(pid):
        process = psutil.Process(pid)
        _log.debug(f"Stopping process - pid={pid}")
        process.terminate()
        try:
            process.wait(60)
        except psutil.TimeoutExpired:
            print("Process did not terminate within 60 seconds. Killing.")
            process.kill()
            time.sleep(10)
            if psutil.pid_exists(pid):
                print(f"Process still exists. State:{process.status()}")
    else:
        _log.debug(f"No process exists with pid={pid}")


def start_emulator(
        sdk_tool_paths: SdkToolPaths, avd_name: str,
        extra_args: typing.Optional[typing.Sequence[str]] = None
) -> subprocess.Popen:
    def check_emulator_running() -> bool:
        """
        Check if an emulator is already running by parsing adb devices output.
        """
        try:
            output = subprocess.check_output([sdk_tool_paths.adb, "devices"], timeout=10, text=True)
            # Filter lines containing "emulator" to detect running emulators
            running_devices = [line for line in output.splitlines() if "emulator" in line]
            return len(running_devices) > 0
        except subprocess.SubprocessError as e:
            _log.error(f"Error checking running emulators: {e}")
            return False

    if check_emulator_running():
        raise RuntimeError(
            "An emulator is already running. Please close it before starting a new one.")

    with contextlib.ExitStack() as emulator_stack, contextlib.ExitStack() as waiter_stack:
        emulator_args = [
            sdk_tool_paths.emulator,
            "-avd",
            avd_name,
            "-memory",
            "4096",
            "-timezone",
            "America/Los_Angeles",
            "-no-snapstorage",
            "-no-audio",
            "-no-boot-anim",
            "-gpu",
            "guest",
            "-delay-adb",
        ]

        # For Linux CIs we must use "-no-window" otherwise you'll get
        #   Fatal: This application failed to start because no Qt platform plugin could be initialized
        #
        # For macOS CIs use a window so that we can potentially capture the desktop and the emulator screen
        # and publish screenshot.jpg and emulator.png as artifacts to debug issues.
        #   screencapture screenshot.jpg
        #   $(ANDROID_SDK_HOME)/platform-tools/adb exec-out screencap -p > emulator.png
        #
        # On Windows it doesn't matter (AFAIK) so allow a window which is nicer for local debugging.
        if is_linux():
            emulator_args.append("-no-window")

        if extra_args is not None:
            emulator_args += extra_args

        emulator_process = emulator_stack.enter_context(_start_process(*emulator_args))
        emulator_stack.callback(_stop_process, emulator_process)

        # we're specifying -delay-adb so use a trivial command to check when adb is available.
        waiter_process = waiter_stack.enter_context(
            _start_process(
                sdk_tool_paths.adb,
                "wait-for-device",
                "shell",
                "ls /data/local/tmp",
            )
        )

        waiter_stack.callback(_stop_process, waiter_process)

        # poll subprocesses.
        # allow 20 minutes for startup as some CIs are slow. TODO: Make timeout configurable if needed.
        sleep_interval_seconds = 10
        end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)

        while True:
            waiter_ret, emulator_ret = waiter_process.poll(), emulator_process.poll()

            if emulator_ret is not None:
                # emulator exited early
                raise RuntimeError(f"Emulator exited early with return code: {emulator_ret}")

            if waiter_ret is not None:
                if waiter_ret == 0:
                    _log.debug("adb wait-for-device process has completed.")
                    break
                raise RuntimeError(f"Waiter process exited with return code: {waiter_ret}")

            if datetime.datetime.now() > end_time:
                raise RuntimeError("Emulator startup timeout")

            time.sleep(sleep_interval_seconds)

        # emulator is started
        emulator_stack.pop_all()

        # loop to check for sys.boot_completed being set.
        # in theory `-delay-adb` should be enough but this extra check seems to be required to be sure.
        while True:
            # looping on device with `while` seems to be flaky so loop here and call getprop once
            args = [
                sdk_tool_paths.adb,
                "shell",
                # "while [[ -z $(getprop sys.boot_completed) | tr -d '\r' ]]; do sleep 5; done; input keyevent 82",
                "getprop sys.boot_completed",
            ]

            _log.debug(f"Starting process - args: {args}")

            getprop_output = subprocess.check_output(args, timeout=10)
            getprop_value = bytes.decode(getprop_output).strip()

            if getprop_value == "1":
                break

            elif datetime.datetime.now() > end_time:
                raise RuntimeError("Emulator startup timeout. sys.boot_completed was not set.")

            _log.debug(
                f"sys.boot_completed='{getprop_value}'. Sleeping for {sleep_interval_seconds} before retrying.")
            time.sleep(sleep_interval_seconds)
        # Verify if the emulator is now running
        if not check_emulator_running():
            raise RuntimeError("Emulator failed to start.")
        return emulator_process


def stop_emulator(
        emulator_proc_or_pid: typing.Union[subprocess.Popen, int], timeout: int = 120
):
    """
    Stops the emulator process, checking its running status before and after stopping.

    :param emulator_proc_or_pid: The emulator process (subprocess.Popen) or PID (int).
    :param timeout: Maximum time (in seconds) to wait for the emulator to stop.
    """

    def is_emulator_running() -> bool:
        """Check if any emulator instance is running using adb."""
        try:
            output = subprocess.check_output(["adb", "devices"], text=True, timeout=10)
            running_devices = [line for line in output.splitlines() if "emulator" in line]
            return len(running_devices) > 0
        except subprocess.SubprocessError as e:
            _log.error(f"Error checking running emulators: {e}")
            return False

    if not is_emulator_running():
        _log.warning("No emulator instances are currently running.")
        return

    if isinstance(emulator_proc_or_pid, subprocess.Popen):
        _log.info("Stopping emulator using subprocess.Popen instance.")
        _stop_process(emulator_proc_or_pid)
    elif isinstance(emulator_proc_or_pid, int):
        _log.info(f"Stopping emulator with PID: {emulator_proc_or_pid}")
        _stop_process_with_pid(emulator_proc_or_pid)
    else:
        raise ValueError("Expected either a PID or subprocess.Popen instance.")

    # Loop to check if the emulator stops within the timeout
    interval = 5
    end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    while is_emulator_running():
        if datetime.datetime.now() > end_time:
            raise RuntimeError(
                f"Failed to stop the emulator within the specified timeout = {timeout} seconds.")
        _log.debug("Emulator still running. Checking again in 5 seconds...")
        time.sleep(interval)

    _log.info("Emulator stopped successfully.")
