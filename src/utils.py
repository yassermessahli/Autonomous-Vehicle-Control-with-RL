import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_sumo_output():
    """
    Captures C-level stdout/stderr (which SUMO uses) and Python-level stdout
    to prevent the terminal from flooding during env.reset().
    """
    # Open a null file
    with open(os.devnull, "w") as devnull:
        # Save old file descriptors
        old_stdout_fd = sys.stdout.fileno()
        old_stderr_fd = sys.stderr.fileno()

        # Duplicate the original file descriptors so we can restore them
        saved_stdout_fd = os.dup(old_stdout_fd)
        saved_stderr_fd = os.dup(old_stderr_fd)

        try:
            # Redirect Python streams to devnull
            sys.stdout.flush()
            sys.stderr.flush()

            # Redirect C-level file descriptors to devnull
            os.dup2(devnull.fileno(), old_stdout_fd)
            os.dup2(devnull.fileno(), old_stderr_fd)

            yield
        finally:
            # Restore everything
            os.dup2(saved_stdout_fd, old_stdout_fd)
            os.dup2(saved_stderr_fd, old_stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
