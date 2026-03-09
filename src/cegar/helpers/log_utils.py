import re
from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter, process_time, strftime

from colorama import Fore, Style, init
from tabulate import tabulate


init(autoreset=True)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class PipelineLogger:
    use_color: bool = True
    show_time: bool = True
    tablefmt: str = "rounded_grid"
    _runtimes_by_stage: OrderedDict = field(default_factory=OrderedDict)
    _run_wall_start: float = field(default_factory=perf_counter)

    def _c(self, text, color):
        if not self.use_color:
            return str(text)
        return f"{color}{text}{Style.RESET_ALL}"

    def _b(self, text):
        if not self.use_color:
            return str(text)
        return f"{Style.BRIGHT}{text}{Style.RESET_ALL}"

    @staticmethod
    def _visible_len(text):
        return len(_ANSI_RE.sub("", str(text)))

    def _ts(self):
        if not self.show_time:
            return ""
        return self._c(f"[{strftime('%H:%M:%S')}] ", Fore.LIGHTBLACK_EX)

    def stage(self, stage_name, message=""):
        stage_tag = self._c(f"{stage_name}", Fore.CYAN + Style.BRIGHT)
        if message:
            print(f"{self._ts()}{stage_tag} {self._c(message, Fore.WHITE)}")
        else:
            print(f"{self._ts()}{stage_tag}")

    def info(self, message):
        print(f"{self._ts()}{self._c('INFO', Fore.WHITE + Style.BRIGHT)} {message}")

    def success(self, message):
        print(f"{self._ts()}{self._c('OK', Fore.GREEN + Style.BRIGHT)} {message}")

    def warn(self, message):
        print(f"{self._ts()}{self._c('WARN', Fore.YELLOW + Style.BRIGHT)} {message}")

    def error(self, message):
        print(f"{self._ts()}{self._c('ERROR', Fore.RED + Style.BRIGHT)} {message}")

    def table(self, title, rows, headers):
        print(self._c(f"{title}", Fore.MAGENTA + Style.BRIGHT))
        styled_headers = [self._b(h) for h in headers]
        print(
            tabulate(
                rows,
                headers=styled_headers,
                tablefmt=self.tablefmt,
                stralign='left',
                numalign='left',
                disable_numparse=True,
            )
        )

    def metrics(self, title, items, key_header="Metric", value_header="Value"):
        if hasattr(items, "items"):
            rows = [(str(k), v) for k, v in items.items()]
        else:
            rows = [(str(k), v) for k, v in items]
        self.table(title, rows, headers=[key_header, value_header])

    def metrics_side_by_side(self, blocks, key_header="Metric", value_header="Value", gap=6):
        """
        Render multiple 2-column metric tables in one row.

        Args:
            blocks: iterable of (title, items), where items is dict-like or sequence of pairs.
            key_header/value_header: column headers for all blocks.
            gap: number of spaces between neighboring tables.
        """
        rendered_blocks = []
        for title, items in blocks:
            if hasattr(items, "items"):
                rows = [(str(k), v) for k, v in items.items()]
            else:
                rows = [(str(k), v) for k, v in items]

            table_txt = tabulate(
                rows,
                headers=[key_header, value_header],
                tablefmt=self.tablefmt,
                stralign='left',
                numalign='left',
                disable_numparse=True,
            )
            lines = [str(title)] + table_txt.splitlines()
            width = max(self._visible_len(line) for line in lines) if lines else 0
            header_rows = {
                i for i, line in enumerate(lines)
                if (key_header in line and value_header in line)
            }
            rendered_blocks.append((lines, width, header_rows))

        if not rendered_blocks:
            return

        max_lines = max(len(lines) for lines, _, _ in rendered_blocks)
        padded = []
        for lines, width, header_rows in rendered_blocks:
            block_lines = lines + ([""] * (max_lines - len(lines)))
            padded_block = []
            for i, line in enumerate(block_lines):
                pad = max(0, width - self._visible_len(line))
                segment = line + (" " * pad)
                if i == 0:
                    segment = self._c(segment, Fore.MAGENTA + Style.BRIGHT)
                elif i in header_rows:
                    segment = self._b(segment)
                padded_block.append(segment)
            padded.append(padded_block)

        spacer = " " * int(gap)
        for i in range(max_lines):
            print(spacer.join(block[i] for block in padded))

    def runtime_line(self, cpu_seconds, wall_seconds, label="Runtime"):
        print(
            f"{self._ts()}{self._c('TIME', Fore.BLUE + Style.BRIGHT)} "
            f"{label}: CPU {float(cpu_seconds):.6f}s | Wall {float(wall_seconds):.6f}s"
        )

    def record_runtime(self, stage, task, cpu_seconds, wall_seconds):
        stage = str(stage)
        task = str(task)
        if stage not in self._runtimes_by_stage:
            self._runtimes_by_stage[stage] = []
        self._runtimes_by_stage[stage].append((task, float(cpu_seconds), float(wall_seconds)))

    def runtime_summary_by_stage(self, title="Runtime Summary By Stage"):
        if not self._runtimes_by_stage:
            return

        rows = []
        staged_items = list(self._runtimes_by_stage.items())
        for idx, (stage, items) in enumerate(staged_items):
            rows.append((stage, "", "", ""))
            for task, cpu, wall in items:
                rows.append(("", task, f"{cpu:.6f}s", f"{wall:.6f}s"))
            if idx < len(staged_items) - 1:
                rows.append(("", "", "", ""))

        self.table("Runtime Summary", rows, headers=["Stage", "Task", "CPU", "Wall"])

    def total_wall_seconds(self):
        return perf_counter() - self._run_wall_start

    def reset_runtimes(self):
        self._runtimes_by_stage.clear()


def timed_call(func, *args, **kwargs):
    """
    Execute a callable and return its output with CPU and wall-clock runtimes.

    Returns:
        (result, cpu_seconds, wall_seconds)
    """
    cpu_start = process_time()
    wall_start = perf_counter()
    result = func(*args, **kwargs)
    cpu_dt = process_time() - cpu_start
    wall_dt = perf_counter() - wall_start
    return result, cpu_dt, wall_dt
