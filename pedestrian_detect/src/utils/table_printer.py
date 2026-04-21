"""Pretty-print aligned tables to text files (port of C# TableCell logic)."""


class TableCell:
    """A single cell in a formatted text table."""

    DELIMITER_MARKER = "*"

    def __init__(self, content: str, left_aligned: bool = True):
        self.content = content
        self.left_aligned = left_aligned
        self._is_header = False

    def set_as_header(self):
        self._is_header = True

    def is_header(self) -> bool:
        return self._is_header

    def aligned_content(self, width: int) -> str:
        if self.left_aligned:
            return self.content.ljust(width)
        else:
            return self.content.rjust(width)

    def content_as_header(self, width: int) -> str:
        if len(self.content) >= width:
            return self.content[:width]
        diff = width - len(self.content)
        left_pad = diff // 2
        right_pad = diff - left_pad
        return " " * left_pad + self.content + " " * right_pad


def print_table(table: list, stream) -> None:
    """Print a formatted table to a stream.

    Args:
        table: List of rows, where each row is a list of TableCell objects.
        stream: File-like object with write() and flush() methods.
    """
    if not table:
        return

    col_count = len(table[0])
    col_sizes = [0] * col_count

    # Determine column widths
    for row in table:
        for col_idx, cell in enumerate(row):
            if cell.is_header():
                break
            if len(cell.content) > col_sizes[col_idx]:
                col_sizes[col_idx] = len(cell.content)

    # Build delimiter string
    delimiter = "+"
    for col_size in col_sizes:
        delimiter += "-" * (2 + col_size) + "+"

    header_width = len(delimiter) - 2

    for row in table:
        if row[0].content == TableCell.DELIMITER_MARKER:
            stream.write(delimiter + "\n")
            continue
        elif row[0].is_header():
            stream.write("|")
            stream.write(row[0].content_as_header(header_width))
            stream.write("|\n")
            continue

        stream.write("|")
        for col_idx, cell in enumerate(row):
            stream.write(" ")
            stream.write(cell.aligned_content(col_sizes[col_idx]))
            stream.write(" |")
        stream.write("\n")
