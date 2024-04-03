"""Transform Python scripts into Jupyter notebook. Allows to split into two different notebooks. 
Typically, one with the questions, and one with responses.

Michel Bierlaire
Fri Mar 29 17:37:16 2024
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import NamedTuple
import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


class SplitTuple(NamedTuple):
    separator: str
    before_file_extension: str
    after_file_extension: str


class Block(ABC):
    def __init__(self, lines: list[str]):
        self.lines: list[str] = lines
        self._start: int | None = None
        self._end: int | None = None

    @property
    def start(self) -> int:
        return self._start  # Return the value of the attribute

    @start.setter
    def start(self, value: int) -> None:
        self._start = value  # Set the new value for the attribute
        self.end = None
        if value is not None:
            self.identifies_block()

    @property
    def end(self) -> int:
        return self._end  # Return the value of the attribute

    @end.setter
    def end(self, value: int) -> None:
        self._end = value  # Set the new value for the attribute
        if value is not None:
            self.clean_block()

    @abstractmethod
    def first_line(self, index: int) -> bool:
        """Check if the line indexed corresponds to the first of a block. The actual implementation depends
        on the type of block"""
        pass

    def process_block(self, index: int) -> None:
        """Sets the starting point"""
        if not self.first_line(index):
            raise AssertionError('Line {line} does not start the block')
        self.start = index

    @abstractmethod
    def identifies_block(self) -> None:
        """Sets the end point"""
        pass

    @abstractmethod
    def clean_block(self) -> None:
        pass

    def get_block(self) -> list[str]:
        if self.start is None:
            raise ValueError('The beginning of the block has not been identified yet.')
        return self.lines[self.start : self.end]

    @abstractmethod
    def get_cell(self) -> NotebookNode:
        pass

    def is_block_empty(self) -> bool:
        if self.start is None or self.end is None:
            raise ValueError('{self.start=} {self.end=}')
        return all(line.strip() == '' for line in self.get_block())


class MarkdownBlock(Block):
    """A markdown block is a sequence of lines starting with #"""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)

    def first_line(self, index: int) -> bool:
        if index >= len(self.lines):
            error_msg = f'Index {index} cannot exceed {len(self.lines)-1}'
            raise IndexError(error_msg)
        if self.lines[index].startswith('#'):
            return True
        return False

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError('The beginning of the block has not been identified yet.')
        if self.end is not None:
            raise AssertionError('Should not be called again')
        for i in range(self.start, len(self.lines)):
            if not self.lines[i].startswith('#'):
                if self.end is not None and self.end != i:
                    raise ValueError(f'{self.start=}  d{self.end=} {i=}')
                self.end = i
                return
        else:
            self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            raise AssertionError('Both should be set when this function is called')
        for i in range(self.start, self.end):
            self.lines[i] = self.lines[i][1:].lstrip().rstrip('\n')

    def get_cell(self) -> NotebookNode:
        the_block = self.get_block()
        return new_markdown_cell('\n'.join(the_block))


class DocstringBlock(Block):
    """A docstring block is a sequence of lines between triple quotations mark"""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)

    def first_line(self, index: int) -> bool:
        if index >= len(self.lines):
            error_msg = f'Index {index} cannot exceed {len(self.lines)-1}'
            raise IndexError(error_msg)
        if self.lines[index].startswith('"""'):
            self.closing_sequence = '"""'
            return True
        if self.lines[index].startswith("'''"):
            self.closing_sequence = "'''"
            return True
        return False

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError('The beginning of the block has not been identified yet.')
        if self.end is not None:
            raise AssertionError('Should not be called again')
        for i in range(self.start + 1, len(self.lines)):
            if self.lines[i].startswith(self.closing_sequence):
                self.end = i + 1
                return
        else:
            self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            raise AssertionError('Both should be set')
        # Remove the triple quotes
        self.lines[self.start] = self.lines[self.start][3:]
        self.lines[self.end - 1] = self.lines[self.end - 1][3:]
        for i in range(self.start, self.end):
            self.lines[i] = self.lines[i].lstrip()

    def get_cell(self) -> NotebookNode:
        the_block = self.get_block()
        return new_markdown_cell('\n'.join(the_block))


class CodeBlock(Block):
    """A code block is anything else"""

    def __init__(self, lines: list[str], other_blocks: list[Block]) -> None:
        super().__init__(lines)
        self.other_blocks: list[Block] = other_blocks

    def first_line(self, index: int) -> bool:
        if index >= len(self.lines):
            error_msg = f'Index {index} cannot exceed {len(self.lines)-1}'
            raise IndexError(error_msg)

        # The first line of a code block is any line not the first line of another block
        for other in self.other_blocks:
            if other.first_line(index=index):
                return False
        return True

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError('The beginning of the block has not been identified yet.')
        if self.end is not None:
            raise AssertionError('Should not be called again')
        for i in range(self.start, len(self.lines)):
            for other in self.other_blocks:
                if other.first_line(index=i):
                    self.end = i
                    return
        else:
            self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            return
        for i in range(self.start, self.end):
            self.lines[i] = self.lines[i].rstrip()

    def get_cell(self) -> NotebookNode:
        the_block = self.get_block()
        return new_code_cell('\n'.join(the_block))


def extract_next_block(lines: list[str]) -> NotebookNode:
    start_index = 0
    a_markdown_block = MarkdownBlock(lines=lines)
    a_docstring_block = DocstringBlock(lines=lines)
    a_code_block = CodeBlock(
        lines=lines, other_blocks=[a_docstring_block, a_markdown_block]
    )
    all_blocks = [a_docstring_block, a_markdown_block, a_code_block]
    while start_index < len(lines):
        for block in all_blocks:
            if block.first_line(start_index):
                block.process_block(index=start_index)
                start_index = block.end
                if not block.is_block_empty():
                    the_block = block.get_cell()
                    yield the_block
                    break


def split_string(original_line: str, separator: str) -> tuple[str, str]:
    """Split the line into two versions"""
    # Split the string
    parts = original_line.split(separator)

    # Extract parts before and after '####'
    before = parts[0].rstrip()  # Remove trailing spaces from the first part
    after = (
        parts[1] if len(parts) > 1 else before
    )  # Set the second part, or or copy the first one if '####' is not found

    return before, after


def preprocess(lines: list[str], separator: str) -> tuple[list[str], list[str]]:
    list_of_splitted_lines = [
        split_string(original_line=line, separator=separator) for line in lines
    ]
    list_before = [term[0] for term in list_of_splitted_lines]
    list_after = [term[1] for term in list_of_splitted_lines]
    return list_before, list_after


def generate_notebook(lines: list[str], filename: str) -> None:
    notebook = new_notebook()

    for cell in extract_next_block(lines):
        notebook.cells.append(cell)

    # Write the notebook to the new file
    with open(filename, 'w', encoding='utf-8') as file:
        nbformat.write(notebook, file)
    print(f'File {filename} created.')


def script_to_notebook(input_script: str, splitting: SplitTuple | None):

    with open(input_script, 'r') as file:
        lines = file.readlines()

    base_name = os.path.splitext(input_script)[0]  # Remove the .py extension

    if splitting:
        before_lines, after_lines = preprocess(
            lines=lines, separator=splitting.separator
        )

        before_notebook = f'{base_name}_{splitting.before_file_extension}.ipynb'
        generate_notebook(lines=before_lines, filename=before_notebook)
        after_notebook = f'{base_name}_{splitting.after_file_extension}.ipynb'
        generate_notebook(lines=after_lines, filename=after_notebook)
        return

    notebook = f'{base_name}.ipynb'
    generate_notebook(lines=lines, filename=notebook)


if __name__ == '__main__':
    splitting = SplitTuple(
        separator='####',
        before_file_extension='responses',
        after_file_extension='questions',
    )
    if len(sys.argv) != 2:
        print('Usage: python script_to_notebook.py')
    else:
        input_filename = sys.argv[1]
        script_to_notebook(input_filename, splitting=splitting)
