"""Some useful matrix functions

Michel Bierlaire
Mon Mar 18 17:30:56 2024
"""

import itertools

import numpy as np


def print_string_matrix(
    matrix: list[list[str]], headers: list[str] | None = None
) -> str:
    """
    Prints a matrix of strings
    :param matrix: matrix of strings to display.
    :param headers: headers of each column.
    :return: formatted string.
    """
    if not matrix:
        return ''

    # Determine the number of columns
    num_columns = len(matrix[0])

    # If headers are not provided, create an empty list of the right size
    if headers is None:
        headers = [''] * num_columns
    else:
        # Ensure headers length matches the data's column count
        assert (
            len(headers) == num_columns
        ), 'Headers length must match data column count.'

    # Determine the maximum width of each column
    column_widths = [
        max(len(str(item)) for item in ([header] + [row[i] for row in matrix]))
        for i, header in enumerate(headers)
    ]

    # Prepare format strings
    header_format = ' | '.join('{{:<{}}}'.format(width) for width in column_widths)
    row_format = header_format  # Same format for rows

    # Initialize an empty string to build the output
    output_str = ''

    # Build the headers string if they are not empty
    if any(headers):
        output_str += header_format.format(*headers) + '\n'
        # Add a separator
        output_str += (
            '-' * sum(column_widths) + '-' * (len(headers) - 1) * 3 + '\n'
        )  # Adjusting for the separators

    # Build the rows string
    for row in matrix:
        output_str += row_format.format(*row) + '\n'

    # Remove the last newline character for a cleaner output
    if output_str.endswith('\n'):
        output_str = output_str[:-1]

    return output_str


def find_opposite_columns(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Identifies pairs of columns where one column is the exact opposite of the other

    :param matrix: input matrix
    :return: list of pairs of matching indices
    """
    # List to hold pairs of indices
    opposite_pairs = []

    # Number of columns
    n_cols = matrix.shape[1]

    # Iterate over all unique pairs of columns
    for i, j in itertools.combinations(range(n_cols), 2):
        # Check if one column is the negative of the other
        if np.array_equal(matrix[:, i], -matrix[:, j]):
            opposite_pairs.append((i, j))

    return opposite_pairs


def find_columns_multiple_identity(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Find columns with only one non-zero entry.

    :param matrix: input matrix
    :return: list of pairs (column_index, row_index) identifying the non-zero element.
    """
    # List to hold the indices of the special columns and the row of the non-zero term
    special_columns = []

    # Number of rows and columns
    n_rows, n_cols = matrix.shape

    # Iterate through each column
    for col in range(n_cols):
        # Find indices of non-zero elements in the current column
        non_zero_indices = np.nonzero(matrix[:, col])[0]

        # Check if the column contains exactly one non-zero element
        if len(non_zero_indices) == 1:
            # Add the column index and the row index of the non-zero element
            special_columns.append((col, int(non_zero_indices[0])))

    return special_columns
