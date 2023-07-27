"""File pareto.py

:author: Michel Bierlaire, EPFL
:date: Tue Jul  4 17:11:29 2023

Implement a Pareto set for generic purposes.
"""
import logging
from datetime import datetime
import tomlkit as tk
from biogeme_optimization.exceptions import OptimizationError

try:
    import matplotlib.pyplot as plt

    CAN_PLOT = True
except ModuleNotFoundError:
    CAN_PLOT = False

DATE_TIME_STRING = '__DATETIME__'

def replace_date_time(a_string):
    """ Replaces the string defined above by the current time and date

    :param a_string: the string to be modified
    :type a_string: str

    :return: the modified string, if the modification has been made. None otherwise
    :rtpye: str
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%B %d, %Y. %H:%M:%S")        
    return a_string.replace(f'{DATE_TIME_STRING}', formatted_datetime)
    
logger = logging.getLogger(__name__)


class SetElement:
    """Specify the elements of the Pareto set. Note that each
    objective is supposed to be minimized.

    """

    def __init__(self, element_id, objectives):
        """Ctor

        :param element_id: identifier of the element
        :type element_id: str

        :param objectives: values of the objective functions
        :type objectives: list(float)
        """
        self.element_id = element_id
        self.objectives = objectives

        if any(obj is None for obj in objectives):
            raise OptimizationError(f'All objectives ust be defined: {objectives}')

    def __eq__(self, other):
        if isinstance(other, SetElement):
            if self.element_id == other.element_id:
                if self.objectives != other.objectives:
                    error_msg = (
                        f'Two elements named {self.element_id} have different '
                        f'objective values: {self.objectives} and '
                        f'{other.objectives}'
                    )
                    raise OptimizationError(error_msg)
                return True
        return False

    def __hash__(self):
        return hash((self.element_id, tuple(self.objectives)))

    def __str__(self):
        return f'{self.element_id} {self.objectives}'

    def __repr__(self):
        return self.element_id

    def dominates(self, other):
        """Verifies if self dominates other.

        :param other: other element to check
        :type other: SetElement
        """
        if len(self.objectives) != len(other.objectives):
            raise OptimizationError(
                f'Incompatible sizes: '
                f'{len(self.objectives)}'
                f' and {len(other.objectives)}'
            )

        if any(my_f > her_f for my_f, her_f in zip(self.objectives, other.objectives)):
            return False

        return self.objectives != other.objectives


class Pareto:
    """This object manages a Pareto set for a list of objectives that
    are each minimized.
    """

    def __init__(self, filename=None):
        self.size_init_pareto = 0
        self.size_init_considered = 0
        self.size_init_invalid = 0
        self.filename = filename
        self.comments = [f'File automatically created on {DATE_TIME_STRING}']
        """Comment to be inserted in the file when dumped, where
            __DATETIME__ is replaced by the current date and time
        
        """
        self.pareto = set()
        """dict containing the pareto set. The keys are the solutions
            (class :class:`biogeme.vns.solutionClass`), and the values are
            the size of the neighborhood that must be applied the next
            time tat the solution is selected by the algorithm.

        """

        self.removed = set()
        """set of solutions that have been in the Pareto set ar some point,
            but have been removed because dominated by another
            solution.
        """

        self.considered = set()
        """set of solutions that have been considered at some point by the
            algorithm
        """

        self.invalid = set()
        """set of solutions that have been deemed invalid
        """

        if filename is not None:
            if self.restore():
                logger.debug('RESTORE PARETO FROM FILE')
                logger.info(
                    f'Pareto set initialized from file with '
                    f'{self.size_init_considered} elements '
                    f'[{self.size_init_pareto} Pareto] and '
                    f'{self.size_init_invalid} invalid elements.'
                )
                logger.debug('RESTORE PARETO FROM FILE: DONE')

            else:
                logger.info(f'Unable to read file {filename}. Pareto set empty.')

    def __str__(self):
        return (
            f'Pareto: {self.pareto} Removed: {self.removed} '
            f'Considered: {self.considered} Invalid: {self.invalid}'
        )

    def dump(self):
        """
        Dump the three sets on a file

        :param comments: comments to include in the file, each entry on a different row
        :type comment: list(str)

        :raise OptimizationError: if a problem has occured during dumping.
        """
        if self.filename is None:
            logger.warning('No Pareto file has been provided')
            return
        doc = tk.document()
        final_comments = [replace_date_time(comment) for comment in self.comments]
        for comment in final_comments:
            doc.add(tk.comment(comment))

        pareto_table = tk.table()
        for elem in self.pareto:
            pareto_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Pareto'] = pareto_table

        considered_table = tk.table()
        for elem in self.considered:
            considered_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Considered'] = considered_table

        removed_table = tk.table()
        for elem in self.removed:
            removed_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Removed'] = removed_table

        invalid_table = tk.table()
        for elem in self.invalid:
            invalid_table[elem.element_id] = [float(obj) for obj in elem.objectives]
        doc['Invalid'] = invalid_table

        with open(self.filename, 'w', encoding='utf-8') as f:
            print(tk.dumps(doc), file=f)

    def get_element_from_id(self, the_id):
        """Returns the element of a set given its ID

        :param the_id: identifiers of the element to return
        :type the_id: str

        :return: found element, or None if element not present
        :rtype: SetElement
        """
        found = {elem for elem in self.considered if elem.element_id == the_id}
        if len(found) == 0:
            return None
        if len(found) > 1:
            error_msg = f'There are {len(found)} elements with ID {the_id}'
            raise OptimizationError(error_msg)
        return next(iter(found))

    def parse_set_from_document(self, document, set_name):
        """Parse a set from the document based on the set name.

        :param document: parsed document
        :type document: tomlkit.document

        :param set_name: name of the set to extract
        :type set_name: str

        :return: set of SetElement objects
        :rtype: set
        """
        try:
            result = {
                SetElement(id, list(values))
                for id, values in document[set_name].items()
            }
        except tk.exceptions.NonExistentKey:
            logger.warning(f'No {set_name} section in pareto file')
        return result

    def restore(self):
        """Restore the Pareto from a file"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
                document = tk.parse(content)
        except FileNotFoundError:
            return False

        self.pareto = self.parse_set_from_document(document, 'Pareto')
        self.considered = self.parse_set_from_document(document, 'Considered')
        self.removed = self.parse_set_from_document(document, 'Removed')
        self.invalid = self.parse_set_from_document(document, 'Invalid')

        self.size_init_pareto = len(self.pareto)
        self.size_init_considered = len(self.considered)
        self.size_init_invalid = len(self.invalid)
        return True

    def length(self):
        """
        Obtain the length of the pareto set.
        """
        return len(self.pareto)

    def length_of_all_sets(self):
        """
        Obtain the length of the four sets.
        """
        return (
            len(self.pareto),
            len(self.considered),
            len(self.removed),
            len(self.invalid),
        )

    def add_invalid(self, element):
        """

        :param element: invalid element to be stored
        :type element: solutionClass

        :return: True if element has been included. False otherwise.
        :rtype: bool

        """
        if element in self.invalid:
            warning_msg = f'Invalid element {element.element_id} has already been inserted in the set'
            logger.debug(warning_msg)
            return False

        self.invalid.add(element)

    def add(self, element):
        """

        - We define the set D as the set of members of the current
          Pareto set that are dominated by the element elem.

        - We define the set S as the set of members of the current
          Pareto set that dominate the element elem.

        If S is empty, we add elem to the Pareto set, and remove all
        elements in D.

        :param element: element to be considered for inclusion in the Pareto set.
        :type element: solutionClass

        :return: True if elemenet has been included. False otherwise.
        :rtype: bool

        """
        if element in self.considered:
            warning_msg = (
                f'Elem {element.element_id} has already been inserted in the set'
            )
            logger.debug(warning_msg)
            return False

        self.considered.add(element)
        S_dominated = set()
        D_dominating = set()
        for k in self.pareto:
            if element.dominates(k):
                D_dominating.add(k)
            if k.dominates(element):
                S_dominated.add(k)
        if S_dominated:
            return False
        self.pareto.add(element)
        self.pareto = {k for k in self.pareto if k not in D_dominating}
        self.removed |= D_dominating
        return True

    def statistics(self):
        """Report some statistics about the Pareto set

        :return: tuple of messages, possibly empty.
        :rtype: tuple(str)
        """
        if self.pareto is None:
            return tuple()
        msg = (
            f'Pareto: {len(self.pareto)} ',
            f'Condidered: {len(self.considered)} ',
            f'Removed: {len(self.removed)}',
        )
        return msg

    def plot(
        self,
        objective_x=0,
        objective_y=1,
        label_x=None,
        label_y=None,
        margin_x=5,
        margin_y=5,
        ax=None,
    ):
        """Plot the members of the set according to two
            objective functions.  They  determine the x- and
            y-coordinate of the plot.

        :param objective_x: index of the objective function to use for the x-coordinate.
        :param objective_x: int

        :param objective_y: index of the objective function to use for the y-coordinate.
        :param objective_y: int

        :param label_x: label for the x_axis
        :type label_x: str

        :param label_y: label for the y_axis
        :type label_y: str

        :param margin_x: margin for the x axis
        :type margin_x: int

        :param margin_y: margin for the y axis
        :type margin_y: int

        :param ax: matplotlib axis for the plot
        :type ax: matplotlib.Axes

        """
        if not CAN_PLOT:
            raise OptimizationError('Install matplotlib.')
        ax = ax or plt.gca()

        if self.length() == 0:
            raise OptimizationError('Cannot plot an empty Pareto set')

        first_elem = next(iter(self.pareto))
        number_of_objectives = len(first_elem.objectives)

        if number_of_objectives < 2:
            raise OptimizationError(
                'At least two objectives functions are required for the plot of '
                'the Pateto set.'
            )

        if objective_x >= number_of_objectives:
            error_msg = (
                f'Index of objective x is {objective_x}, but there are '
                f'only {number_of_objectives}. Give a number between 0 '
                'and {number_of_objectives-1}.'
            )
            raise OptimizationError(error_msg)

        par_x = [elem.objectives[objective_x] for elem in self.pareto]
        par_y = [elem.objectives[objective_y] for elem in self.pareto]

        con_x = [elem.objectives[objective_x] for elem in self.considered]
        con_y = [elem.objectives[objective_y] for elem in self.considered]

        rem_x = [elem.objectives[objective_x] for elem in self.removed]
        rem_y = [elem.objectives[objective_y] for elem in self.removed]

        inv_x = [elem.objectives[objective_x] for elem in self.invalid]
        inv_y = [elem.objectives[objective_y] for elem in self.invalid]

        ax.axis(
            [
                min(par_x) - margin_x,
                max(par_x) + margin_x,
                min(par_y) - margin_y,
                max(par_y) + margin_y,
            ]
        )
        ax.plot(par_x, par_y, 'o', label='Pareto')
        ax.plot(rem_x, rem_y, 'x', label='Removed')
        ax.plot(con_x, con_y, ',', label='Considered')
        if self.invalid:
            ax.plot(inv_x, inv_y, '*', label='Invalid')
        if label_x is None:
            label_x = f'Objective {objective_x}'
        if label_y is None:
            label_y = f'Objective {objective_y}'

        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.legend()
        return ax
