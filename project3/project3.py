import sys
import time


class Sudoku:
    """
        Sudoku class, which models a Sudoku game.

        Based on Peter Norvig's Suggested Sudoku setup
    """

    def __init__(self):
        """
            Initialize digits, rows, columns, the grid, squares, units, peers, and values.
        """
        self.digits = '123456789'
        self.rows = 'ABCDEFGHI'
        self.cols = self.digits
        self.grid = dict()
        self.squares = self.cross_product(self.rows, self.cols)
        unitlist = ([self.cross_product(self.rows, c) for c in self.cols] + \
                    [self.cross_product(r, self.cols) for r in self.rows] + \
                    [self.cross_product(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
        self.units = dict((s, [u for u in unitlist if s in u]) for s in self.squares)
        self.peers = dict((s, set(sum(self.units[s], [])) - set([s])) for s in self.squares)
        self.values = dict((s, self.digits) for s in self.squares)
        # pprint.pprint(self.units)

    @staticmethod
    def cross_product(A, B):
        """
            Return the cross product of A and B
        """
        return [a + b for a in A for b in B]

    def __str__(self):
        """
            Convert the grid into a human-readable string
        """
        output = ''
        width = 2 + max(len(self.grid[s]) for s in self.squares)
        line = '+'.join(['-' * (width * 3)] * 3)
        for r in self.rows:
            output += (''.join(
                (self.grid[r + c] if self.grid[r + c] not in '0.' else '').center(width) + ('|' if c in '36' else '')
                for c in self.digits)) + "\n"
            if r in 'CF': output += line + "\n"
        return output

    def load_file(self, filename):
        """
            Load the file into the grid dictionary. Note that keys
            are in the form '[A-I][1-9]' (e.g., 'E5').
        """
        with open(filename) as f:
            grid = ''.join(f.readlines())
        grid_values = self.grid_values(grid)
        self.grid = grid_values

    def grid_values(self, grid):
        """
            Convert grid into a dict of {square: char} with '0' or '.' for empties.
        """
        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))

    def solve(self):
        self.init_values()
        ###############################################
        # uncomment each line to run the right search
        ################################################
        # result = self.search_FC(self.propagate(self.values))
        # result = self.search_BT_FC(self.propagate(self.values))
        result = self.search_MRV(self.propagate(self.values))
        if result:
            self.copy_to_grid(result)
        else:
            print('could not solve')

    def copy_to_grid(self, val):
        for k, v in val.items():
            self.grid[k] = v

    def search_FC(self, values):

        if self.is_solved(values):
            return values

        for cell, domain in values.items():
            if len(domain) > 1:
                for variable in domain:
                    values[cell] = variable
                    self.search_FC(self.propagate(values))

    def search_BT_FC(self, values):
        """" solving with backtracking based on forward checking"""
        if self.forward_check(values):
            return False

        if self.is_solved(values):
            return values

        for cell, domain in values.items():
            if len(domain) > 1:
                for variable in domain:
                    # passing a copy of values into the next call ensures that the state of current values
                    # remains intact. hence no need for explicitly un propagating
                    values[cell] = variable
                    solved = self.search_BT_FC(self.propagate(values.copy()))
                    if solved:
                        return solved
                return False

    def search_MRV(self, values):
        """ Using minimum remaining values to pick next cell to fill"""

        if self.forward_check(values):
            return False

        if self.is_solved(values):
            return values

        cell, domain = self.get_cell_min_remaining_constraint(values)
        if len(domain) > 1:
            for variable in domain:
                # passing a copy of values into the next call ensures that the state of current values
                # remains intact. hence no need for explicitly un propagating
                values[cell] = variable
                solved = self.search_MRV(self.propagate(values.copy()))
                if solved:
                    return solved
            return False

    def get_cell_min_remaining_constraint(self, values):
        return min([(k, v) for k, v in values.items() if len(v) > 1], key=lambda x: len(x[1]))

    @staticmethod
    def forward_check(values):
        return any([v == '' for i, v in values.items()])

    @staticmethod
    def solved(values):
        return all([len(v) == 1 for i, v in values.items()])

    def is_solved(self, values):
        if not all([len(v) == 1 for i, v in values.items()]):
            return False

        for unit in self.units:
            for section in self.units[unit]:
                val_str = ''
                for cell in section:
                    val_str += values[cell]
                if "".join(sorted(val_str)) != self.digits:
                    return False
        return True

    def propagate(self, values):
        """
            TODO: Code the Constraint Propagation Technique Here
        """
        # looping through all single constraints as a result of propagate
        # and propagating till there is no new single values been created
        while True:
            already_solved = [c for c in values if len(values[c]) == 1]
            for cell, value in values.items():
                if len(value) == 1:
                    values = self.__propagate(cell, value, values)

            just_solved = [c for c in values if len(values[c]) == 1]
            if already_solved == just_solved:
                break
        return values

    def __propagate(self, cell, value, values):
        for peer in self.peers[cell]:
            values[peer] = values[peer].replace(value, '')
        return values

    def search(self, values):
        """
            TODO: Code the Backtracking Search Technique Here
        """

        return values

    def init_values(self):
        for cell, value in self.grid.items():
            if value not in '0.':
                self.values[cell] = value


def main():
    '''
        The loop reads in as many files as you've passed on the command line.
        Example to read two easy files from the command line:
            python project3.py sudoku_easy1.txt sudoku_easy2.txt
    '''
    for x in range(1, len(sys.argv)):
        file = sys.argv[x]
        s = Sudoku()
        s.load_file(file)
        print("\n==============================================")
        print(sys.argv[x].center(46))
        print("==============================================\n")
        print(s)
        print("\n----------------------------------------------\n")
        start = time.time()
        s.solve()
        t = time.time() - start
        print('(%.2f seconds)\n' % t)
        print(s)


main()
