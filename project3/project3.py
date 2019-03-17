import sys


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
        # self.search(self.propagate())
        # self.solve_FC()
        val = self.solve_bt_fc(self.values)
        if val:
            for k, v in val.items():
                self.grid[k] = v

    def solve_FC(self):
        """
            Solve the problem by propagation and backtracking.
        """

        for k in self.grid:
            if self.forward_check(self.values):
                print("not solvable")
                return False
            if self.grid[k] in '0.':
                self.grid[k] = self.values[k][0]
                self.propagate(k, self.values[k][0], self.values)
                # self.update_grid()
            # print(self)

    def solve_mrv(self):
        pass

    def solve_bt_fc(self, values):
        if self.forward_check(values):
            return False

        if self.is_solved(values):
            return values

        for cell, domain in values.items():
            if len(domain) > 1:
                for variable in domain:
                    values_new = self.propagate(cell, variable, values.copy())
                    solved = self.solve_bt_fc(values_new)
                    if solved:
                        return solved
                return False

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

    def update_grid(self):
        for k in self.grid:
            if self.grid[k] in '0.':
                if len(self.values[k]) == 1:
                    self.grid[k] = self.values[k]

    def propagate(self, cell, value, values):
        """
            TODO: Code the Constraint Propagation Technique Here
        """
        values[cell] = value
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
                self.values = self.propagate(cell, value, self.values)


def main():
    s = Sudoku()
    '''
        The loop reads in as many files as you've passed on the command line.
        Example to read two easy files from the command line:
            python project3.py sudoku_easy1.txt sudoku_easy2.txt
    '''
    for x in range(1, len(sys.argv)):
        s.load_file(sys.argv[x])
        print("\n==============================================")
        print(sys.argv[x].center(46))
        print("==============================================\n")
        print(s)
        print("\n----------------------------------------------\n")
        s.solve()
        print(s)


main()
