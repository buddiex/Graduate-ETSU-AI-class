from typing import List, Any, Dict


def display(p_board: List[Any]) -> None:
    """
        Generic display method that a board (an nXn list) passed via pBoard
    """
    for r in range(size):
        line = " "
        for c in range(size):
            if p_board[r][c] == 0:
                st = "0"
            else:
                st = "."
            line += st + " "
        print(line)


def addToThreats(row: int, col: int, change: int) -> None:
    """
        Propagates constraints to the:
            col - Current column passed. past the row
            diagonals - the Diagonals of row,col
        Change of +1 adds a threat (increases constraint) at each location
        Change of -1 removes a threat (decreases constraint) at each location
    """
    # Change the column past the current row
    for j in range(0, row):
        board[j][col] += change
    # Change the diagonal beyond the current row
    for j in range(row + 1, size):
        board[j][col] += change
        if col + (j - row) < size:
            board[j][col + (j - row)] += change
        if col - (j - row) >= 0:
            board[j][col - (j - row)] += change


def backtracking_search(depth: int = 0) -> bool:
    """
        Recursive version of the backtracking search
    """

    if depth == size:
        display(board)
        return True
    else:
        for c in range(size):
            if board[depth][c] == 0:
                addToThreats(depth, c, 1)
                status = backtracking_search(depth + 1)
                if status:
                    return True
                addToThreats(depth, c, -1)
    return False


"""
********************************
   MAIN ALGORITHM
********************************
"""
queens: Dict[str, str] = {}  # queens[i] contains the column of each queen
# size = int(sys.argv[1]) # Get the board size from the command line
size = 4  # Get the board size from the command line

# Create the board as a two-dimensional list
board = [[0 for c in range(size)] for r in range(size)]

if not backtracking_search():
    print("NO SOLUTION!")
