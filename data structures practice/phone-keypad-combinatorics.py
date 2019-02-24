from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    co_ordinate: tuple
    steps_taken: int
    linage: list


def required_function(d:int, n: int)->list:
    assert 0 <= d <= 9, "digits must be between 0 and 9"
    out = []
    if n == 1:
        return [str(d)]

    pad = []
    graph = defaultdict(list)
    movement = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    start_node = None
    # populate the pad and graph
    for r in range(4):
        inner = []
        for c in range(3):
            node = Node("fd", (r, c), 0, [])
            if (r, c) in [(3, 0), (3, 2)]:
                node.key = None
            if (r, c) == (3, 1):
                node.key = 0
            inner.append(node)
            # populate graph
            for m in movement:
                if 0 <= r + m[0] <= 3 and 0 <= c + m[1] <= 2:
                    graph[(r, c)].append((r + m[0], c + m[1]))

            if node.co_ordinate == (3, 1) and d == 0:
                start_node = node
            elif node.key == d:
                start_node = node
        pad.append(inner)

    get_node = lambda i, j: pad[i][j]
    start_node.steps_taken = 1
    start_node.linage.append(start_node.key)
    dq = deque([start_node])

    while dq:
        current_node = dq.popleft()
        for child in graph[current_node.co_ordinate]:
            child = get_node(*child)
            if current_node.steps_taken < n and child.key:
                child.steps_taken = current_node.steps_taken + 1
                if child.key in current_node.linage:
                    child.linage.extend(current_node.linage[::-1])
                else:
                    child.linage.extend(current_node.linage)
                child.linage.append(child.key)
                dq.appendleft(child)

    get_node(*start_node.co_ordinate).linage.pop()
    for l in [k.linage for p in pad for k in p if k.steps_taken == n]:
        for i in range(0, len(l), n):
            out.append("".join(map(str, l[i:i + n])))
    return out


rtn = required_function(2, 3)
print(rtn)

"""# Phone Key-Pad Combinatorics

A standard phone number pad has the following layout:

```txt
1 2 3
4 5 6
7 8 9
  0
```

Using this layout and a starting digit we can generate numbers as follows:

- The first digit is the starting digit and each subsequent digit is directly left, right, above or below the previous digit on the number pad.

> For example if the starting digit is 1, 1256 is a valid number, but 1289 is not valid because 8 is not directly next to 2.

## Your Task

Write a function that takes a starting digit d and an integer n as input and returns a list of all unique numbers of length n generated in this way.

### Test cases

```txt
f(5, 1) = [5]
f(1, 3) = [121, 123, 125, 141, 145, 147]
f(2, 3) = [ 212, 214, 232, 236, 254, 256, 252, 258 ]
f(3, 3) = [ 321, 323, 325, 365, 363, 369 ]
f(4, 3) = [ 454, 456, 452, 458, 412, 414, 478, 474 ]
f(5, 2) = [ 54, 56, 52, 58 ]
f(0, 5) = [ 0878, 0874, 0898, 0896, 0854, 0856, 0852, 0858, 0808 ]
"""
