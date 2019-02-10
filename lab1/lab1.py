import itertools

conf_room = [
    [8, 9, 13, 15],
    [9, 11, 16],
    [8, 9, 10, 12],
    [10, 12, 13, 15, 16],
    [8, 9, 10, 11, 12, 13, 14, 15, 16]
]

yourself = [
    [8, 12, 13, 15, 16],
    [8, 9, 11, 13, 14],
    [8, 12, 13, 15, 16],
    [8, 9, 11, 13, 14],
    [8, 9, 11, 13, 14, 15]
]

anna = [
    [9, 10, 11, 13, 15],
    [9, 11, 16],
    [9, 10, 11],
    [9, 10, 13],
    [8, 9, 11, 12, 15]
]

bob = [
    [10, 11, 13, 15],
    [8, 9, 10, 11],
    [10, 11, 13, 15],
    [8, 9, 13, 15],
    [8, 9, 11, 13]
]

carrie = [
    [8, 9, 10, 13, 14, 15],
    [11, 12, 13, 14, 15],
    [8, 9, 10, 13, 14],
    [12, 13, 14, 15],
    [8, 9, 10, 11, 13]
]


def goal(*args):
    return len(set(args)) == 1


def generateAndTest2(*args, goal):
    return [i[0] for i in itertools.product(*args) if goal(*i)]


schedule = [conf_room, yourself, anna, bob, carrie]
days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY']

# pass each column to the generatetest funtion
for n, l in enumerate(zip(*schedule)):
    print(days[n] + ':')
    for i in generateAndTest2(*l, goal=goal):
        print('- POTENTIAL MEETING at:', i)
