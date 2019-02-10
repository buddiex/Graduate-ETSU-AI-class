from math import sqrt
from operator import itemgetter

from graphics import *

# Use the ETSU-Official Colors - GO BUCS!
etsu_blue = color_rgb(4, 30, 68)
etsu_gold = color_rgb(255, 199, 44)
etsu_bg = color_rgb(223, 209, 167)


class PriorityQueue:
    """
        Simple Priority Queue Class using a Python list
    """

    def __init__(self):
        self.queue = []

    def put(self, item, priority):
        """
            Add the item and sort by priority
        """
        node = [item, priority]
        self.queue.append(node)
        self.queue.sort(key=itemgetter(1))

    def get(self):
        """
            Return the highest-priority item in the queue
        """
        if len(self.queue) == 0:
            return None
        node = self.queue.pop(0)
        return node[0]

    def empty(self):
        """
            Return True if the queue has no items
        """
        return len(self.queue) == 0


class Field:
    """
        Class Field uses the graphics.py library

        It simulates an x by y field that contains Polygons, Lines, and Points.

        Search Space: Vertexes of each polygon, Start & End locations
    """

    def __init__(self, width, height, intitle):
        """
            Create lists of points, polygons
        """
        self.points = []
        self.path = []
        self.polygons = []
        self.extras = []
        self.width = width
        self.height = height
        self.start = Point(0, 0)
        self.end = Point(0, 0)
        self.win = GraphWin(title=intitle, width=width, height=height)

    def set_coords(self, x1, y1, x2, y2):
        """
            Set the viewport of the Field
        """
        self.win.setCoords(x1, y1, x2, y2)

    def set_background(self, color):
        """
            Set the background color
        """
        self.win.setBackground(color)

    def add_polygon(self, polygon):
        """
            Add the polygon and the vertexes of the Polygon
        """
        self.points = self.points + polygon.getPoints()
        self.polygons.append(polygon)
        polygon.draw(self.win)

    def add_start(self, start):
        """
            Add and display the starting location
        """
        self.points.append(start)
        self.start = start
        c = Circle(start, 2)
        c.setFill('green')
        self.extras.append(c)
        text = Text(Point(start.x - 8, start.y + 10), 'Start')
        text.setSize(10)
        text.setTextColor('black')
        self.extras.append(text)
        text.draw(self.win)
        c.draw(self.win)

    def add_end(self, end):
        """
            Add and display the ending location
        """
        self.points.append(end)
        self.end = end
        c = Circle(end, 2)
        c.setFill('red')
        self.extras.append(c)
        text = Text(Point(end.x - 2, end.y - 10), 'End')
        text.setSize(10)
        text.setTextColor('black')
        self.extras.append(text)
        text.draw(self.win)
        c.draw(self.win)

    def get_neighbors(self, node):
        """
          Returns a list of neighbors of node -- Vertexes that the node can see.
          All vertexes are within node's line-of-sight.
        """
        neighbors = []

        # Loop through vertexes
        for point in self.points:
            # Ignore the vertex if it is the same as the node passed
            if point == node:
                continue

            intersects = False

            # Create a line that represents a potential path segment
            path_segment = Line(node, point)

            # Loop through the Polygons in the Field
            for o in self.polygons:
                # If the path segment intersects the Polygon, ignore it.
                if o.intersects(path_segment):
                    intersects = True
                    break

            # If the path segment does not intersect the Polygon, it is a
            #  valid neighbor.
            if not intersects:
                neighbors.append(point)

        return neighbors

    def wait(self):
        """
            Pause the Window for action
        """
        self.win.getMouse()

    def close(self):
        """
            Closes the Window after a pause
        """
        self.win.getMouse()
        self.win.close()

    def reset(self):
        for extra in self.extras:
            extra.undraw()
        self.extras = []

    def backtrack(self, came_from, node):
        """
            Recreate the path located.

            Requires a came_from dictionary that contains the parents of each node.
            The node passed is the end of the path.
        """
        current = node
        self.path.append(current)
        parent = came_from[str(current)]
        while parent != self.start:
            line = Line(current, parent)
            line.setOutline("green")
            line.setArrow("first")
            self.extras.append(line)
            line.draw(self.win)
            current = parent
            parent = came_from[str(current)]
            self.path.append(current)
        line = Line(current, parent)
        line.setOutline("green")
        line.setArrow("first")
        line.draw(self.win)
        self.extras.append(line)
        self.path.append(parent)
        self.path.reverse()

    @staticmethod
    def straight_line_distance(point1, point2):
        """
            Returns the straight-line distance between point 1 and point 2
        """
        return sqrt((point1.x - point2.x) ** 2 + (point1.x - point2.x) ** 2)

    def astar_search(self):
        """
        Create the A* Search Here Use the Backtrack method to draw the final path when your
             algorithm locates the end point
        """
        node = self.start

        frontier = PriorityQueue()
        frontier.put(node, 0)
        came_from = {}
        cost_so_far = {}

        came_from[str(node)] = None
        cost_so_far[str(node)] = 0

        while frontier:
            current_item = frontier.get()
            if current_item == self.end:
                break
            for neigbor in self.get_neighbors(current_item):
                str_neigbor = str(neigbor)
                path_cost = cost_so_far[str(current_item)] + 1
                if str_neigbor not in cost_so_far or path_cost < cost_so_far[str_neigbor]:
                    cost_so_far[str_neigbor] = path_cost
                    frontier.put(neigbor, path_cost + self.straight_line_distance(neigbor, self.end))
                    came_from[str_neigbor] = current_item
        self.backtrack(came_from, current_item)

        return cost_so_far[str(current_item)]


def main():
    f = Field(700, 400, "Search Space")
    f.set_coords(90, 500, 400, 700)
    f.set_background(etsu_bg)

    # Setup Polygons
    p1 = Polygon(Point(240, 616), Point(220, 666), Point(251, 670), Point(272, 647))
    p1.setFill(etsu_blue)
    p2 = Polygon(Point(341, 655), Point(359, 667), Point(374, 651), Point(366, 577))
    p2.setFill(etsu_gold)
    p3 = Polygon(Point(311, 530), Point(311, 559), Point(339, 578), Point(361, 560), Point(361, 528), Point(336, 516))
    p3.setFill(etsu_blue)
    p4 = Polygon(Point(105, 628), Point(151, 670), Point(180, 629), Point(156, 577), Point(113, 587))
    p4.setFill(etsu_gold)
    p5 = Polygon(Point(118, 517), Point(245, 517), Point(245, 557), Point(118, 557))
    p5.setFill(etsu_blue)
    p6 = Polygon(Point(300, 583), Point(333, 583), Point(333, 665), Point(280, 665))
    p6.setFill(etsu_gold)
    p7 = Polygon(Point(252, 594), Point(290, 562), Point(264, 538))
    p7.setFill(etsu_blue)
    p8 = Polygon(Point(198, 635), Point(217, 574), Point(182, 574))
    p8.setFill(etsu_gold)
    p9 = Polygon(Point(190, 675), Point(210, 675), Point(210, 650), Point(190, 645))
    p9.setFill(etsu_blue)
    p10 = Polygon(Point(280, 540), Point(305, 550), Point(300, 505), Point(280, 510))
    p10.setFill(etsu_gold)
    p11 = Polygon(Point(230, 600), Point(250, 620), Point(240, 580))
    p11.setFill(etsu_blue)
    p12 = Polygon(Point(270, 680), Point(360, 695), Point(340, 675), Point(260, 666))
    p12.setFill(etsu_gold)
    p13 = Polygon(Point(263, 600), Point(263, 630), Point(272, 630), Point(272, 600))
    p13.setFill(etsu_blue)
    p14 = Polygon(Point(130, 672), Point(150, 692), Point(210, 685))
    p14.setFill(etsu_gold)
    p15 = Polygon(Point(366, 516), Point(380, 516), Point(380, 570))
    p15.setFill(etsu_gold)
    p16 = Polygon(Point(120, 560), Point(140, 560), Point(140, 575), Point(120, 575))
    p16.setFill(etsu_gold)
    p17 = Polygon(Point(220, 675), Point(240, 690), Point(220, 690))
    p17.setFill(etsu_gold)

    f.add_polygon(p1)
    f.add_polygon(p2)
    f.add_polygon(p3)
    f.add_polygon(p4)
    f.add_polygon(p5)
    f.add_polygon(p6)
    f.add_polygon(p7)
    f.add_polygon(p8)
    f.add_polygon(p9)
    f.add_polygon(p10)
    f.add_polygon(p11)
    f.add_polygon(p12)
    f.add_polygon(p13)
    f.add_polygon(p14)
    f.add_polygon(p15)
    f.add_polygon(p16)
    f.add_polygon(p17)

    """
        Try BOTH SETS of Start and End Points
    """
    f.add_start(Point(105, 670))
    f.add_end(Point(390, 550))

    path_cost = f.astar_search()

    print("Straight Line Distance from Start to Goal: %f" % f.straight_line_distance(f.start, f.end))
    print("Path Cost from Start to Goal: %f" % path_cost)
    print("Path----------")
    print(f.path)

    f.wait()  # Click to continue
    f.reset()  # Reset the Field

    f.add_start(Point(110, 580))
    f.add_end(Point(390, 640))

    path_cost = f.astar_search()

    print("Straight Line Distance from Start to Goal: %f" % f.straight_line_distance(f.start, f.end))
    print("Path Cost from Start to Goal: %f" % path_cost)
    print("Path----------")
    print(f.path)

    f.wait()  # Click to continue
    f.reset()  # Reset the Field

    f.add_start(Point(180, 650))
    f.add_end(Point(350, 510))

    path_cost = f.astar_search()

    print("Straight Line Distance from Start to Goal: %f" % f.straight_line_distance(f.start, f.end))
    print("Path Cost from Start to Goal: %f" % path_cost)
    print("Path----------")
    print(f.path)

    f.close()


main()
