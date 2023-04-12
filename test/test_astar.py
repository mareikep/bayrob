import math
import unittest

from calo.core.astar import AStar, Node, BiDirAStar


class SubNode(Node):

    def __init__(self, pos_x, pos_y, goal_x, goal_y, g, parent):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = (pos_y, pos_x)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal = (goal_y, goal_x)
        self.parent = parent
        self._g = g
        super().__init__()

    def g(self) -> float:
        return self._g

    def h(self) -> float:
        dy = self.pos_x - self.goal_x
        dx = self.pos_y - self.goal_y
        return math.sqrt(dy ** 2 + dx ** 2)

    def __lt__(self, other) -> bool:
        return self.f < other.f


class SubAStar(AStar):
    GRID = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],  # 0 are free path whereas 1's are obstacles
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]

    DELTA = [[-1, 0], [0, -1], [1, 0], [0, 1]]  # up, left, down, right

    def __init__(self, start, goal):
        self.start = SubNode(start[1], start[0], goal[1], goal[0], 0, None)
        self.goal = SubNode(goal[1], goal[0], goal[1], goal[0], 99999, None)
        super().__init__(self.start, self.goal)

    def generate_successors(self, node):
        successors = []
        for action in SubAStar.DELTA:
            pos_x = node.pos_x + action[1]
            pos_y = node.pos_y + action[0]
            # check if agent stays within grid lines
            if not (0 <= pos_x <= len(SubAStar.GRID[0]) - 1 and 0 <= pos_y <= len(SubAStar.GRID) - 1):
                continue

            # check for collision
            if SubAStar.GRID[pos_y][pos_x] != 0:
                continue

            successors.append(
                SubNode(
                    pos_x,
                    pos_y,
                    self.target.pos_x,
                    self.target.pos_y,
                    node.g() + 1,
                    node,
                )
            )
        return successors

    def isgoal(self, node, onlygoal=True):
        return node.pos == self.target.pos

    def retrace_path(self, node):
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.pos_y, current_node.pos_x))
            current_node = current_node.parent
        path.reverse()
        return path


class AStarAlgorithmTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.init = (0, 0)
        cls.goal = (len(SubAStar.GRID) - 1, len(SubAStar.GRID[0]) - 1)
        for elem in SubAStar.GRID:
            print(elem)

    def test_astar_path(self) -> None:
        a_star = SubAStar(AStarAlgorithmTests.init, AStarAlgorithmTests.goal)
        self.path = a_star.search()

        self.assertEqual(self.path, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)], msg='A* path incorrect')

    def test_bdir_astar_path(self) -> None:
        bidir_astar = BiDirAStar(SubAStar, AStarAlgorithmTests.init, AStarAlgorithmTests.goal)
        self.path = bidir_astar.search()

        self.assertEqual(self.path, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)], msg='Bi-directional A* path incorrect')

    def tearDown(self) -> None:
        res = [[SubAStar.GRID[y][x] if (y, x) not in self.path else 8 for x in range(len(SubAStar.GRID))] for y in range(len(SubAStar.GRID[0]))]
        print()
        for elem in res:
            print(elem)
