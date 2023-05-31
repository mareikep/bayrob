import math
import unittest

from calo.core.astar import AStar, Node, BiDirAStar
from typing import List, Any

from calo.utils.utils import angledeg


class State:
    def __init__(
            self,
            posx: float,
            posy: float
    ):
        self.posx = posx
        self.posy = posy


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

    ACTIONS = [[-1, 0], [0, -1], [1, 0], [0, 1]]  # up, left, down, right

    def __init__(
            self,
            initstate: State,
            goalstate: State,  # might be belief state later
    ):
        super().__init__(initstate, goalstate)

    def generate_successors(self, node) -> List[Node]:
        successors = []
        for action in SubAStar.ACTIONS:
            pos_x = node.state.posx + action[1]
            pos_y = node.state.posy + action[0]

            # check if agent stays within grid lines
            if not (0 <= pos_x <= len(SubAStar.GRID[0]) - 1 and 0 <= pos_y <= len(SubAStar.GRID) - 1):
                continue

            # check for collision
            if SubAStar.GRID[pos_y][pos_x] != 0:
                continue

            state = State(
                posx=pos_x,
                posy=pos_y
            )

            successors.append(
                Node(
                    state=state,
                    g=node.g + 1,
                    h=self.h(state),
                    parent=node,
                )
            )

        return successors

    def isgoal(self, node) -> bool:
        return node.state.posx == self.goalstate.posx and node.state.posy == self.goalstate.posy

    def h(self, state) -> float:
        # dy = state.posx - self.goalnode_x
        # dx = state.posy - self.goalnode_y
        # return math.sqrt(dy ** 2 + dx ** 2)

        dx = state.posx - self.goalstate.posx
        dy = state.posy - self.goalstate.posy
        return math.sqrt(dx ** 2 + dy ** 2)

    def retrace_path(self, node) -> Any:
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.state.posy, current_node.state.posx))
            current_node = current_node.parent
        path.reverse()
        return path


class AStarAlgorithmTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.init = State(
            posx=0,
            posy=0
        )

        cls.goal = State(
            posx=len(SubAStar.GRID) - 1,
            posy=len(SubAStar.GRID[0]) - 1
        )

        for elem in SubAStar.GRID:
            print(elem)

    def test_astar_path(self) -> None:
        a_star = SubAStar(AStarAlgorithmTests.init, AStarAlgorithmTests.goal)
        self.path = a_star.search()

        self.assertEqual(self.path, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)], msg='A* path incorrect')

    def test_bdir_astar_path(self) -> None:
        bidir_astar = BiDirAStar(SubAStar, SubAStar, AStarAlgorithmTests.init, AStarAlgorithmTests.goal)
        self.path = bidir_astar.search()

        self.assertTrue(self.path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)] or
                        self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)],
                        msg='Bi-directional A* path incorrect')

    def tearDown(self) -> None:
        res = [[SubAStar.GRID[y][x] if (y, x) not in self.path else 8 for x in range(len(SubAStar.GRID))] for y in range(len(SubAStar.GRID[0]))]
        print()
        for elem in res:
            print(elem)
