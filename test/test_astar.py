import heapq
import math
import unittest
from typing import List, Any

from bayrob.core.astar import AStar, Node, BiDirAStar


class GridWorld:
    GRID = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],  # 0 are free path whereas 1's are obstacles
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]

    ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # up, left, right, down
    OBSTACLE = 1
    FREE = 0
    STEP = 8

    # symbol representations for pretty printing
    REP = {
        FREE: '\u2B1C',  # empty box
        OBSTACLE: '\u2B1B',  # filled box
        STEP: '\u22C6',  # star
        (-1, 0): '\u2191',  # arrow up
        (0, -1): '\u2190',  # arrow left
        (1, 0): '\u2193',  # arrow down
        (0, 1): '\u2192',  # arrow right
        None: '\u2666'  # diamond
    }

    def __init__(self):
        pass

    @staticmethod
    def strworld(
            grid,
            legend=True
    ):
        lgnd = f'\n\n{GridWorld.REP[GridWorld.FREE]} Free cell\n' \
               f'{GridWorld.REP[GridWorld.OBSTACLE]} Obstacle\n' \
               f'{GridWorld.REP[None]} Goal\n' \
               f'{" ".join([GridWorld.REP[x] for x in GridWorld.ACTIONS])} Action executed\n'
        if grid is None:
            return lgnd

        world = '\n' + '\n'.join(
            [' '.join([GridWorld.REP[grid[row][col]] for col in range(len(grid[row]))]) for row in range(len(grid))])
        return world + (lgnd if legend else '\n')


class State:
    def __init__(
            self,
            posx: float,  # column
            posy: float  # row
    ):
        self.posx = posx
        self.posy = posy

    def __eq__(self, other):
        return self.posx == other.posx and self.posy == other.posy

    def similarity(self,
                   other: 'State'
    ) -> float:
        return 1 if self == other else 0

    def __str__(self) -> str:
        return f'<State pos: ({str(self.posx)}/{str(self.posy)})>'

    def __repr__(self):
        return str(self)


class Goal:
    def __init__(
            self,
            posx: int,
            posy: int
    ):
        self.posx = posx
        self.posy = posy

    def __str__(self) -> str:
        return f'<State pos: ({str(self.posx)}/{str(self.posy)})>'

    def __repr__(self) -> str:
        return str(self)


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goalstate: State,  # might be belief state later
            state_similarity: float = 1,
            goal_confidence: float = 1,
    ):
        super().__init__(
            initstate,
            goalstate,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        init = Node(state=self.initstate, g=0., h=self.h(self.initstate), parent=None)
        heapq.heappush(self.open, (init.f, init))

    def generate_successors(
            self,
            node
    ) -> List[Node]:
        successors = []
        for action in GridWorld.ACTIONS:
            pos_x = node.state.posx + action[1]
            pos_y = node.state.posy + action[0]

            # check if agent stays within grid lines
            if not (0 <= pos_x <= len(GridWorld.GRID[0]) - 1 and 0 <= pos_y <= len(GridWorld.GRID) - 1):
                continue

            # check for collision
            if GridWorld.GRID[pos_y][pos_x] != GridWorld.FREE:
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

    def plot(
            self,
            path
    ):
        print(f'FOUND GOAL!')
        path = self.retrace_path(path)

        # generate mapping from path step (=position) to action executed from this position
        actions = {k: v for k, v in zip(path, [(b[0] - a[0], b[1] - a[1]) for a, b in list(zip(path, path[1:]))])}

        # draw path steps into grid (use action symbols)
        res = [[GridWorld.GRID[y][x] if (y, x) not in path else actions.get((y, x), None) for x in
                range(len(GridWorld.GRID))] for y in range(len(GridWorld.GRID[0]))]
        print(GridWorld.strworld(res, legend=False))

    def isgoal(
            self,
            node
    ) -> bool:
        return node.state.posx == self.goal.posx and node.state.posy == self.goal.posy

    def h(
            self,
            state
    ) -> float:
        # Euclidean distance
        dx = state.posx - self.goal.posx
        dy = state.posy - self.goal.posy
        return math.sqrt(dx ** 2 + dy ** 2)

    def retrace_path(
            self,
            node
    ) -> Any:
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.state.posy, current_node.state.posx))
            current_node = current_node.parent
        path.reverse()
        return path


class SubAStar_BW(SubAStar):

    def __init__(
            self,
            initstate: State,
            goalstate: State,  # might be belief state later
            state_similarity: float = 1,
            goal_confidence: float = 1,
    ):
        super().__init__(
            goalstate,
            initstate,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )


class AStarAlgorithmTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.init = State(
            posx=0,
            posy=0
        )

        cls.goal = State(
            posx=len(GridWorld.GRID[0]) - 1,
            posy=len(GridWorld.GRID) - 1,
        )

        print(GridWorld.strworld(GridWorld.GRID, legend=False))

    def test_astar_path(self) -> None:
        a_star = SubAStar(
            AStarAlgorithmTests.init,
            AStarAlgorithmTests.goal
        )
        self.path = a_star.search()

        self.assertTrue(self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)] or
                        self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)], msg='A* path incorrect')

    def test_bw_astar_path(self) -> None:
        a_star = SubAStar_BW(
            AStarAlgorithmTests.init,
            AStarAlgorithmTests.goal
        )
        self.path = a_star.search()

        self.assertTrue(self.path == [(6, 6), (5, 6), (4, 6), (4, 5), (3, 5), (2, 5), (1, 5), (1, 4), (1, 3), (1, 2), (0, 2), (0, 1), (0, 0)], msg='A* path incorrect')

    def test_bdir_astar_path(self) -> None:
        bidir_astar = BiDirAStar(SubAStar, SubAStar_BW, AStarAlgorithmTests.init, AStarAlgorithmTests.goal)
        self.path = bidir_astar.search()

        self.assertTrue(self.path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)] or
                        self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6)],
                        msg='Bi-directional A* path incorrect')

    def tearDown(self) -> None:
        print(f'Plotting result for {self.__str__()}')

        # generate mapping from path step (=position) to action executed from this position
        actions = {k: v for k, v in zip(self.path, [(b[0] - a[0], b[1] - a[1]) for a, b in list(zip(self.path, self.path[1:]))])}

        # draw path steps into grid (use action symbols)
        res = [[GridWorld.GRID[y][x] if (y, x) not in self.path else actions.get((y, x), None) for x in range(len(GridWorld.GRID))] for y in range(len(GridWorld.GRID[0]))]
        print(GridWorld.strworld(res, legend=False))
        print('=======================================================================================================')

    @classmethod
    def tearDownClass(cls) -> None:
        print(GridWorld.strworld(None))
