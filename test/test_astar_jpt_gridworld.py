import heapq
import math
import os
import unittest
from typing import List, Any, Union, Dict

from jpt.base.intervals import ContinuousSet

from calo.core.astar import AStar, Node
from calo.utils import locs
from dnutils import first
from jpt import JPT
from jpt.distributions import Integer


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
            posx: Integer,  # column
            posy: Integer,  # row
            dirx: Integer = None,
            diry: Integer = None,
            ctree: Any = None,
            leaf: Any = None,
    ):
        self.posx = posx
        self.posy = posy
        self.dirx = dirx
        self.diry = diry
        self.ctree = ctree
        self.leaf = leaf

    def __eq__(self, other):
        return self.posx == other.posx and self.posy == other.posy

    def similarity(self,
                   other: 'State'
    ) -> float:
        return min(
            [
                Integer.jaccard_similarity(self.posx, other.posx),
                Integer.jaccard_similarity(self.posy, other.posy)
            ]
        )

    def __str__(self) -> str:
        return f'<State pos: ({str(first(self.posx.mpe()[1]))}/{str(first(self.posy.mpe()[1]))})>'

    def __repr__(self):
        return str(self)


class Goal:
    def __init__(
            self,
            posx: Union[float, ContinuousSet],
            posy: Union[float, ContinuousSet]
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
            goalstate: Goal,  # might be belief state later
            models: Dict
    ):
        self.models = models
        super().__init__(initstate, goalstate)

    def init(self):
        init = Node(state=self.initstate, g=0., h=self.h(self.initstate), parent=None)
        heapq.heappush(self.open, (init.f, init))

    def generate_steps(
            self,
            node
    ) -> List[Any]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param node: the current node
        :type node: SubNode
        """
        evidence = {
            'x_in': node.state.posx.mpe()[1],
            'y_in': node.state.posy.mpe()[1],
            'xdir_in': node.state.dirx.mpe()[1],
            'ydir_in': node.state.diry.mpe()[1]
        }

        condtrees = [
            [
                tn,
                tree.conditional_jpt(
                    evidence=tree.bind(
                        {k: v for k, v in evidence.items() if k in tree.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False)
            ] for tn, tree in self.models.items()
        ]
        return [(leaf, treename, tree) for treename, tree in condtrees if tree is not None for _, leaf in tree.leaves.items()]

    def generate_successors(
            self,
            node
    ) -> List[Node]:
        successors = []
        for succ, tn, t in self.generate_steps(node):

            # get distributions representing current belief state
            posx = node.state.posx
            posy = node.state.posy
            dirx = node.state.dirx
            diry = node.state.diry

            # generate new position distribution by shifting position delta distributions by expectation of position
            # belief state

            if 'x_out' in succ.value:
                posx = posx + succ.value['x_out']

            if 'y_out' in succ.value:
                posy = posy + succ.value['y_out']

            # generate new orientation distribution by shifting orientation delta distributions by expectation of
            # orientation belief state
            if 'xdir_out' in succ.value:
                dirx = dirx + succ.value['xdir_out']

            if 'ydir_out' in succ.value:
                diry = diry + succ.value['ydir_out']

            # initialize new belief state for potential successor
            state = State(
                posx=posx,
                posy=posy,
                ctree=t,
                leaf=succ,
                dirx=dirx,
                diry=diry
            )

            successors.append(
                Node(
                    state=state,
                    g=node.g + self.stepcost(state),
                    h=self.h(state),
                    parent=node
                )
            )
        return successors

    def isgoal(
            self,
            node
    ) -> bool:
        return node.state.posx == self.goalstate.posx and node.state.posy == self.goalstate.posy

    def stepcost(
            self,
            state
    ) -> float:
        # distance (Euclidean) travelled so far (from init_pos to current position)
        dx = first(self.initstate.posx.mpe()[1]) - first(state.posx.mpe()[1])
        dy = first(self.initstate.posy.mpe()[1]) - first(state.posy.mpe()[1])

        return math.sqrt(dx ** 2 + dy ** 2)

    def h(
            self,
            state: State
    ) -> float:
        p = 1
        if not any([x is None for x in [state.posx, state.posy]]):
            p *= state.posx.p(self.goalstate.posx) * state.posy.p(self.goalstate.posy)
        return p

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

    def plot(
            self,
            path
    ):
        # generate mapping from path step (=position) to action executed from this position
        actions = {k: v for k, v in
                   zip(path, [(first(b[0][1]) - first(a[0][1]), first(b[1][1]) - first(a[1][1])) for a, b in list(zip(path, path[1:]))])}

        # draw path steps into grid (use action symbols)
        res = [[GridWorld.GRID[y][x] if (y, x) not in path else actions.get((y, x), None) for x in
                range(len(GridWorld.GRID))] for y in range(len(GridWorld.GRID[0]))]
        print(GridWorld.strworld(res, legend=False))


class AStarGridworldJPTTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            'MOVE': JPT.load(os.path.join(locs.examples, 'gridagent', 'gridagent-MOVE.tree')),
            'TURN': JPT.load(os.path.join(locs.examples, 'gridagent', 'gridagent-TURN.tree')),
        }

        posx = {0}
        posy = {0}
        dirx = {1}
        diry = {0}

        posteriors = cls.models['MOVE'].posterior(
            evidence={
                'x_in': posx,
                'y_in': posy,
                'xdir_in': dirx,
                'ydir_in': diry
            }
        )

        cls.initstate = State(
            posx=posteriors['x_in'],
            posy=posteriors['y_in'],
            dirx=posteriors['xdir_in'],
            diry=posteriors['ydir_in']
        )

        cls.goalstate = Goal(
            posx={6},
            posy={6}
        )

        print(GridWorld.strworld(GridWorld.GRID, legend=False))

    def test_astar_path(self) -> None:
        self.a_star = SubAStar(
            AStarGridworldJPTTests.initstate,
            AStarGridworldJPTTests.goalstate,
            models=self.models
        )
        self.path = self.a_star.search()

        # self.assertTrue(
        #     self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5),
        #                   (5, 6), (6, 6)] or
        #     self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (5, 5),
        #                   (5, 6), (6, 6)], msg='A* path incorrect')

    def tearDown(self) -> None:
        # generate mapping from path step (=position) to action executed from this position
        actions = {k: v for k, v in
                   zip(self.path, [(b[0] - a[0], b[1] - a[1]) for a, b in list(zip(self.path, self.path[1:]))])}

        # draw path steps into grid (use action symbols)
        res = [[GridWorld.GRID[y][x] if (y, x) not in self.path else actions.get((y, x), None) for x in
                range(len(GridWorld.GRID))] for y in range(len(GridWorld.GRID[0]))]
        print(GridWorld.strworld(res, legend=False))

    @classmethod
    def tearDownClass(cls) -> None:
        print(GridWorld.strworld(None))
