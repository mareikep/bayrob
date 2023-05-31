import math
import os
from pathlib import Path
from typing import List, Dict, Union

from jpt.base.intervals import ContinuousSet

from calo.core.astar import AStar, Node
from calo.utils import locs
from calo.utils.utils import angledeg, pnt2line
from jpt.trees import Leaf, JPT


class State:
    def __init__(
            self,
            posx: Union[float, ContinuousSet],
            posy: Union[float, ContinuousSet],
            dirx: Union[float, ContinuousSet] = None,
            diry: Union[float, ContinuousSet] = None
    ):
        self.posx = posx
        self.posy = posy
        self.dirx = dirx
        self.diry = diry


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goalstate: State,  # might be belief state later
            models: Dict
    ):
        self.models = models
        super().__init__(initstate, goalstate)

    def stepcost(self, state) -> float:
        # distance (Euclidean) travelled so far (from init_pos to current position)
        cost = 0.
        dx = self.initstate.posx - state.posx
        dy = self.initstate.posy - state.posy
        cost += math.sqrt(dx ** 2 + dy ** 2)

        # difference in orientation (from init dir to current dir)
        cost += angledeg([state.dirx, state.diry], [self.initstate.dirx, self.initstate.diry])

        return cost

    def h(self, state) -> float:
        # Euclidean distance from current position to goal node
        cost = 0.
        if isinstance(self.goalstate.posx, ContinuousSet) and isinstance(self.goalstate.posy, ContinuousSet):
            # assuming the goal area is a rectangle, calculate the minimum distance between the current position (= point)
            # to the nearest edge of the rectangle
            cost += min([d for d, _ in [
                pnt2line([state.posx, state.posy], [self.goalstate.posx.lower, self.goalstate.posy.lower], [self.goalstate.posx.lower, self.goalstate.posy.upper]),
                pnt2line([state.posx, state.posy], [self.goalstate.posx.lower, self.goalstate.posy.lower], [self.goalstate.posx.upper, self.goalstate.posy.lower]),
                pnt2line([state.posx, state.posy], [self.goalstate.posx.lower, self.goalstate.posy.upper], [self.goalstate.posx.upper, self.goalstate.posy.upper]),
                pnt2line([state.posx, state.posy], [self.goalstate.posx.upper, self.goalstate.posy.lower], [self.goalstate.posx.upper, self.goalstate.posy.upper])
            ]])
        else:
            # current position and goal position are points
            dx = state.posx - self.goalstate.posx
            dy = state.posy - self.goalstate.posy
            cost += math.sqrt(dx ** 2 + dy ** 2)

        # if no directions are given, return costs at this point
        if any([x is None for x in [state.dirx, state.diry]]):
            return cost  # TODO

        # difference in orientation (current dir to dir to goal node)
        # vec to goal node:
        dx = self.goalstate.posx - state.posx
        dy = self.goalstate.posy - state.posy
        cost += angledeg([state.dirx, state.diry], [dx, dy])

        return cost

    def generate_steps(self, node) -> List[Leaf]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param node: the current node
        :type node: SubNode
        """
        evidence = {
            'x_in': node.state.posx,
            'y_in': node.state.posy,
            'xdir_in': node.state.dirx,
            'ydir_in': node.state.diry}

        condtrees = [[tn, tree.conditional_jpt(evidence=tree.bind({k: v for k, v in evidence.items() if k in tree.varnames}))] for tn, tree in self.models.items()]

        return [(leaf, treename) for treename, tree in condtrees for _, leaf in tree.leaves.items()]

    def generate_successors(self, node) -> List[Node]:
        successors = []
        for succ, tn in self.generate_steps(node):

            # update position
            pos_x = node.state.posx
            pos_y = node.state.posy
            if 'x_out' in succ.value and 'y_out' in succ.value:
                pos_x = succ.value['x_out'].expectation()
                pos_y = succ.value['y_out'].expectation()

            # update direction
            dir_x = node.state.dirx
            dir_y = node.state.diry
            if 'xdir_out' in succ.value and 'ydir_out' in succ.value:
                dir_x = succ.value['xdir_out'].expectation()
                dir_y = succ.value['ydir_out'].expectation()

            state = State(
                posx=pos_x,
                posy=pos_y,
                dirx=dir_x,
                diry=dir_y
            )

            # TODO: check model collision? --> succ.value['collision'].expectation()

            successors.append(
                Node(
                    state=state,
                    g=node.g + self.stepcost(state),
                    h=self.h(state),
                    parent=node,
                    tree=tn,  # TODO: remove! --> only for debugging
                    leaf=succ  # TODO: remove! --> only for debugging
                )
            )
        return successors

    def isgoal(self, node) -> bool:
        # true if node pos and target pos are equal
        if isinstance(self.goalstate.posx, ContinuousSet) and isinstance(self.goalstate.posy, ContinuousSet):
            return self.goalstate.posx.contains_value(node.state.posx) and self.goalstate.posy.contains_value(node.state.posy)
        else:
            return node.state.posx == self.goalstate.posx and node.state.posy == self.goalstate.posy


if __name__ == "__main__":
    models = dict(
        [
            (
                treefile.name,
                JPT.load(str(treefile))
            )
            for p in [os.path.join(locs.examples, 'robotaction', '2023-05-16_14:02')]
            for treefile in Path(p).rglob('*.tree')
        ]
    )

    initstate = State(
        posx=0,
        posy=0,
        dirx=1,
        diry=0
    )

    goalstate = State(
        posx=5,
        posy=3,
        dirx=None,
        diry=None
    )

    a_star = SubAStar(
        initstate=initstate,
        goalstate=goalstate,
        models=models
    )

    path = a_star.search()
    print(path)
