import math
import os
import unittest

from pathlib import Path

from calo.core.astar import AStar, Node, BiDirAStar
from typing import List, Any

from calo.utils import locs
from calo.utils.utils import tovariablemapping, angledeg, pnt2line
from jpt.base.intervals import ContinuousSet

from jpt.trees import Leaf, JPT


class SubNode(Node):

    def __init__(self, init_pos_x, init_pos_y, pos_x, pos_y, goal_x, goal_y, parent, init_dir_x=None, init_dir_y=None, dir_x=None, dir_y=None, leaf=None, tree=None):
        self.init_pos_x = init_pos_x
        self.init_pos_y = init_pos_y
        self.init_pos = (init_pos_x, init_pos_y)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = (pos_x, pos_y)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal = (goal_x, goal_y)
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir = (dir_x, dir_y)
        self.init_dir_x = init_dir_x
        self.init_dir_y = init_dir_y
        self.init_dir = (init_dir_x, init_dir_y)
        self.parent = parent
        self.leaf = leaf
        self.tree = tree  # this is only for debugging (prettyprint)
        super().__init__()

    def g(self) -> float:
        # distance (Euclidean) travelled so far (from init_pos to current position)
        gcost = 0.
        dx = self.init_pos_x - self.pos_x
        dy = self.init_pos_y - self.pos_y
        gcost += math.sqrt(dx ** 2 + dy ** 2)

        # if no directions are given, return costs at this point
        if any([x is None for x in [self.dir_x, self.dir_y, self.init_dir_x, self.init_dir_y]]):
            return gcost

        # difference in orientation (from init_dir to current dir)
        gcost += angledeg([self.dir_x, self.dir_y], [self.init_dir_x, self.init_dir_y])

        return gcost

    def h(self) -> float:
        # Euclidean distance from current position to goal node
        hcost = 0.
        if isinstance(self.goal_x, ContinuousSet) and isinstance(self.goal_y, ContinuousSet):
            # assuming the goal area is a rectangle, calculate the minimum distance between the current position (= point)
            # to the nearest edge of the rectangle
            hcost += min([d for d, _ in [
                pnt2line([self.pos_x, self.pos_y], [self.goal_x.lower, self.goal_y.lower], [self.goal_x.lower, self.goal_y.upper]),
                pnt2line([self.pos_x, self.pos_y], [self.goal_x.lower, self.goal_y.lower], [self.goal_x.upper, self.goal_y.lower]),
                pnt2line([self.pos_x, self.pos_y], [self.goal_x.lower, self.goal_y.upper], [self.goal_x.upper, self.goal_y.upper]),
                pnt2line([self.pos_x, self.pos_y], [self.goal_x.upper, self.goal_y.lower], [self.goal_x.upper, self.goal_y.upper])
            ]])
        else:
            # current position and goal position are points
            dx = self.pos_x - self.goal_x
            dy = self.pos_y - self.goal_y
            hcost += math.sqrt(dx ** 2 + dy ** 2)

        # if no directions are given, return costs at this point
        if any([x is None for x in [self.dir_x, self.dir_y]]):
            return hcost

        # difference in orientation (current dir to dir to goal node)
        # vec to goal node:
        dx = self.goal_x - self.pos_x
        dy = self.goal_y - self.pos_y
        hcost += angledeg([self.dir_x, self.dir_y], [dx, dy])

        return hcost

    def __lt__(self, other) -> bool:
        return self.f < other.f

    def __str__(self):  # only for debugging
        current_node = self
        path = ""
        while current_node is not None:
            if current_node.parent is not None:
                path += f"-{current_node.tree}({current_node.leaf.idx})"
            current_node = current_node.parent
        return "H" + path


class SubAStar(AStar):

    def __init__(self, start, goal, dir, models):
        startnode = SubNode(start[0], start[1], start[0], start[1], goal[0], goal[1], None, init_dir_x=dir[0], init_dir_y=dir[1], dir_x=dir[0], dir_y=dir[1], leaf=None, tree=None)
        goalnode = SubNode(start[0], start[1], goal[0], goal[1], goal[0], goal[1], None)
        self.models = models
        super().__init__(startnode, goalnode)

    def generate_candidatesteps(self, node) -> List[Leaf]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param query: a variable-interval mapping
        :type query: jpt.variables.VariableMap or None
        """
        evidence = tovariablemapping({'x_in': node.pos_x, 'y_in': node.pos_y, 'xdir_in': node.dir_x, 'ydir_in': node.dir_y, }, self.models)
        condtrees = [[tn, tree.conditional_jpt(evidence=evidence)] for tn, tree in self.models.items()]
        return [(leaf, treename) for treename, tree in condtrees for _, leaf in tree.leaves.items()]

    def generate_successors(self, node) -> List[Node]:
        successors = []
        for succ, tn in self.generate_candidatesteps(node):

            # update position
            pos_x = node.pos_x
            pos_y = node.pos_y
            if 'x_out' in succ.value and 'y_out' in succ.value:
                pos_x = node.pos_x + succ.value['x_out'].expectation()
                pos_y = node.pos_y + succ.value['y_out'].expectation()

            # update direction
            dir_x = node.dir_x
            dir_y = node.dir_y
            if 'xdir_out' in succ.value and 'ydir_out' in succ.value:
                dir_x = succ.value['xdir_out'].expectation()
                dir_y = succ.value['ydir_out'].expectation()

            # check if agent stays within grid lines
            # if not (0 <= pos_x <= len(SubAStar.GRID[0]) - 1 and 0 <= pos_y <= len(SubAStar.GRID) - 1):
            #     continue

            # check for collision TODO: model obstacles manually? --> or check for succ.value['collision'].expectation()
            # if SubAStar.GRID[pos_y][pos_x] != 0:
            #     continue

            successors.append(
                SubNode(
                    self.startnode.init_pos_x,
                    self.startnode.init_pos_y,
                    pos_x,
                    pos_y,
                    self.goalnode.pos_x,
                    self.goalnode.pos_y,
                    node,
                    init_dir_x=self.startnode.init_dir_x,
                    init_dir_y=self.startnode.init_dir_y,
                    dir_x=dir_x,
                    dir_y=dir_y,
                    leaf=succ,
                    tree=tn
                )
            )
        return successors

    def isgoal(self, node) -> bool:
        # true if node pos and target pos are equal
        if isinstance(self.goalnode.goal_x, ContinuousSet) and isinstance(self.goalnode.goal_y, ContinuousSet):
            return self.goalnode.goal_x.contains_value(node.pos_x) and self.goalnode.goal_y.contains_value(node.pos_y)
        else:
            return node.pos == self.goalnode.goal

    def retrace_path(self, node) -> Any:
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.pos_y, current_node.pos_x))
            current_node = current_node.parent
        path.reverse()
        return path


if __name__ == "__main__":
    init = (0, 0)
    initdir = (0, 1)
    goal = (0, 1)
    models = dict([(treefile.name, JPT.load(str(treefile))) for p in
                       [os.path.join(locs.examples, 'robotaction', '2023-05-16_14:02')] for treefile in
                       Path(p).rglob('*.tree')])

    a_star = SubAStar(init, goal, initdir, models)
    path = a_star.search()
    print(path)
