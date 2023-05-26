import heapq
from typing import List, Any

import dnutils

from calo.utils.constants import calologger

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Node:
    """Abstract Node class for abstract A* Algorithm"""

    def __init__(self):
        self.g_ = self.g()
        self.h_ = self.h()

    def h(self) -> float:
        raise NotImplementedError

    def g(self) -> float:
        raise NotImplementedError

    @property
    def f(self) -> float:
        return self.g() + self.h()

    def __lt__(self, other) -> bool:
        raise NotImplementedError


class AStar:
    """Abstract A* class. Inheriting classes need to implement functions for
    goal check, path retraction and successor generation."""
    def __init__(self, start, goal, **kwargs):
        """

        """
        self.startnode = start
        self.goalnode = goal
        self.__dict__.update(kwargs)

        self.open = []
        heapq.heappush(self.open, (self.startnode.f, self.startnode))
        self.closed = []

        self.reached = False

    def generate_successors(self, node) -> List[Node]:
        raise NotImplementedError

    def isgoal(self, node) -> bool:
        """Check if current node is goal node"""
        raise NotImplementedError

    def retrace_path(self, node) -> Any:
        """Path from init_pos to goal"""
        raise NotImplementedError

    def search(self) -> None:
        while self.open:
            cf, cur_node = heapq.heappop(self.open)

            # OLD----------------------
            if self.isgoal(cur_node):
                self.reached = True
                return self.retrace_path(cur_node)
            # NEW----------------------
            # if valid hypothesis is found (hypothesis reaches goal AND precondition is init state), stop searching.
            # if self.isgoal(cur_node, onlygoal=False):
            #     logger.warning('FOUND VALID HYPOTHESIS!', cur_node)
            #     self.reached = True
            #     return self.retrace_path(cur_node)
            #
            # # if hypothesis does not match goal at all, drop it. (FIXME: should not happen, as generate_successors shouldn't have selected it in the first place
            # if not self.isgoal(cur_node, onlygoal=True) and cur_node.identifiers:
            #     logger.warning('Hypothesis candidate', cur_node.id, 'does not match goal. Drop it!')
            #     heapq.heappush(self.closed, (cf, cur_node))
            #     continue
            # # else: current hypothesis promising (goal is met but no complete path yet), find new nodes to prepend
            # logger.info('Hypothesis candidate', cur_node.id, 'matches goal. Expanding!')
            # /NEW---------------------

            heapq.heappush(self.closed, (cf, cur_node))
            successors = self.generate_successors(cur_node)

            for c in successors:
                if c in self.closed:
                    continue

                if c not in self.open:
                    heapq.heappush(self.open, (c.f, c))
                else:
                    cf_, c_ = heapq.heappop(self.open)

                    if c.g < c_.g:
                        heapq.heappush(self.open, (c.f, c))
                    else:
                        heapq.heappush(self.open, (cf_, c_))

        if not self.reached:
            return self.startnode


class BiDirAStar:

    def __init__(self, fastar, bastar, start, goal, **kwargs):
        self.f_astar = fastar(start, goal, **kwargs)
        self.b_astar = bastar(goal, start, **kwargs)
        self.reached = False

    def retrace_path(self, fnode, bnode) -> Any:
        fpath = self.f_astar.retrace_path(fnode)
        bpath = self.b_astar.retrace_path(bnode)
        bpath.reverse()

        path = fpath
        path.extend([p for p in bpath if p not in fpath])
        return path

    def common_node(self, fnode, bnode) -> bool:
        return bnode.pos == fnode.pos or bnode.pos == bnode.goalnode or fnode.pos == fnode.goalnode

    def search(self) -> None:
        while self.f_astar.open or self.b_astar.open:
            _, cur_fnode = heapq.heappop(self.f_astar.open)
            _, cur_bnode = heapq.heappop(self.b_astar.open)

            # if both paths have common node
            if self.common_node(cur_fnode, cur_bnode):
                self.reached = True
                return self.retrace_path(cur_fnode, cur_bnode)

            heapq.heappush(self.f_astar.closed, (cur_fnode.f, cur_fnode))
            heapq.heappush(self.b_astar.closed, (cur_bnode.f, cur_bnode))

            self.f_astar.goalnode = cur_bnode
            self.b_astar.goalnode = cur_fnode

            successors = {
                self.f_astar: self.f_astar.generate_successors(cur_fnode),
                self.b_astar: self.b_astar.generate_successors(cur_bnode),
            }

            for astar in [self.f_astar, self.b_astar]:
                for c in successors[astar]:
                    if c in astar.closed:
                        continue

                    if c not in astar.open:
                        heapq.heappush(astar.open, (c.f, c))
                    else:
                        # retrieve the best current path
                        _, c_ = heapq.heappop(astar.open)

                        if c.g < c_.g:
                            heapq.heappush(astar.open, (c.f, c))
                        else:
                            heapq.heappush(astar.open, (c_.f, c_))

        if not self.reached:
            return self.f_astar.init_pos
