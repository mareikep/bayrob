import heapq
from typing import List, Any

import dnutils

from calo.utils.constants import calologger

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Node:
    """Abstract Node class for abstract A* Algorithm
    """

    def __init__(
            self,
            state: Any,
            g: float,
            h: float,
            parent: 'Node' = None,
            tree: str = None,  # TODO: remove! --> only for debugging
            leaf: Any = None  # TODO: remove! --> only for debugging
    ):
        self.g = g
        self.h = h
        self.state = state
        self.parent = parent
        self.tree = tree  # TODO: remove! --> only for debugging
        self.leaf = leaf  # TODO: remove! --> only for debugging

    @property
    def f(self) -> float:
        return self.g + self.h

    def __lt__(
            self,
            other
    ) -> bool:
        return self.f < other.f

    def __repr__(self):  # TODO: remove! --> only for debugging
        current_node = self
        path = ""
        while current_node is not None:
            if current_node.parent is not None:
                path = f"-{current_node.tree}({current_node.leaf.idx})" + path
            current_node = current_node.parent
        return "H" + path


class AStar:
    """Abstract A* class. Inheriting classes need to implement functions for
    goal check, path retraction and successor generation."""
    def __init__(
            self,
            initstate: Any,
            goalstate: Any,
            **kwargs):
        """

        """
        self.initstate = initstate
        self.goalstate = goalstate
        self.__dict__.update(kwargs)

        self.open = []
        self.closed = []

        self.reached = False

    def h(
            self,
            state: Any
    ) -> float:
        raise NotImplementedError

    def g(
            self,
            state: Any
    ) -> float:
        raise NotImplementedError

    def stepcost(
            self,
            state: Any
    ) -> float:
        raise NotImplementedError

    def generate_successors(
            self,
            node: Node
    ) -> List[Node]:
        raise NotImplementedError

    def isgoal(
            self,
            node: Node
    ) -> bool:
        """Check if current node is goal node"""
        raise NotImplementedError

    def retrace_path(
            self,
            node
    ) -> Any:
        current_node = node
        path = []
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        path.reverse()
        return path

    def search(self) -> Any:
        logger.debug(f'Searching path from {self.initstate} to {self.goalstate}')

        init = Node(state=self.initstate, g=0., h=self.h(self.initstate), parent=None)
        heapq.heappush(self.open, (init.f, init))

        while self.open:
            cf, cur_node = heapq.heappop(self.open)

            if self.isgoal(cur_node):
                self.reached = True

                try:
                    self.plot(cur_node)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')

                return self.retrace_path(cur_node)

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
            logger.warning(f'Could not find a path from {self.initstate} to {self.goalstate}')
            return [init]

    def plot(
            self,
            node
    ) -> None:
        raise NotImplementedError


class BiDirAStar:

    def __init__(
            self,
            fastar: type,
            bastar: type,
            initstate: Any,
            goalstate: Any,
            **kwargs
    ):
        self.initstate = initstate
        self.goalstate = goalstate
        self.f_astar = fastar(initstate, goalstate, **kwargs)
        self.b_astar = bastar(goalstate, initstate, **kwargs)
        self.reached = False

    def retrace_path(
            self,
            fnode: Node,
            bnode: Node
    ) -> List:
        fpath = self.f_astar.retrace_path(fnode)
        bpath = self.b_astar.retrace_path(bnode)
        bpath.reverse()

        path = fpath
        path.extend([p for p in bpath if p not in fpath])

        return path

    def common_node(
            self,
            fnode: Node,
            bnode: Node
    ) -> bool:

        # if current position of each fnode and bnode is identical
        if bnode.state.posx == fnode.state.posx and bnode.state.posy == fnode.state.posy:
            return True

        # ...or current position of forward node has reached goal state
        if fnode.state.posx == self.f_astar.goalstate.posx and fnode.state.posy == self.f_astar.goalstate.posy:
            return True

        # ...or current position of backward node has reached goal state
        if bnode.state.posx == self.b_astar.goalstate.posx and bnode.state.posy == self.b_astar.goalstate.posy:
            return True

        return False

    def search(self) -> Any:
        init = Node(state=self.f_astar.initstate, g=0., h=self.f_astar.h(self.f_astar.initstate), parent=None)
        goal = Node(state=self.b_astar.initstate, g=0., h=self.b_astar.h(self.b_astar.initstate), parent=None)
        heapq.heappush(self.f_astar.open, (init.f, init))
        heapq.heappush(self.b_astar.open, (goal.f, goal))

        while self.f_astar.open or self.b_astar.open:
            _, cur_fnode = heapq.heappop(self.f_astar.open)
            _, cur_bnode = heapq.heappop(self.b_astar.open)

            # if both paths have common node
            if self.common_node(cur_fnode, cur_bnode):
                self.reached = True

                try:
                    self.f_astar.plot(cur_fnode)
                    self.b_astar.plot(cur_bnode)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')

                return self.retrace_path(cur_fnode, cur_bnode)

            heapq.heappush(self.f_astar.closed, (cur_fnode.f, cur_fnode))
            heapq.heappush(self.b_astar.closed, (cur_bnode.f, cur_bnode))

            self.f_astar.goalstate = cur_bnode.state  # TODO: check!
            self.b_astar.goalstate = cur_fnode.state  # TODO: check!

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
            logger.warning(f'Could not find a path from {self.initstate} to {self.goalstate}')
            return [init]
