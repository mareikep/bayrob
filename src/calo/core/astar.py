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
    ):
        self.g = g
        self.h = h
        self.state = state
        self.parent = parent

    @property
    def f(self) -> float:
        print(self.g, self.h)
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
            path = f" {repr(current_node.state)}{' ->' if path else ''}{path}"
            current_node = current_node.parent
        return f"<Node{path}>"


class AStar:
    """Abstract A* class. Inheriting classes need to implement functions for
    goal check, path retraction and successor generation."""
    def __init__(
            self,
            initstate: Any,
            goalstate: Any,
            state_similarity: float = .9,
            goal_confidence: float = .01,
            **kwargs
    ):
        self.initstate = initstate
        self.goalstate = goalstate
        self._state_similarity = state_similarity
        self._goal_confidence = goal_confidence
        self.__dict__.update(kwargs)

        self.open = []
        self.closed = []

        self.reached = False
        self.init()

    def __str__(self):
        return f'<A* init: {self.initstate}; goal: {self.goalstate}>'

    def __repr__(self):
        return str(self)

    def init(self):
        raise NotImplementedError

    @property
    def state_similarity(self) -> float:
        return self._state_similarity

    @state_similarity.setter
    def state_similarity(
            self,
            s: float
    ) -> None:
        self._state_similarity = s

    @property
    def goal_confidence(self) -> float:
        return self._goal_confidence

    @goal_confidence.setter
    def goal_confidence(
            self,
            c: float
    ) -> None:
        self._goal_confidence = c

    def h(
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

        init = self.open[0]

        while self.open:
            cf, cur_node = heapq.heappop(self.open)
            self.plot(self.retrace_path(cur_node))
            if self.isgoal(cur_node):
                self.reached = True

                try:
                    self.plot(cur_node)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')

                print(f'Num open {len(self.open)}, num closed {len(self.closed)}; total: {len(self.open) + len(self.closed)}')
                return self.retrace_path(cur_node)

            heapq.heappush(self.closed, (cf, cur_node))
            successors = self.generate_successors(cur_node)

            for c in successors:
                if any([c.state.similarity(s.state) > self.state_similarity for _, s in self.closed]):
                    continue

                heapq.heappush(self.open, (c.f, c))

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
            f_astar: type,
            b_astar: type,
            initstate: Any,
            goalstate: Any,
            state_similarity: float = .9,
            goal_confidence: float = .01,
            **kwargs
    ):
        self.initstate = initstate
        self.goalstate = goalstate
        self.f_astar = f_astar(initstate, goalstate, **kwargs)
        self.b_astar = b_astar(initstate, goalstate, **kwargs)
        self._state_similarity = state_similarity
        self._goal_confidence = goal_confidence
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
        if bnode.state.similarity(fnode.state) > self.state_similarity:
            # bnode.state.posx == fnode.state.posx and bnode.state.posy == fnode.state.posy:
            return True

        # ...or current position of forward node has reached goal state
        if fnode.state.similarity(self.f_astar.goalstate) > self.state_similarity:
            # if fnode.state.posx == self.f_astar.goalstate.posx and fnode.state.posy == self.f_astar.goalstate.posy:
            return True

        # ...or current position of backward node has reached goal state
        if bnode.state.similarity(self.b_astar.goalstate) > self.state_similarity:
            # if bnode.state.posx == self.b_astar.goalstate.posx and bnode.state.posy == self.b_astar.goalstate.posy:
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
