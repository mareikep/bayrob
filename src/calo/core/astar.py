import heapq
import traceback
from datetime import datetime
from typing import List, Any

import dnutils

from calo.utils.constants import calologger, cs
from calo.utils.utils import dhms

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
        return self.g + self.h

    def __lt__(
            self,
            other
    ) -> bool:
        return self.f < other.f

    def __eq__(self, other):
        if other is None:
            return False
        return self.state == other.state and self.parent == other.parent

    def __str__(self) -> str:  # TODO: remove! --> only for debugging
        plen = 1
        path = ""
        current_node = self
        while current_node.parent is not None:
            plen += 1
            path = f" {repr(current_node.state)}{' ==>' if path else ''}{path}"
            current_node = current_node.parent
        return f"<NODE ({plen}): {path}>"

    def __repr__(self) -> str:
        plen = 1
        current_node = self
        while current_node.parent is not None:
            plen += 1
            current_node = current_node.parent
        return f"<Node ({plen}): {repr(self.state)}>"

    def __len__(self) -> int:
        cnt = 1
        cn = self
        while cn is not None:
            cnt += 1
            cn = cn.parent
        return cnt


class AStar:
    """Abstract A* class. Inheriting classes need to implement functions for
    goal check, path retraction and successor generation."""
    def __init__(
            self,
            initstate: Any,
            goal: Any,
            state_similarity: float = .9,
            goal_confidence: float = .01,
            **kwargs
    ):
        self.initstate = initstate
        self.goal = goal
        self._state_similarity = state_similarity
        self._goal_confidence = goal_confidence
        self.__dict__.update(kwargs)

        self.open = []
        self.closed = []

        self.reached = False
        self.init()

    def __str__(self):
        return f'A* [init: {self.initstate} | goal specification: {self.goal} ]'

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
        ts = datetime.now()
        logger.debug(f'{ts.strftime("%Y-%m-%d_%H:%M:%S")} Searching path from {self.initstate} to {self.goal}')
        plotme = False
        while self.open:
            cf, cur_node = heapq.heappop(self.open)
            if plotme:
                self.plot(cur_node)
            if self.isgoal(cur_node):
                tse = datetime.now()
                d, h, m, s = dhms(tse - ts)
                logger.info(f'{tse.strftime("%Y-%m-%d_%H:%M:%S")}: Found path from {self.initstate} to {self.goal}:\n'
                            f'{cs.join([str(p) for p in self.retrace_path(cur_node)])}.\n'
                            f'Took me only {d} days, {h} hours, {m} minutes and {s} seconds!')
                self.reached = True

                try:
                    self.plot(cur_node)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')
                except:
                    traceback.print_exc()

                return self.retrace_path(cur_node)

            heapq.heappush(self.closed, (cf, cur_node))
            successors = self.generate_successors(cur_node)

            for c in successors:
                if (c.f, c) in self.closed:
                    continue

                if (c.f, c) in self.open:
                    continue

                heapq.heappush(self.open, (c.f, c))

        if not self.reached:
            tse = datetime.now()
            d, h, m, s = dhms(tse-ts)
            logger.warning(f'{tse.strftime("%Y-%m-%d_%H:%M:%S")}: Could not find a path from {self.initstate} to {self.goal}. '
                           f'Took me {d} days, {h} hours, {m} minutes and {s} seconds to figure that out.')
            return []

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
            goal: Any,
            state_similarity: float = .9,
            goal_confidence: float = .01,
            **kwargs
    ):
        self.state_t = type(initstate)
        self.goal_t = type(goal)
        self.initstate = initstate
        self.goal = goal
        self.f_astar = f_astar(initstate, goal, **kwargs)
        self.b_astar = b_astar(initstate, goal, **kwargs)
        self._state_similarity = state_similarity
        self._goal_confidence = goal_confidence
        self.reached = False

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
        if bnode.state.similarity(fnode.state) >= self.state_similarity:
            return True

        # ...or current position of forward node has reached goal state
        if self.b_astar.isgoal(bnode):
            return True

        # ...or current position of backward node has reached goal state
        if self.f_astar.isgoal(fnode):
            return True

        return False

    def search(self) -> Any:
        if self.f_astar.open == []:
            init = Node(state=self.f_astar.initstate, g=0., h=self.f_astar.h(self.f_astar.initstate), parent=None)
            heapq.heappush(self.f_astar.open, (init.f, init))

        if self.b_astar.open == []:
            goal = Node(state=self.b_astar.goal, g=0., h=self.b_astar.h(self.b_astar.goal), parent=None)
            heapq.heappush(self.b_astar.open, (goal.f, goal))

        while self.f_astar.open or self.b_astar.open:
            _, cur_fnode = heapq.heappop(self.f_astar.open)
            _, cur_bnode = heapq.heappop(self.b_astar.open)

            # if both paths have common node
            if self.common_node(cur_fnode, cur_bnode):
                self.reached = True

                try:
                    self.f_astar.plot(cur_fnode)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')
                except:
                    logger.error('Could not plot result for unknown reasons. Skipping...')

                try:
                    self.b_astar.plot(cur_bnode)
                except NotImplementedError:
                    logger.info('Could not plot result. Function not implemented.')
                except:
                    logger.error('Could not plot result for unknown reasons. Skipping...')

                return self.retrace_path(cur_fnode, cur_bnode)

            heapq.heappush(self.f_astar.closed, (cur_fnode.f, cur_fnode))
            heapq.heappush(self.b_astar.closed, (cur_bnode.f, cur_bnode))

            self.f_astar.goal = cur_bnode.state  # TODO: check!
            self.b_astar.goal = cur_fnode.state  # TODO: check!

            successors = {
                self.f_astar: self.f_astar.generate_successors(cur_fnode),
                self.b_astar: self.b_astar.generate_successors(cur_bnode),
            }

            for astar in [self.f_astar, self.b_astar]:
                for c in successors[astar]:
                    if (c.f, c) in astar.closed:
                        continue

                    if (c.f, c) not in astar.open:
                        heapq.heappush(astar.open, (c.f, c))
                    else:
                        # retrieve the best current path
                        _, c_ = heapq.heappop(astar.open)

                        if c.g < c_.g:
                            heapq.heappush(astar.open, (c.f, c))
                        else:
                            heapq.heappush(astar.open, (c_.f, c_))

        if not self.reached:
            logger.warning(f'Could not find a path from {self.initstate} to {self.goal}')
            return [init]
