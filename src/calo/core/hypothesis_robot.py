from __future__ import annotations

import math

from calo.core.hypothesis import Hypothesis
from jpt.base.intervals import ContinuousSet

from jpt.variables import VariableMap


class Hypothesis_Robot(Hypothesis):
    """Hypothesis for Robot Action example"""

    def __init__(self, idx, start, goal, steps=None):
        self.init = start
        self.startx = start['x_in']
        self.starty = start['y_in']
        self.startdirx = start['xdir_in']
        self.startdiry = start['ydir_in']
        if all([isinstance(x, ContinuousSet) for x in list(goal.values())]):
            self.goalx = goal['x_out'].lower + (goal['x_out'].upper - goal['x_out'].lower) / 2
            self.goaly = goal['y_out'].lower + (goal['y_out'].upper - goal['y_out'].lower) / 2
        else:
            self.goalx = goal['x_out']
            self.goaly = goal['y_out']
        super().__init__(idx, steps=steps)

    def __add__(self, other):
        pass

    def copy(self, idx) -> Hypothesis:
        start = {'x_in': self.startx, 'y_in': self.starty, 'xdir_in': self.startdirx, 'ydir_in': self.startdiry}
        goal = VariableMap([(k, v) for k, v in self.goal.items()])
        if idx is None:
            hyp_ = self.__class__(self.identifiers, start, goal)
        else:
            hyp_ = self.__class__([idx] + self.identifiers, start, goal)
        hyp_.steps = [s.copy() for s in self.steps]
        hyp_.goal = {k: v for k, v in self.goal.items()}
        return hyp_

    def g(self) -> float:
        # distance (Euclidean) travelled so far (from current position to goal)
        gcost = 0.
        lastx = self.startx
        lasty = self.starty
        for s in self.steps:
            if "M.tree" == s.tree:
                dx = lastx - s.leaf.value['x_out'].expectation()
                dy = lasty - s.leaf.value['y_out'].expectation()
                gcost += math.sqrt(dx ** 2 + dy ** 2)
                lastx = s.leaf.value['x_out'].expectation()
                lasty = s.leaf.value['y_out'].expectation()
            else:
                gcost += 1  # constant costs of 1 for each turn
        return gcost

    def h(self) -> float:
        # Euclidean distance from first move step (i.e. current position) to start node
        hcost = 0.
        for step in self.steps:
            if "x_in" in step.value and "y_in" in step.value:
                dx = self.startx - step.value['x_in'].expectation()
                dy = self.starty - step.value['y_in'].expectation()
                return math.sqrt(dx ** 2 + dy ** 2)

        return hcost

            # if not dirset and "xdir_in" in step.leaf.value and "ydir_in" in step.leaf.value:
            #     # difference in orientation from first step to start state
            #     rad = math.atan2(step.leaf.value['ydir_in'].expectation() - start_ydir, step.leaf.value['xdir_in'].expectation() - start_xdir)
            #     deg = abs(math.degrees(rad))
            #     orientationdiff += deg

    def plot(self) -> None:
        from matplotlib import pyplot as plt
        from matplotlib import patches
        fig, ax = plt.subplots()

        print(self.goal)
        # plot goal area
        goal = {k.name: v for k, v in self.goal.items()}
        ax.scatter(self.goalx, self.goaly, marker='^', color='green')
        ax.add_patch(patches.Rectangle((goal['x_out'].lower, goal['y_out'].lower), goal['x_out'].upper - goal['x_out'].lower, goal['y_out'].upper - goal['y_out'].lower, linewidth=1, color='green', alpha=.2))
        ax.annotate('GOAL', (self.goalx, self.goaly))

        # plot starting position
        ax.scatter(self.startx, self.starty, marker='^', color='hotpink')
        ax.quiver(self.startx, self.starty, self.startdirx, self.startdiry, color='hotpink', width=0.002)
        ax.annotate('START', (self.startx, self.starty))

        # plot expected position result of hypothesis (blue triangle)
        res_x = self.result['x_out'].result
        res_y = self.result['y_out'].result
        ax.scatter(res_x, res_y, marker='^', label=f'Result {self.id}', color='blue')

        if 'xdir_out' in self.result and 'ydir_out' in self.result:
            # plot expected direction result of hypothesis (thick blue arrow)
            res_dirx = self.result['xdir_out'].result
            res_diry = self.result['ydir_out'].result
            ax.quiver(res_x, res_y, res_dirx, res_diry, color='blue', width=0.003)


        # generate plot legend labels
        for i, s in enumerate(self.steps):
            if "M.tree" == s.tree:
                lbl = f'{i+1}: MOVE {s.leaf.value["numsteps"].expectation():.2f} STEPS (Leaf #{s.leaf.idx});\n '
                xmin = s.leaf.value["x_out"].quantile(.05)
                xmax = s.leaf.value["x_out"].quantile(.95)
                ymin = s.leaf.value["y_out"].quantile(.05)
                ymax = s.leaf.value["y_out"].quantile(.95)

                xexp = s.leaf.value["x_out"].expectation()
                yexp = s.leaf.value["y_out"].expectation()
                xdirexp = s.leaf.value["xdir_in"].expectation()
                ydirexp = s.leaf.value["ydir_in"].expectation()

                # plot expectation area and expectation value from move step (blue rectangle, red star)
                ax.add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, color='cornflowerblue', alpha=.2))
                ax.scatter(xexp, yexp, marker='*', label=lbl, c='red')
                ax.annotate(f'Expectation Step {i+1} (Leaf #{s.leaf.idx})', (xmax, ymin))
                #-------------------------------------------------------------------------------------------------------

                xpathmin = s.leaf.path["x_in"].lower
                xpathmax = s.leaf.path["x_in"].upper
                ypathmin = s.leaf.path["y_in"].lower
                ypathmax = s.leaf.path["y_in"].upper
                # plot precondition area (yellow rectangle)
                ax.add_patch(patches.Rectangle((xpathmin, ypathmin), xpathmax-xpathmin, ypathmax-ypathmin, linewidth=1, color='yellow', alpha=.2))
                ax.annotate(f'Precondition Step {i+1} (Leaf #{s.leaf.idx})', (xpathmax, ypathmin))

                # plot expected move direction (red arrow)
                ax.quiver(xexp, yexp, xdirexp, ydirexp, color='red', linestyle='dotted', width=0.001)

                #-------------------------------------------------------------------------------------------------------
            else:
                lbl = f'{i+1}: TURN {s.leaf.value["angle"].expectation():.2f} DEGREES (Leaf #{s.leaf.idx});\n '
                xdirmin = s.leaf.value["xdir_out"].quantile(.05)
                xdirmax = s.leaf.value["xdir_out"].quantile(.95)
                ydirmin = s.leaf.value["ydir_out"].quantile(.05)
                ydirmax = s.leaf.value["ydir_out"].quantile(.95)

                xdirexp = s.leaf.value["xdir_out"].expectation()
                ydirexp = s.leaf.value["ydir_out"].expectation()

                xdirin = s.leaf.value["xdir_in"].expectation()
                ydirin = s.leaf.value["ydir_in"].expectation()



                for i_, s_ in enumerate(self.steps[i:]):
                    if 'M.tree' == s_.tree:
                        xexp = s_.leaf.value["x_out"].expectation()
                        yexp = s_.leaf.value["y_out"].expectation()
                        # plot expected turn directions (green lower exp, grey exp, red upper exp)
                        # ax.quiver(xexp, yexp, xdirmin, ydirmin, color='green', width=0.001)
                        ax.quiver(xexp, yexp, xdirexp, ydirexp, color='darkorchid', width=0.002, label=lbl)
                        ax.quiver(xexp, yexp, xdirin, ydirin, color='darkorchid', width=0.001, linestyle='dashed', label=lbl)
                        # ax.quiver(xexp, yexp, xdirmax, ydirmax, color='red', width=0.001)
                        break

        plt.grid()
        plt.legend()
        plt.show()
