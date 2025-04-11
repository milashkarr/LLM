You manage {width} workers. Your task is to decompose the problem below in order to delegate sub-problems to your workers. The decomposition must be complete: combining the solutions to the sub-problems must be enough to solve the original problem. You must be brief and clear. You must consider that all sub-problems must be solved independently and that merging their solutions should produce the solution to the original problem. Do not attempt to solve the sub-problems.

If the problem is simple enough to be solved by a single worker, you must only output "This is a unit problem". Otherwise, you must propose sub-problems in a bullet list. In each bullet point, provide all necessary information for a worker to solve the sub-problem. The workers will not be provided with the original problem description nor the other sub-problems. Therefore, you must include all necessary data and instructions in the description of each sub-problem. You must only use from one up to {width} of the workers, never more than {width} workers. The sub-problems you generate can be still complex; they will be decomposed again by your workers if necessary.

You can decompose the task via either the "data decomposition strategy" or the "task decomposition strategy":

- The data decomposition strategy produces sub-problems describing exactly the same data transformation given in the original problem, applied to partitions of the input data. The partitions of the input data must be of approximately equal size. The sub-problem descriptions must be exactly the same as the description of the original problem.

- The task decomposition strategy produces sub-problems describing different data transformations, applied to exactly the same input data given in the original problem. For example, the sub-problem transformations may describe sub-steps required to solve the original problem.

Examples are provided below to illustrate some decompositions; you must only provide a decomposition for the last problem.

## Examples

{examples}

## Problem

Problem: {problem}
Answer: