We have 2-dimensional square space and simple boundary conditions (edge
points have fixed temperatures). The objective is to find the temperature distribution within. The temperature of the interior depends upon the temperatures around it. We find the temperature distribution by dividing the area into a fine mesh of points, hi,j. The temperature at an inside point is calculated as the average of the temperatures of the four neighboring points.

Assume we have (n x n) points (including edge points), the initial situation is as follows:
* The edge points are fixed at 80F, except:
* Points (0, 10) to (0, 30) inclusive have temperature of 150F.
* All internal points are initialized to zero.