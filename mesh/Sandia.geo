
Point(1) = {0, 0, 0, 0.0001};
Point(2) = {0, 0.0036, 0, 0.0001};
Point(3) = {0, 0.0091, 0, 0.0001};
Point(4) = {0, 0.15, 0, 0.003};
Point(5) = {0.4, 0.15, 0, 0.003};
Point(6) = {0.4, 0, 0, 0.001};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(8) = {3, 4, 5, 6, 1, 2};
Plane Surface(8) = {8};
