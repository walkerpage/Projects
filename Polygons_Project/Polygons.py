#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dwalkerpage
"""

import math

class Polygon:

    def __init__(self, n_vertices, circumradius):
        if n_vertices >= 3:
            self.n_vertices = n_vertices
            self.n_edges = self.n_vertices
            self.circumradius = circumradius
        else:
            raise ValueError('The number of vertices must be at least 3.')

    def __repr__(self):
        return (f'Polygon(n_vertices={self.n_vertices}, '
                f'circumradius={self.circumradius})')

    def edge_len(self):
        edge_len = (2 
                    * self.circumradius 
                    * math.sin(math.pi / self.n_vertices))
        return edge_len
    
    def apothem(self):
        apothem = (self.circumradius
                   * math.cos(math.pi / self.n_vertices))
        return apothem
    
    def surface_area(self):
        surface_area = (0.5
                        * self.n_vertices
                        * self.edge_len()
                        * self.apothem())
        return surface_area
    
    def perimeter(self):
        return self.n_vertices * self.edge_len()
    
    def interior_angle(self):
        interior_angle = ((self.n_vertices - 2)
                          * (180 / self.n_vertices))
        return interior_angle
    
    def __eq__(self, other):
        if isinstance(other, Polygon):
            return (self.n_vertices == other.n_vertices
                    and self.circumradius == other.circumradius)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Polygon):
            return self.n_vertices > other.n_vertices
        else:
            return NotImplemented

def test_polygon():
    rel_tol = 0.001
    abs_tol = 0.001

    try:
        p = Polygon(1, 10)
        assert False
    except ValueError:
        pass
    
    n_vertices = 3
    circumradius = 1
    p = Polygon(n_vertices, circumradius)
    assert str(p) == f'Polygon(n_vertices=3, circumradius=1)'
    assert p.n_vertices == n_vertices, (f'actual: {p.n_vertices}, '
                                        f' expected: {n_vertices}')
    assert p.n_edges == n_vertices, (f'actual: {p.n_edges}, '
                                     f' expected: {n_vertices}')
    assert p.circumradius == circumradius , (f'actual: {p.circumradius}, '
                                             f' expected: {circumradius}')
    assert math.isclose(p.interior_angle(), 60,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.interior_angle()}, '
                                           f' expected: {60}')
    
    n_vertices = 4
    circumradius = 1
    p = Polygon(n_vertices, circumradius)
    assert math.isclose(p.interior_angle(), 90,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.interior_angle()}, '
                                           f' expected: {90}')
    assert math.isclose(p.surface_area(), 2.0,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.surface_area()}, '
                                           f' expected: {2.0}')
    assert math.isclose(p.edge_len(), math.sqrt(2),
                       rel_tol=rel_tol,
                       abs_tol=abs_tol), (f'actual: {p.edge_len()}, '
                                          f' expected: {math.sqrt(2)}')
    assert math.isclose(p.perimeter(), 4 * math.sqrt(2),
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.perimeter()}, '
                                           f' expected: {4 * math.sqrt(2)}')
    assert math.isclose(p.apothem(), 0.707,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.apothem()}, '
                                           f' expected: {0.707}')

    n_vertices = 6
    circumradius = 2
    p = Polygon(n_vertices, circumradius)
    assert math.isclose(p.interior_angle(), 120,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.interior_angle()}, '
                                           f' expected: {120}')
    assert math.isclose(p.surface_area(), 10.3923,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.surface_area()}, '
                                           f' expected: {10.3923}')
    assert math.isclose(p.edge_len(), 2.0,
                       rel_tol=rel_tol,
                       abs_tol=abs_tol), (f'actual: {p.edge_len()}, '
                                          f' expected: {2.0}')
    assert math.isclose(p.perimeter(), 12.0,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.perimeter()}, '
                                           f' expected: {12.0}')
    assert math.isclose(p.apothem(), 1.73205,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.apothem()}, '
                                           f' expected: {1.73205}')

    n_vertices = 12
    circumradius = 3
    p = Polygon(n_vertices, circumradius)
    assert math.isclose(p.interior_angle(), 150,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.interior_angle()}, '
                                           f' expected: {150}')
    assert math.isclose(p.surface_area(), 27,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.surface_area()}, '
                                           f' expected: {27}')
    assert math.isclose(p.edge_len(), 1.55291,
                       rel_tol=rel_tol,
                       abs_tol=abs_tol), (f'actual: {p.edge_len()}, '
                                          f' expected: {1.55291}')
    assert math.isclose(p.perimeter(), 18.635,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.perimeter()}, '
                                           f' expected: {18.635}')
    assert math.isclose(p.apothem(), 2.89778,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol), (f'actual: {p.apothem()}, '
                                           f' expected: {2.89778}')

    p1 = Polygon(3, 100)
    p2 = Polygon(10, 10)
    p3 = Polygon(15, 10)
    p4 = Polygon(15, 100)
    p5 = Polygon(15, 100)

    assert p2 > p1
    assert p2 < p3
    assert p3 != p4
    assert p1 != p4
    assert p4 == p5

class Polygons:

    def __init__(self, max_vertices, circumradius):
        if max_vertices < 3:
            raise ValueError('The max number of vertices must be at least 3.')
        else:
            self.max_vertices = max_vertices
            self.circumradius = circumradius
            self.polygons = [Polygon(n_vertices, circumradius)
                             for n_vertices in range(3, max_vertices+1)
                             ]

    def __repr__(self):
        return (f'Polygons(max_vertices={self.max_vertices}, '
                 'circumradius={self.circumradius})')
    
    def __len__(self):
        return self.max_vertices - 2
    
    def __getitem__(self, value):
        return self.polygons[value]
    
    def max_efficiency(self):
        efficiencies = [polygon.surface_area()/polygon.perimeter()
                        for polygon in self.polygons
                        ]
        max_efficiency = max(efficiencies)
        return self.polygons[efficiencies.index(max_efficiency)]

def test_polygons():
    try:
        p = Polygons(1, 10)
        assert False
    except ValueError:
        pass

    max_vertices = 10
    circumradius = 2
    p = Polygons(max_vertices, circumradius)
    assert p.max_vertices == max_vertices, (f'actual: {p.max_vertices}, '
                                            f'expected: {max_vertices}')
    assert p.circumradius == circumradius, (f'actual: {p.circumradius}, '
                                            f'expected: {circumradius}')
    assert p.polygons == [Polygon(n_vertices, circumradius)
                          for n_vertices in range(3, max_vertices+1)], \
                         (f'actual: {p.polygons}, '
                          f'expected: {[Polygon(n_vertices, circumradius) for n_vertices in range(3, max_vertices+1)]}')

    assert len(p) == 8, (f'actual: {len(p)}, '
                         f'expected: {8}')

    assert p[0] == Polygon(3, 2), (f'actual: {p[0]}, '
                                   f'expected: {Polygon(3, 2)}')
    assert p[-1] == Polygon(10, 2), (f'actual: {p[-1]}, '
                                     f'expected: {Polygon(10, 2)}')

if __name__ == '__main__':
    test_polygon()
    test_polygons()