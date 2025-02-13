# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        center = alien.get_centroid()
        radius = alien.get_width()
        
        for wall in walls:
            distance = point_segment_distance(center, ((wall[0], wall[1]), (wall[2], wall[3])))
            if distance <= radius:
                return True
    else:
        head, tail = alien.get_head_and_tail()
        alien_segment = (head, tail)
        width = alien.get_width()
        
        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            if do_segments_intersect(alien_segment, wall_segment):
                return True
            if segment_distance(alien_segment, wall_segment) <= width:
                return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    width, height = window
    if alien.is_circle():
       center = alien.get_centroid()
       radius = alien.get_width()
       if center[0] - radius < 0 or center[0] + radius > width or center[1] - radius < 0 or center[1] + radius > height:
           return False
    else:
        head, tail = alien.get_head_and_tail()
        alien_width = alien.get_width()
        if (head[0] - alien_width <= 0 or head[0] + alien_width >= width or
           tail[0] - alien_width <= 0 or tail[0] + alien_width >= width or
           head[1] - alien_width <= 0 or head[1] + alien_width >= height or
           tail[1] - alien_width <= 0 or tail[1] + alien_width >= height):
           return False
    return True


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    if polygon[0][0] == polygon[1][0] and polygon[1][0] == polygon[2][0] and polygon[2][0] == polygon[3][0]:
        if point[0] == polygon[0][0] and point[1]<= max(polygon[i][1] for i in range(4)) and point[1] >= min(polygon[i][1] for i in range(4)):
            return True
        return False
    if polygon[0][1] == polygon[1][1] and polygon[1][1] == polygon[2][1] and polygon[2][1] == polygon[3][1]:
        if point[1] == polygon[0][1] and point[0]<= max(polygon[i][0] for i in range(4)) and point[0] >= min(polygon[i][0] for i in range(4)):
            return True 
        return False  
    for i in range(4):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % 4]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        if edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0] < 0:
            return False
    return True


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if does_alien_touch_wall(alien, walls):
        return True
    current_point = alien.get_centroid()
    alien.set_alien_pos(waypoint)
    if does_alien_touch_wall(alien, walls):
        return True
    alien.set_alien_pos(current_point)
    if alien.is_circle():
        radius = alien.get_width()
        for wall in walls:
            if do_segments_intersect((current_point, waypoint), ((wall[0], wall[1]), (wall[2], wall[3]))):
                return True
            distance_to_wall = point_segment_distance((wall[0], wall[1]), (current_point, waypoint))
            if distance_to_wall <= radius:
                return True
            distance_to_wall = point_segment_distance((wall[2], wall[3]), (current_point, waypoint))
            if distance_to_wall <= radius:
                return True
    else:
        head, tail = alien.get_head_and_tail()
        alien_width = alien.get_width()

        direction = (waypoint[0] - current_point[0], waypoint[1] - current_point[1])
        future_head = (head[0] + direction[0], head[1] + direction[1])
        future_tail = (tail[0] + direction[0], tail[1] + direction[1])

        parallelogram = [head, tail, future_tail, future_head]

        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            for i in range(len(parallelogram)):
                segment = (parallelogram[i], parallelogram[(i + 1) % 4])
                if do_segments_intersect(segment, wall_segment):
                    return True
                if point_segment_distance((wall[0], wall[1]), (parallelogram[i], parallelogram[(i+1) % 4])) <= alien_width:
                    return True
                if point_segment_distance((wall[2], wall[3]), (parallelogram[i], parallelogram[(i+1) % 4])) <= alien_width:
                    return True

        for wall in walls:
            if is_point_in_polygon((wall[0], wall[1]), parallelogram) or is_point_in_polygon((wall[2], wall[3]), parallelogram):
                return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    px, py = p
    (x1, y1), (x2, y2) = s
    dx, dy = px - x1, py - y1
    segment_length_squared = (x2 - x1)**2 + (y2 - y1)**2
    
    if segment_length_squared == 0:
        return np.sqrt(dx**2 + dy**2)
    
    t = max(0, min(1, (dx * (x2 - x1) + dy * (y2 - y1)) / segment_length_squared))
    x_proj = x1 + t * (x2 - x1)
    y_proj = y1 + t * (y2 - y1)
    return np.sqrt((px - x_proj)**2 + (py - y_proj)**2)

def on_segment(p, q, r):
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    Ax, Ay = s1[0][0], s1[0][1]
    Bx, By = s1[1][0], s1[1][1]
    Cx, Cy = s2[0][0], s2[0][1]
    Dx, Dy = s2[1][0], s2[1][1]

    sin_AC_AD = (Cx - Ax) * (Dy - Ay) - (Cy - Ay) * (Dx - Ax)
    sin_BC_BD = (Cx - Bx) * (Dy - By) - (Cy - By) * (Dx - Bx)
    sin_CA_CB = (Ax - Cx) * (By - Cy) - (Ay - Cy) * (Bx - Cx)
    sin_DA_DB = (Ax - Dx) * (By - Dy) - (Ay - Dy) * (Bx - Dx)
    if sin_AC_AD * sin_BC_BD <= 0 and sin_CA_CB * sin_DA_DB <= 0:
        # Special case: Check for collinearity
        if sin_AC_AD == 0 and sin_BC_BD == 0 and sin_CA_CB == 0 and sin_DA_DB == 0:
            # The segments are collinear; check if they overlap
            if (on_segment((Ax, Ay), (Cx, Cy), (Bx, By)) or
                on_segment((Ax, Ay), (Dx, Dy), (Bx, By)) or
                on_segment((Cx, Cy), (Ax, Ay), (Dx, Dy)) or
                on_segment((Cx, Cy), (Bx, By), (Dx, Dy))):
                return True
            return False
        return True
    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0.0
    distances = [
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1)
    ]
    return min(distances)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
