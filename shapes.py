import numpy as np

class Shapes:
    @staticmethod
    def circle(center_x, center_y, radius):
        def shape(x, y):
            return (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2

        return shape

    @staticmethod
    def rectangle(min_x, min_y, max_x, max_y):
        def shape(x, y):
            return min_x <= x <= max_x and min_y <= y <= max_y

        return shape

    @staticmethod
    def triangle(x1, y1, x2, y2, x3, y3):
        def shape(x, y):
            def sign(p1x, p1y, p2x, p2y, p3x, p3y):
                return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

            d1 = sign(x, y, x1, y1, x2, y2)
            d2 = sign(x, y, x2, y2, x3, y3)
            d3 = sign(x, y, x3, y3, x1, y1)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            return not (has_neg and has_pos)

        return shape

    @staticmethod
    def naca_airfoil(center_x: float, center_y: float, chord: float, angle_of_attack: float, naca_number: int | str = '23015'):
        """
        :param center_x: X-coordinate of the center of the airfoil
        :param center_y: Y-cooordinate of the center of the airfoil
        :param chord: Chord length of airfoil
        :param angle_of_attack: Angle of attack of airfoil (radians)
        :param naca_number: Naca number, default 23015
        :return: shape function
        """
        # Ensure naca_number is a string
        naca_number = str(naca_number)

        # Parse NACA number
        m = int(naca_number[0]) / 100  # maximum camber
        p = int(naca_number[1]) / 10  # location of maximum camber
        t = int(naca_number[2:]) / 100  # maximum thickness

        def shape(x, y):
            # Translate and rotate the point
            x_r = (x - center_x) * np.cos(angle_of_attack) + (y - center_y) * np.sin(angle_of_attack)
            y_r = -(x - center_x) * np.sin(angle_of_attack) + (y - center_y) * np.cos(angle_of_attack)

            if 0 <= x_r <= chord:
                x_c = x_r / chord

                # Calculate camber line
                if x_c <= p:
                    y_c = (m / (p ** 2)) * (2 * p * x_c - x_c ** 2)
                    dy_c = (2 * m / (p ** 2)) * (p - x_c)
                else:
                    y_c = (m / ((1 - p) ** 2)) * ((1 - 2 * p) + 2 * p * x_c - x_c ** 2)
                    dy_c = (2 * m / ((1 - p) ** 2)) * (p - x_c)

                # Calculate thickness distribution
                y_t = 5 * t * (0.2969 * np.sqrt(
                    x_c) - 0.1260 * x_c - 0.3516 * x_c ** 2 + 0.2843 * x_c ** 3 - 0.1015 * x_c ** 4)

                # Combine camber and thickness
                theta = np.arctan(dy_c)
                y_u = (y_c + y_t * np.cos(theta)) * chord
                y_l = (y_c - y_t * np.cos(theta)) * chord

                # Check if point is inside the airfoil
                return y_l <= y_r <= y_u
            else:
                return False

        return shape
