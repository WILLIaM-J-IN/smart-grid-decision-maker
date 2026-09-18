import math


class Shape:
    def getarea(self):
        pass

    def getperim(self):
        pass


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def getarea(self):
        return self.width * self.height

    def getperim(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def getarea(self):
        return math.pi * (self.radius ** 2)

    def getperim(self):
        return 2 * math.pi * self.radius


if __name__ == "__main__":
    my_rect = Rectangle(5, 10)
    my_circle = Circle(4)

    print("矩形宽=5,高=10")X
    print("矩形面积:")
    print(my_rect.getarea())
    print("矩形周长:")
    print(my_rect.getperim())

    print("圆形半径=4")
    print("圆形面积:")
    print(round(my_circle.getarea(), 2))
    print("圆形周长:")
    print(round(my_circle.getperim(), 2))