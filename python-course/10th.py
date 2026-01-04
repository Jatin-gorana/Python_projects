# class C2dvec():
#     def __init__(self,i,j):
#         self.icap = i
#         self.jcap = j
#
#     def __str__(self):
#         return f"{self.icap}i + {self.jcap}j"
#
# class C3dvec(C2dvec):
#     def __init__(self,i,j,k):
#         super().__init__(i,j)
#         self.kcap = k
#
#     def __str__(self):
#         return f"{self.icap}i + {self.jcap}j + {self.kcap}k"
#
#
# v2d = C2dvec(2,4)
# v3d = C3dvec(2,4,5)
# print(v2d)
# print(v3d)

# class Animals:
#     animaltype = "mammal"
#
#
# class Pets():
#     color = "white"
#
#
# class Dog(Pets):
#     @staticmethod
#     def bark():
#         print("Bow Bow!!!")
#
#
# d = Dog()
# d.bark()

# class Employee:
#     salary = 1000
#     increment = 1.5
#     @property
#     def salaryafterincrement(self):
#         return self.salary  * self.increment
#
#     @salaryafterincrement.setter
#     def salaryafterincrement(self, sai):
#         self.increment = sai/self.salary
#
# e = Employee()
# print(e.salaryafterincrement)
# print(e.increment)
# e.salaryafterincrement = 2000
# print(e.increment)

# (a+bi)(c+di) = (ac - bd) + (ad + bc)i
# class Complex:
#     def __init__(self, r, i):
#         self.real = r
#         self.imaginary = i

#     def __add__(self, c):
#         return Complex(self.real + c.real, self.imaginary + c.imaginary)

#     def __mul__(self, c):
#         mulreal = self.real * c.real - self.imaginary * c.imaginary
#         mulimg = self.real * c.imaginary + self.imaginary * c.real
#         return Complex(mulreal, mulimg)

#     def __str__(self):
#         return f"{self.real} + {self.imaginary}i"


# c1 = Complex(3, 4)
# c2 = Complex(36, 5)
# print(c1 * c2)

# class Vec():
#     def __init__(self, vec):
#         self.vec = vec

#     def __str__(self):
#         str1 = ""
#         index = 0
#         for i in self.vec:
#             str1 += f"{i}a{index} + "
#             index += 1
#         return str1[:-1]

#     def __add__(self, vec2):
#         newList = []
#         for i in range(len(self.vec)):
#             newList.append(self.vec[i] + vec2.vec[i])
#         return Vec(newList)

#     def __mul__(self, vec2):
#         sum = 0
#         for i in range(len(self.vec)):
#             sum += self.vec[i] * vec2.vec[i]
#             return sum


# '''n number of elements can be put in this vector'''
# v1 = Vec([1,4,6])   
# v2 = Vec([1,6,3])
# print(v1+v2)
# print(v1*v2)

# The Perfect Guess
import random
a = random.randint(1,100)
print("..................The Perfect Guess Game..................")
b = None
n = 1
i = 0
while(b!=a):
    b = int(input("Enter a number between 1 to 100 "))
    if b==a:
        print("Your guess is perfect!!!!!")
    elif b>a:
        print("Lower number please")
        n += 1
    elif b<a:
        print("Higher number please")
        n += 1
    else:
        print("Please enter in the range of 1 to 100")

print("You took ",n, "no of gueses")