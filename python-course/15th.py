# # Lambda functions:- One liner functions.
# func = lambda a : a + 5
# square = lambda X : X*X
# sum = lambda a, b, c : a+b+c

# X = 3
# print(func(X))
# print(square(X))
# print(sum(X, 1, 3))

# name = "JAtin"
# channel = "TechTricks"
# type = "Technical"
# a = "This is {} and his channel is {} and it is type of {}".format(name,channel,type)
# print(a)

# def func(num):
#     return num*num

# l = [1,2,4]
# # Method 1:-
# l2 = []
# for item in l:
#     l2.append(func(item))
# print(l2)

# # Method 2:-
# print(list(map(func, l)))   # list is written to typecast map function into list.


# def greater_than_5(num):
#     if num>5:
#         return True
#     else:
#         return False
    
# l = [1,2,3,4,5,6,6,7,887,67,56,46]
# print(list(filter(greater_than_5, l)))

# from functools import reduce
# sum = lambda a, b: a+b
# l = [1,2,3,4]
# value = reduce(sum, l)
# print(value)


# The similarities in map,filter and reduce is these all contain 2 arguments 
# First is the function name
# Second is the iterator or say list name it may be anything.