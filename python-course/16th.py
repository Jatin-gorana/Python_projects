# name = str(input("Enter your name:- "))
# marks = int(input("Marks:- "))
# phone_number = int(input("Enter your phone number:- "))
# print("The name of the student is {}, his marks are {} and phone number is {}".format(name, marks, phone_number))

# l = [str(i*7) for i in range(1,11)]
# print(l)
# verticaltable = "\n".join(l)
# print(verticaltable)

# l = [1, 2, 3, 5, 10, 15, 34, 45, 50, 45]
# # def is_divby_5(num):
# #     return num%5==0
# # OR
# a = filter(lambda num: num % 5 == 0, l)
# # a = list(filter(is_divby_5, l))
# print(list(a))


from functools import reduce
l = [334,234,345,46,567,457,90]
# print(max(l))
a = reduce(max, l)  # reduce does this process sequentially.
print(a)