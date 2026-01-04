# num = int(input("Enter the number "))
# for i in range(1,11):
#     print(str(num) + "x" + str(i) + "=" + str(num * i))

# l1 = ["Harry", "Soham", "Sachin", "Rahul"]
# for name in l1:
#     if name.startswith("S"):
#         print("Hello" + name)

# num = int(input("Enter a number: "))
# i = 1
# while i != 11:
#     print(f"{num} x {i}={num*i}")
#     i = i + 1

# num = int(input("Enter a number to check whether it is prime or not:- "))
# prime = True
# for i in range(2,num):
#     if (num%i == 0):
#         prime = False
#         break
#
# if prime:
#     print("The number is prime")
# else:
#     print("The number is not prime")


# num = int(input("Enter a number: "))
#
# if num < 0:
#     print("Enter a positive number\n")
# else:
#     sum = 0
#     # use while loop to iterate un till zero
#     while (num > 0):
#         sum += num
#         num -= 1
#     print("The sum is", sum)

#
# num = int(input("Enter a number: "))
# fact = 1
# for i in range(1,num+1):
#     fact = fact * i
#
# print(f"The factorial of {num} is {fact}")
#
# n = 4
# for i in range(4):
#     print("*" * (i+1))


# n = 3
# for i in range(3):
#     print(" " * (n-i-1), end="")
#     print("*" * (2*i+1), end="")
#     print(" " * (n-i-1))

# side = int(input("Please Enter side of a Square  : "))
#
# print("Hollow Square Star Pattern")
# for i in range(side):
#     for j in range(side):
#         if(i == 0 or i == side - 1 or j == 0 or j == side - 1):
#             print('*', end = '  ')
#         else:
#             print(' ', end = '  ')
#     print()

#
# num = int(input("Enter a number: "))
# i = 11
# while i != 0:
#     print(f"{num} x {i}={num*i}")
#     i = i - 1

# num = int(input("Enter a number "))
# for i in reversed(range(11)):
#     print(f"{num}x{i}={num*i}")


