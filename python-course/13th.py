# def readffile(filename):
#     try:
#         with open(filename, "r") as f:
#             print(f.read())
#     except FileNotFoundError:
#         print(f"{filename} file does not exist.")

# readffile("1.txt")
# readffile("2.txt")

#write a program to use enumerate function.

# listt = [1,2,3,4,5,5,6,77,7]
# for i,item in enumerate(listt):
#     if i == 2 or i == 4 or i == 6:
#         print(f"{i + 1}th element is {item}")

# num = int(input("Enter a number: "))

# table = [num * i for i in range(1,11)]
# print(table)

# a = int(input("Enter a number: "))
# b = int(input("Enter a number: "))
# try:
#     print(a/b)
# except ZeroDivisionError:
#     print("∞")

num = int(input("Enter a number: "))
table = [num * i for i in range(1,11)]
print(table)

with open("tables.txt", "a") as f:
    f.write(str(table))
    f.write("\n")