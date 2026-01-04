# def cel_to_fer():
#     c = int(input("Enter temperature in celcius: "))
#     f = c * 9 / 5 + 32
#     return "Temperature in Farenhiet is: ", str(f)
#
#
# d = cel_to_fer()
# print(d)


# Python program to find the sum of natural using recursive function
#
# def recur_sum(n):
#    if n <= 1:
#        return n
#    else:
#        return n + recur_sum(n-1)
#
# # change this value for a different result
# num = int(input("Enter a number\n"))
#
# if num < 0:
#    print("Enter a positive number")
# else:
#    print("The sum is",recur_sum(num))

#
# n = int(input("Enter no of stars to print on 1st line\n"))
# for i in range(n):
#     print("*" * (n-i))

# Program to convert inches to cm
# def inch_to_cm():
#     inch = int(input("Enter inches\n"))
#     cm = inch * 2.54
#     print(inch,"inches = ",cm,"cm")
#
# inch_to_cm()

# def remove_and_strip(string,word):
#     newStr = string.replace(word, "")
#     return newStr.strip()
#
# this = "    Jatin is a good boy     "
# n = remove_and_strip(this,"Jatin")
# print(n)

# def multi_tab():
#     n = int(input("Enter a no\n"))
#     for i in range(0,11):
#         print(f"{n}*{i}={n*i}")
#
# multi_tab()
