# class Programmer:
#     company = "Microsoft"
#
#     def __init__(self, name, product):
#         self.name = name
#         self.product = product
#
#     def getInfo(self):
#         print(f"The name of {self.company} programmer is {self.name} and the product is {self.product}")
#
#
# harry = Programmer("Harry", "Skype")
# alka = Programmer("Jatin", "Props")
# alka.getInfo()

#
# class Calculator():
#     def __init__(self, num):
#         self.number = num
#
#     def sq(self):
#         print(f"The square of {self.number} is {self.number ** 2}")
#
#     def cube(self):
#         print(f"The cube of {self.number} is {self.number ** 3}")
#
#     def sqrt(self):
#         print(f"The squareroot of {self.number} is {self.number ** 0.5}")


# a = Calculator(3)
# a.sq()
# a.sqrt()
# a.cube()
#
# class Sample():
#     a = "Jatin"
#
#
# obj = Sample()
# obj.a = "Harry"
#
# print(Sample.a)
# print(obj.a)


class Train:
    def __init__(self, name, fare, seats):
        self.name = name
        self.fare = fare
        self.seats = seats

    def getInfo(self):
        print("***********************")
        print(f"The name of train is {self.name}")
        print(f"The seats available in the train are {self.seats}")

    def getFareInfo(self):
        print(f"The price of the ticket is: Rs.{self.fare}")

    def bookTickets(self):
        if self.seats>0:
            print(f"Your ticket has been booked! Your seat number is {self.seats}")
            self.seats = self.seats - 1
        else:
            print("Sorry this train is full! Kindly try in tatkal")
    def cancelTicket(self):
        print("Your ticket has been cancelled!")
        self.seats = self.seats + 1


intercity = Train("Intercity Express: 14290", 390, 120)
intercity.getInfo()
intercity.bookTickets()
intercity.getInfo()
intercity.getFareInfo()

intercity.cancelTicket()
intercity.getInfo()